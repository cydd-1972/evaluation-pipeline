"""v4 plus add: fact extraction + similarity-gated update judge + incremental UPSERT."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import asyncpg

from core.infra.checkpoint import load_json_list, write_json_list
from core.infra.data_loader import load_locomo_dataset
from core.infra.db import (
    apply_memory_incremental_writes,
    backfill_memory_embeddings,
    list_memories_for_user,
    provision_workspace_database,
)
from core.infra.embedding import EmbeddingClient
from core.infra.ids import build_conversation_user_id
from core.infra.llm_client import PipelineLLM
from core.infra.memory_ops import apply_global_memory_delta
from core.infra.progress import ProgressBar
from core.infra.transcript import (
    format_session_structured_json,
    format_sessions_structured_json,
    iter_sessions,
)
from core.paths import EVAL_PIPELINE_ROOT
from v1_mem0.add import _resolve_memory_prompt_limit, _run_batched, _truncate_memory_for_prompt
from v3_global.add import (
    _GlobalConvState,
    _conv_fully_done,
    _fallback_memory_from_session,
    _memory_items_from_db,
    _serialize_memory_snapshot,
    _upsert_snapshot,
    resolve_memory_prompt_path,
)
from v4_global.add import (
    _coerce_fact_items,
    _load_template,
    _normalize_anchor_time_iso,
    _resolved_time_value,
    _similarity,
    _token_set,
)

DEFAULT_UPDATE_JUDGE_PROMPT_PATH = EVAL_PIPELINE_ROOT / "prompts" / "memory_update_judge_v4_plus.txt"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_update_judge_prompt_path(raw: str | Path | None) -> Path:
    if raw is None or not str(raw).strip():
        return DEFAULT_UPDATE_JUDGE_PROMPT_PATH
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return EVAL_PIPELINE_ROOT / "prompts" / path.name
    return EVAL_PIPELINE_ROOT / path


def _top_candidates(
    *,
    old_by_id: dict[str, dict[str, Any]],
    fact_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for memory_id, old_item in old_by_id.items():
        old_text = str(old_item.get("text") or "").strip()
        if not old_text:
            continue
        ranked.append(
            {
                "id": memory_id,
                "text": old_text,
                "anchor_time": str(old_item.get("anchor_time") or "").strip(),
                "similarity": _similarity(fact_text, old_text),
            }
        )
    ranked.sort(key=lambda item: (-float(item["similarity"]), str(item["id"])))
    return ranked[: max(1, int(top_k or 1))]


def _heuristic_midrange_decision(
    *,
    fact_text: str,
    fact_type: str,
    candidate: dict[str, Any],
    session_anchor_time: str,
) -> tuple[str, str | None, str | None]:
    old_text = str(candidate.get("text") or "")
    old_anchor = str(candidate.get("anchor_time") or "").strip() or session_anchor_time
    old_resolved = _resolved_time_value(old_text, old_anchor) if old_anchor else ""
    new_resolved = _resolved_time_value(fact_text, session_anchor_time) if session_anchor_time else ""

    if old_resolved and new_resolved and old_resolved != new_resolved:
        return "ADD", None, None

    score = float(candidate.get("similarity") or 0.0)
    old_tokens = _token_set(old_text)
    new_tokens = _token_set(fact_text)
    is_more_specific = score >= 0.75 and len(new_tokens - old_tokens) >= 2
    has_time_addition = not old_resolved and bool(new_resolved)
    is_plan = fact_type == "plan"
    if is_more_specific or has_time_addition or is_plan:
        return "UPDATE", str(candidate.get("id") or "").strip() or None, fact_text
    return "ADD", None, None


def _judge_midrange_fact(
    llm: PipelineLLM,
    *,
    fact_text: str,
    fact_type: str,
    session_anchor_time: str,
    candidates: list[dict[str, Any]],
    update_judge_template: str,
) -> dict[str, Any]:
    payload = {
        "new_fact": fact_text,
        "fact_type": fact_type,
        "session_anchor_time": session_anchor_time,
        "candidate_memories": [
            {
                "id": str(item.get("id") or "").strip(),
                "text": str(item.get("text") or "").strip(),
                "anchor_time": str(item.get("anchor_time") or "").strip(),
                "similarity": round(float(item.get("similarity") or 0.0), 4),
            }
            for item in candidates
        ],
    }
    prompt = update_judge_template.format(input_json=json.dumps(payload, ensure_ascii=False, indent=2))
    judged, call_meta = llm.chat_json_object_with_meta(prompt, required_key="decision", max_attempts=4)
    decision = str(judged.get("decision") or "").strip().upper()
    target_id = str(judged.get("target_id") or "").strip() or None
    merged_text = str(judged.get("merged_text") or "").strip() or None
    if decision not in {"NONE", "UPDATE", "ADD"}:
        raise ValueError(f"unsupported decision: {decision!r}")
    return {
        "decision": decision,
        "target_id": target_id,
        "merged_text": merged_text,
        "reason": str(judged.get("reason") or "").strip(),
        "raw_output": str(call_meta.get("raw_text") or ""),
        "response_format": str(call_meta.get("response_format") or ""),
    }


def _decide_ops_from_facts_v4_plus(
    llm: PipelineLLM,
    *,
    old_memory: list[dict[str, Any]],
    new_facts: list[dict[str, str]],
    session_anchor_time: str,
    update_judge_template: str,
    none_similarity_threshold: float,
    add_similarity_threshold: float,
    update_candidate_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    used_ids = {str(item.get("id") or "").strip() for item in old_memory if str(item.get("id") or "").strip()}
    known_texts = {str(item.get("text") or "").strip().lower() for item in old_memory if str(item.get("text") or "").strip()}
    old_by_id = {
        str(item.get("id") or "").strip(): item
        for item in old_memory
        if str(item.get("id") or "").strip() and str(item.get("text") or "").strip()
    }
    decision_traces: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "facts_total": len(new_facts),
        "ops_add": 0,
        "ops_update": 0,
        "ops_none": 0,
        "midrange_llm_calls": 0,
        "midrange_llm_update": 0,
        "midrange_llm_add": 0,
        "midrange_llm_none": 0,
        "midrange_llm_fallbacks": 0,
        "slot_aggregates_disabled": True,
        "update_judge_model": llm.model,
        "update_none_similarity_threshold": none_similarity_threshold,
        "update_add_similarity_threshold": add_similarity_threshold,
        "update_candidate_top_k": update_candidate_top_k,
    }

    prepared: list[dict[str, Any]] = []
    for fact_item in new_facts:
        fact_text = str(fact_item.get("fact") or "").strip()
        fact_type = str(fact_item.get("type") or "event").strip().lower()
        if not fact_text:
            continue
        candidates = _top_candidates(old_by_id=old_by_id, fact_text=fact_text, top_k=update_candidate_top_k)
        top_score = float(candidates[0]["similarity"]) if candidates else 0.0
        prepared.append(
            {
                "fact_text": fact_text,
                "fact_type": fact_type,
                "candidates": candidates,
                "top_score": top_score,
            }
        )

    prepared.sort(key=lambda item: (-float(item["top_score"]), item["fact_text"].lower()))
    claimed_update_ids: set[str] = set()
    ops: list[dict[str, Any]] = []

    for item in prepared:
        fact_text = str(item["fact_text"])
        fact_type = str(item["fact_type"])
        candidates = list(item["candidates"])
        top_score = float(item["top_score"])

        if fact_text.lower() in known_texts:
            meta["ops_none"] += 1
            decision_traces.append(
                {
                    "fact_text": fact_text,
                    "fact_type": fact_type,
                    "top_score": top_score,
                    "decision": "NONE",
                    "decision_source": "known_text_duplicate",
                    "candidates": candidates,
                }
            )
            continue

        if top_score >= none_similarity_threshold:
            meta["ops_none"] += 1
            decision_traces.append(
                {
                    "fact_text": fact_text,
                    "fact_type": fact_type,
                    "top_score": top_score,
                    "decision": "NONE",
                    "decision_source": "similarity_threshold_none",
                    "candidates": candidates,
                }
            )
            continue

        if not candidates or top_score <= add_similarity_threshold:
            new_id = _next_numeric_id(used_ids)
            used_ids.add(new_id)
            ops.append({"id": new_id, "text": fact_text, "event": "ADD", "anchor_time": session_anchor_time})
            known_texts.add(fact_text.lower())
            meta["ops_add"] += 1
            decision_traces.append(
                {
                    "fact_text": fact_text,
                    "fact_type": fact_type,
                    "top_score": top_score,
                    "decision": "ADD",
                    "decision_source": "similarity_threshold_add",
                    "target_id": new_id,
                    "candidates": candidates,
                }
            )
            continue

        chosen = None
        try:
            chosen = _judge_midrange_fact(
                llm,
                fact_text=fact_text,
                fact_type=fact_type,
                session_anchor_time=session_anchor_time,
                candidates=candidates,
                update_judge_template=update_judge_template,
            )
            meta["midrange_llm_calls"] += 1
        except ValueError:
            meta["midrange_llm_fallbacks"] += 1
            decision, target_id, merged_text = _heuristic_midrange_decision(
                fact_text=fact_text,
                fact_type=fact_type,
                candidate=candidates[0],
                session_anchor_time=session_anchor_time,
            )
            chosen = {
                "decision": decision,
                "target_id": target_id,
                "merged_text": merged_text,
                "reason": "heuristic_fallback",
                "raw_output": "",
                "response_format": "",
            }

        decision = str(chosen.get("decision") or "").upper()
        target_id = str(chosen.get("target_id") or "").strip() or None
        merged_text = str(chosen.get("merged_text") or "").strip() or fact_text

        if decision == "NONE":
            meta["ops_none"] += 1
            meta["midrange_llm_none"] += 1
            decision_traces.append(
                {
                    "fact_text": fact_text,
                    "fact_type": fact_type,
                    "top_score": top_score,
                    "decision": "NONE",
                    "decision_source": "midrange_llm",
                    "target_id": target_id,
                    "merged_text": merged_text,
                    "reason": str(chosen.get("reason") or ""),
                    "raw_output": str(chosen.get("raw_output") or ""),
                    "response_format": str(chosen.get("response_format") or ""),
                    "candidates": candidates,
                }
            )
            continue

        if decision == "UPDATE":
            candidate_by_id = {str(c.get("id") or "").strip(): c for c in candidates}
            if target_id not in candidate_by_id or target_id in claimed_update_ids:
                decision = "ADD"
            else:
                old_item = old_by_id.get(target_id or "")
                old_text = str(old_item.get("text") or "").strip() if old_item else ""
                old_anchor = str(old_item.get("anchor_time") or "").strip() if old_item else ""
                update_text = merged_text or fact_text
                if not update_text or update_text.lower() == old_text.lower():
                    meta["ops_none"] += 1
                    meta["midrange_llm_none"] += 1
                    continue
                ops.append(
                    {
                        "id": target_id,
                        "text": update_text,
                        "event": "UPDATE",
                        "old_memory": old_text,
                        "anchor_time": old_anchor or session_anchor_time,
                    }
                )
                claimed_update_ids.add(target_id)
                known_texts.add(update_text.lower())
                meta["ops_update"] += 1
                meta["midrange_llm_update"] += 1
                decision_traces.append(
                    {
                        "fact_text": fact_text,
                        "fact_type": fact_type,
                        "top_score": top_score,
                        "decision": "UPDATE",
                        "decision_source": "midrange_llm",
                        "target_id": target_id,
                        "merged_text": update_text,
                        "reason": str(chosen.get("reason") or ""),
                        "raw_output": str(chosen.get("raw_output") or ""),
                        "response_format": str(chosen.get("response_format") or ""),
                        "candidates": candidates,
                    }
                )
                continue

        new_id = _next_numeric_id(used_ids)
        used_ids.add(new_id)
        ops.append({"id": new_id, "text": fact_text, "event": "ADD", "anchor_time": session_anchor_time})
        known_texts.add(fact_text.lower())
        meta["ops_add"] += 1
        meta["midrange_llm_add"] += 1
        decision_traces.append(
            {
                "fact_text": fact_text,
                "fact_type": fact_type,
                "top_score": top_score,
                "decision": "ADD",
                "decision_source": "midrange_llm" if str(chosen.get("reason") or "") != "heuristic_fallback" else "heuristic_fallback",
                "target_id": new_id,
                "merged_text": fact_text,
                "reason": str(chosen.get("reason") or ""),
                "raw_output": str(chosen.get("raw_output") or ""),
                "response_format": str(chosen.get("response_format") or ""),
                "candidates": candidates,
            }
        )

    return ops, meta, decision_traces


def _next_numeric_id(used: set[str]) -> str:
    i = 0
    while True:
        candidate = str(i)
        if candidate not in used:
            return candidate
        i += 1


def _decide_global_memory_v4_plus_sync(
    llm: PipelineLLM,
    *,
    speaker_a: str,
    speaker_b: str,
    old_memory: list[dict[str, Any]],
    history_sessions: list[Any],
    current_session: Any,
    memory_template: str,
    update_judge_template: str,
    memory_prompt_max_items: int | None = None,
    none_similarity_threshold: float = 0.92,
    add_similarity_threshold: float = 0.55,
    update_candidate_top_k: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    meta: dict[str, Any] = {"add_write_mode": "incremental", "slot_aggregates_disabled": True}
    trace: dict[str, Any] = {
        "llm_model": llm.model,
        "memory_before_count": len(old_memory),
        "memory_before": _serialize_memory_snapshot(old_memory),
    }
    old_trim = _truncate_memory_for_prompt(old_memory, limit=memory_prompt_max_items)
    history_json = format_sessions_structured_json(history_sessions)
    current_json = format_session_structured_json(current_session)
    prompt = memory_template.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        old_memory_json=json.dumps(old_trim, ensure_ascii=False, indent=2),
        history_sessions_json=history_json,
        current_session_json=current_json,
    )
    try:
        session_time_raw = str(getattr(current_session, "date_time", "") or "").strip()
        session_anchor_time, used_llm_time = _normalize_anchor_time_iso(llm, session_time_raw=session_time_raw)
        if used_llm_time:
            meta["anchor_time_llm_fallback"] = True

        payload, extract_meta = llm.chat_json_object_with_meta(prompt, required_key="facts", max_attempts=6)
        facts = _coerce_fact_items(payload)
        meta["memory_prompt"] = "extract"
        meta["facts_extracted"] = len(facts)
        trace["extract"] = {
            "raw_output": str(extract_meta.get("raw_text") or ""),
            "response_format": str(extract_meta.get("response_format") or ""),
            "parsed_payload": payload,
            "facts": facts,
            "session_anchor_time": session_anchor_time,
        }
        ops, op_stats, decision_traces = _decide_ops_from_facts_v4_plus(
            llm,
            old_memory=old_memory,
            new_facts=facts,
            session_anchor_time=session_anchor_time,
            update_judge_template=update_judge_template,
            none_similarity_threshold=none_similarity_threshold,
            add_similarity_threshold=add_similarity_threshold,
            update_candidate_top_k=update_candidate_top_k,
        )
        meta.update(op_stats)
        merged, db_writes, delta_stats = apply_global_memory_delta(old_memory, ops)
        meta.update(delta_stats)
        trace["decision_traces"] = decision_traces
        trace["operations"] = ops
        trace["db_writes"] = db_writes
        trace["memory_after_count"] = len(merged)
        trace["memory_after"] = _serialize_memory_snapshot(merged)
        return merged, db_writes, meta, trace
    except ValueError:
        pass

    meta["memory_fallback"] = True
    meta["failed_prompts"] = ["extract"]
    fallback = _fallback_memory_from_session(old_memory, current_session)
    old_ids = {str(item.get("id") or "") for item in old_memory}
    session_time_raw = str(getattr(current_session, "date_time", "") or "").strip()
    session_anchor_time, used_llm_time = _normalize_anchor_time_iso(llm, session_time_raw=session_time_raw)
    if used_llm_time:
        meta["anchor_time_llm_fallback"] = True
    delta_items = [
        {
            "id": str(item.get("id") or ""),
            "text": str(item.get("text") or ""),
            "event": "ADD",
            "anchor_time": session_anchor_time,
        }
        for item in fallback
        if str(item.get("id") or "") not in old_ids and str(item.get("text") or "").strip()
    ]
    merged, db_writes, delta_stats = apply_global_memory_delta(old_memory, delta_items)
    meta.update(delta_stats)
    trace["extract"] = {
        "raw_output": "",
        "response_format": "",
        "parsed_payload": {"facts": []},
        "facts": [],
        "session_anchor_time": session_anchor_time,
        "fallback": True,
    }
    trace["decision_traces"] = []
    trace["operations"] = delta_items
    trace["db_writes"] = db_writes
    trace["memory_after_count"] = len(merged)
    trace["memory_after"] = _serialize_memory_snapshot(merged)
    return merged, db_writes, meta, trace


async def run_add_global_v4_plus(
    *,
    dataset_path: str | Path,
    workspace_dir: Path,
    database_url: str | None,
    workspace_name: str,
    database_prefix: str,
    reset_database: bool,
    max_conversations: int | None,
    max_sessions_per_conversation: int | None,
    llm: PipelineLLM | None = None,
    progress_label: str | None = None,
    add_llm_concurrency: int = 1,
    add_history_window: int = 2,
    add_flush_per_session: bool = True,
    memory_prompt_path: str | Path | None = None,
    memory_prompt_max_items: int | None = None,
    backfill_embeddings: bool = True,
    update_judge_prompt_path: str | Path | None = None,
    none_similarity_threshold: float = 0.92,
    add_similarity_threshold: float = 0.55,
    update_candidate_top_k: int = 3,
) -> dict[str, Any]:
    resolved_llm = llm or PipelineLLM()
    prompt_path = resolve_memory_prompt_path(memory_prompt_path)
    memory_template = _load_template(prompt_path)
    judge_prompt_path = resolve_update_judge_prompt_path(update_judge_prompt_path)
    update_judge_template = _load_template(judge_prompt_path)
    prompt_limit = _resolve_memory_prompt_limit(memory_prompt_max_items, default=None)
    limit_label = "unlimited" if prompt_limit is None else str(prompt_limit)
    print(
        f"[add-global-v4-plus] memory_prompt={prompt_path.name} judge_prompt={judge_prompt_path.name} "
        f"memory_prompt_max_items={limit_label} write=incremental slot_aggregates=off "
        f"none>={none_similarity_threshold:.2f} add<={add_similarity_threshold:.2f} top_k={int(update_candidate_top_k)}",
        flush=True,
    )
    llm_batch = max(1, int(add_llm_concurrency or 1))
    history_window = max(0, int(add_history_window or 0))

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    add_trace_path = workspace_dir / "add_trace.jsonl"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add-global-v4-plus] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
            f"({len(snapshot_for_reset)} conversation(s)) - resume mode",
            flush=True,
        )

    db_url = await provision_workspace_database(
        workspace_name=workspace_name,
        database_prefix=database_prefix,
        base_database_url=database_url,
        reset=effective_reset,
    )
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "workspace.json").write_text(
        json.dumps(
            {
                "database_url": db_url,
                "workspace_name": workspace_name,
                "add_backend": "global_v4_plus",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    conversations = load_locomo_dataset(dataset_path, max_conversations=max_conversations)
    session_plans = [
        (conversation, session)
        for conversation in conversations
        for session in iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
    ]
    print(
        f"[add-global-v4-plus] conversations={len(conversations)} sessions={len(session_plans)} "
        f"history_window={history_window} persist_per_session={add_flush_per_session} llm_batch={llm_batch}",
        flush=True,
    )

    snapshot: list[dict[str, Any]] = []
    snapshot_by_conv: dict[int, dict[str, Any]] = {}
    if not effective_reset:
        snapshot = load_json_list(add_snapshot_path)
        snapshot_by_conv = {
            int(item.get("conversation_idx")): item
            for item in snapshot
            if item.get("conversation_idx") is not None
        }

    conn = await asyncpg.connect(db_url)
    conv_states: list[_GlobalConvState] = []
    for conversation in conversations:
        sessions = iter_sessions(conversation, max_sessions=max_sessions_per_conversation)
        existing = snapshot_by_conv.get(int(conversation.idx))
        if existing and _conv_fully_done(existing, len(sessions)):
            print(f"[add-global-v4-plus] conv{conversation.idx} skipped (complete in snapshot)", flush=True)
            continue
        start_idx = 0
        memory: list[dict[str, Any]] = []
        conv_entry = dict(existing) if isinstance(existing, dict) else {}
        if existing:
            start_idx = len(existing.get("sessions") or [])
            rows = await list_memories_for_user(conn, str(build_conversation_user_id(conversation.idx)))
            memory = _memory_items_from_db(rows)
            print(
                f"[add-global-v4-plus] conv{conversation.idx} resume from session {start_idx + 1}/{len(sessions)} "
                f"(memory={len(memory)})",
                flush=True,
            )
        state = _GlobalConvState(
            conversation=conversation,
            sessions=sessions,
            memory=memory,
            conv_entry=conv_entry,
            start_session_idx=start_idx,
        )
        state.conv_entry["add_backend"] = "global_v4_plus"
        state.conv_entry["add_write_mode"] = "incremental"
        conv_states.append(state)

    progress = ProgressBar("add-global-v4-plus", total=len(session_plans) or None, unit="session", label=progress_label)
    done_sessions = sum(
        len(iter_sessions(c, max_sessions=max_sessions_per_conversation))
        for c in conversations
        if snapshot_by_conv.get(int(c.idx)) and _conv_fully_done(
            snapshot_by_conv[int(c.idx)],
            len(iter_sessions(c, max_sessions=max_sessions_per_conversation)),
        )
    )
    if done_sessions:
        progress.update(done_sessions)

    try:
        max_session_count = max((len(cs.sessions) for cs in conv_states), default=0)
        for session_index in range(max_session_count):
            session_work: list[tuple[_GlobalConvState, Any]] = []
            for cs in conv_states:
                if session_index < cs.start_session_idx:
                    continue
                if session_index < len(cs.sessions):
                    session_work.append((cs, cs.sessions[session_index]))
            if not session_work:
                continue

            for cs, session in session_work:
                progress.set_description(f"add-global-v4-plus conv{cs.conversation.idx} session{session.index}")
                progress.set_postfix_str(f"{cs.conversation.speaker_a} & {cs.conversation.speaker_b}")

            async def _memory_job(
                item: tuple[_GlobalConvState, Any],
            ) -> tuple[_GlobalConvState, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
                cs, session = item
                hist_start = max(0, session_index - history_window)
                history_sessions = cs.sessions[hist_start:session_index]
                updated, db_writes, meta, trace = await asyncio.to_thread(
                    _decide_global_memory_v4_plus_sync,
                    resolved_llm,
                    speaker_a=cs.conversation.speaker_a,
                    speaker_b=cs.conversation.speaker_b,
                    old_memory=cs.memory,
                    history_sessions=history_sessions,
                    current_session=session,
                    memory_template=memory_template,
                    update_judge_template=update_judge_template,
                    memory_prompt_max_items=memory_prompt_max_items,
                    none_similarity_threshold=none_similarity_threshold,
                    add_similarity_threshold=add_similarity_threshold,
                    update_candidate_top_k=update_candidate_top_k,
                )
                return cs, session, updated, db_writes, meta, trace

            memory_rows = await _run_batched(session_work, batch_size=llm_batch, worker=_memory_job)

            for cs, session, updated, db_writes, meta, trace in memory_rows:
                cs.memory = updated
                user_id = str(build_conversation_user_id(cs.conversation.idx))
                written = 0
                write_stats = {"added": 0, "updated": 0}
                if add_flush_per_session and db_writes:
                    written, write_stats = await apply_memory_incremental_writes(
                        conn,
                        user_id=user_id,
                        items=db_writes,
                        session=session,
                    )
                session_log = {
                    "session_index": int(session.index),
                    "session_time": session.date_time,
                    "memory_count": len(updated),
                    "written": written,
                    "delta_writes": len(db_writes),
                    "db_added": write_stats["added"],
                    "db_updated": write_stats["updated"],
                    "operations": db_writes,
                    "model_operations": trace.get("operations") or [],
                    "memory": _serialize_memory_snapshot(updated),
                    **meta,
                }
                trace_record = {
                    "conversation_idx": int(cs.conversation.idx),
                    "session_index": int(session.index),
                    "session_time": session.date_time,
                    "speaker_a": cs.conversation.speaker_a,
                    "speaker_b": cs.conversation.speaker_b,
                    "user_id": user_id,
                    "written": written,
                    "delta_writes": len(db_writes),
                    "db_added": write_stats["added"],
                    "db_updated": write_stats["updated"],
                    "meta": meta,
                    **trace,
                }
                cs.conv_entry.setdefault("conversation_idx", cs.conversation.idx)
                cs.conv_entry.setdefault("speaker_a", cs.conversation.speaker_a)
                cs.conv_entry.setdefault("speaker_b", cs.conversation.speaker_b)
                cs.conv_entry.setdefault("add_backend", "global_v4_plus")
                cs.conv_entry.setdefault("add_write_mode", "incremental")
                cs.conv_entry["user_id"] = user_id
                cs.conv_entry.setdefault("sessions", [])
                cs.conv_entry["sessions"].append(session_log)
                cs.conv_entry["memory_count"] = len(updated)
                _upsert_snapshot(snapshot, cs.conv_entry)
                write_json_list(add_snapshot_path, snapshot)
                _append_jsonl(add_trace_path, trace_record)
                progress.update(1)
                print(
                    f"[add-global-v4-plus] conv{cs.conversation.idx} session{session.index} "
                    f"memory={len(updated)} delta_writes={len(db_writes)} "
                    f"upserted={written} (add={write_stats['added']} upd={write_stats['updated']}) "
                    f"midrange_calls={meta.get('midrange_llm_calls', 0)}",
                    flush=True,
                )

        if backfill_embeddings:
            embedder = EmbeddingClient()
            print(
                f"[add-global-v4-plus] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
                flush=True,
            )
            embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
            print(f"[add-global-v4-plus] embeddings written: {embedded_count}", flush=True)
        else:
            embedded_count = 0
            print("[add-global-v4-plus] embedding backfill skipped (backfill_embeddings=false)", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "global_v4_plus",
        "add_write_mode": "incremental",
        "embeddings_written": embedded_count,
    }
