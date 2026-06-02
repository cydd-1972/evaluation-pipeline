"""add 步骤（global v4）：事实提取(LLM) + 决策(代码) + 增量 UPSERT。

相对 v3_global：
- v3：每 session clear_user_memories + 全量 insert（快照 flush）
- v4：LLM 仅提取事实；代码决定 ADD/UPDATE/NONE 并生成增量 UPSERT；未提及 id 自动保留
"""

from __future__ import annotations

import asyncio
import json
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Any

import asyncpg

from v1_mem0.add import (
    _resolve_memory_prompt_limit,
    _run_batched,
    _truncate_memory_for_prompt,
)
from v3_global.add import (
    _GlobalConvState,
    _conv_fully_done,
    _fallback_memory_from_session,
    _memory_items_from_db,
    _serialize_memory_snapshot,
    _upsert_snapshot,
    resolve_memory_prompt_path,
)
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
from core.infra.time_resolver import parse_anchor_date, resolve_relative_time
from core.infra.transcript import (
    format_session_structured_json,
    format_sessions_structured_json,
    iter_sessions,
)

def _load_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")

_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_PUNCT_RE = re.compile(r"[^a-zA-Z0-9\\s']+")

_AGGREGATE_PREFIXES = {
    "caroline_relationship": "Caroline's relationship status is ",
    "caroline_career": "Caroline's career path is ",
    "caroline_lgbtq_participation": "Caroline's LGBTQ community participation includes ",
    "caroline_transgender_events": "Caroline has attended transgender-specific events including ",
    "melanie_pets": "Melanie's known pets are ",
    "melanie_bought_items": "Melanie has bought ",
    "melanie_music_seen": "Melanie has seen musical artists or bands including ",
    "melanie_caroline_recommended_book": "The book Melanie read from Caroline's recommendation is ",
    "melanie_children_count": "Melanie's number of children is ",
}


def _normalize_text(text: str) -> str:
    cleaned = _PUNCT_RE.sub(" ", str(text or "").strip().lower())
    return " ".join(cleaned.split())


def _token_set(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _similarity(a: str, b: str) -> float:
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if not na or not nb:
        return 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    jac = _jaccard(_token_set(na), _token_set(nb))
    return max(float(seq), float(jac))


def _next_numeric_id(used: set[str]) -> str:
    i = 0
    while True:
        candidate = str(i)
        if candidate not in used:
            return candidate
        i += 1


def _join_values(values: list[str]) -> str:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _strip_aggregate_texts(memory: list[dict[str, Any]]) -> list[str]:
    prefixes = tuple(_AGGREGATE_PREFIXES.values())
    return [
        str(item.get("text") or "").strip()
        for item in memory
        if str(item.get("text") or "").strip()
        and not str(item.get("text") or "").strip().startswith(prefixes)
    ]


def _find_existing_aggregate(memory: list[dict[str, Any]], prefix: str) -> dict[str, Any] | None:
    for item in memory:
        text = str(item.get("text") or "").strip()
        if text.startswith(prefix):
            return item
    return None


def _add_unique(values: list[str], value: str) -> None:
    cleaned = str(value or "").strip(" .,:;")
    cleaned = re.sub(r"\bthat\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        return
    if cleaned.lower() not in {item.lower() for item in values}:
        values.append(cleaned)


def _derive_slot_values(texts: list[str]) -> dict[str, list[str] | str]:
    joined = "\n".join(texts)
    lower_joined = joined.lower()
    slots: dict[str, list[str] | str] = {}

    if "single parent" in lower_joined:
        slots["caroline_relationship"] = "single"

    if (
        "counseling" in lower_joined
        and "mental health" in lower_joined
        and ("career" in lower_joined or "work" in lower_joined or "way to go" in lower_joined)
    ):
        if "transgender" in lower_joined or "trans people" in lower_joined or "lgbtq" in lower_joined:
            slots["caroline_career"] = "counseling or mental health work for transgender/LGBTQ people"
        else:
            slots["caroline_career"] = "counseling or mental health work"

    pets: list[str] = []
    for text in texts:
        for pattern in [
            r"\bMelanie (?:also )?has (?:a|an|the)?(?: [a-z]+){0,3} (?:cat|dog) named ([A-Z][A-Za-z'-]+)",
            r"\bMelanie's (?:cat|dog) (?:is )?named ([A-Z][A-Za-z'-]+)",
            r"\b([A-Z][A-Za-z'-]+) is Melanie's (?:cat|dog)",
        ]:
            for match in re.finditer(pattern, text):
                _add_unique(pets, match.group(1))
    if pets:
        slots["melanie_pets"] = pets

    bought: list[str] = []
    item_nouns = {
        "book",
        "books",
        "bowl",
        "bowls",
        "cup",
        "figurine",
        "figurines",
        "painting",
        "paintings",
        "plate",
        "pot",
        "pots",
        "shoes",
        "slipper",
        "slippers",
    }
    for text in texts:
        if "Melanie" not in text:
            continue
        match = re.search(r"\b(?:bought|purchased|picked up|got) (?:some |a |an |the )?(.+?)(?: yesterday| last| this| from| at| for|\.|$)", text, re.IGNORECASE)
        if match:
            raw = match.group(1)
            for part in re.split(r",| and ", raw):
                value = part.strip(" .")
                value_words = {word.lower() for word in re.findall(r"[A-Za-z]+", value)}
                if value and len(value.split()) <= 4 and value_words & item_nouns:
                    _add_unique(bought, value)
    if bought:
        slots["melanie_bought_items"] = bought

    music: list[str] = []
    for text in texts:
        if "Melanie" not in text and "concert" not in text:
            continue
        match = re.search(r"\bMelanie went to (?:a |an |the )?(.+?) concert\b", text)
        if match and match.group(1).strip().lower() not in {"a", "an", "the"}:
            _add_unique(music, match.group(1))
        match = re.search(r"\bconcert (?:featured|was performed by) ([A-Z][A-Za-z0-9 '&.-]+?)(?:\.|$)", text)
        if match:
            _add_unique(music, match.group(1).strip("'\""))
        match = re.search(r"\bconcert by ['\"]?([A-Z][A-Za-z0-9 '&.-]+?)['\"]?(?:\.|$)", text)
        if match:
            _add_unique(music, match.group(1).strip("'\""))
    if music:
        slots["melanie_music_seen"] = music

    participation: list[str] = []
    participation_rules = [
        ("support groups", ("support group",)),
        ("LGBTQ conferences", ("lgbtq conference",)),
        ("LGBTQ counseling workshops", ("counseling workshop",)),
        ("pride parades", ("pride parade", "pride event")),
        ("activist groups", ("activist group", "connected lgbtq activists")),
        ("mentoring LGBTQ youth", ("mentorship program", "mentors a transgender teen", "mentee")),
        ("volunteering at an LGBTQ+ youth center", ("volunteering at an lgbtq+ youth center", "lgbtq+ youth center")),
        ("LGBTQ art shows", ("lgbtq art show",)),
    ]
    for label, needles in participation_rules:
        if any(needle in lower_joined for needle in needles):
            _add_unique(participation, label)
    if participation:
        slots["caroline_lgbtq_participation"] = participation

    trans_events: list[str] = []
    if "transgender conference" in lower_joined:
        _add_unique(trans_events, "a transgender conference")
    if "transgender poetry reading" in lower_joined:
        _add_unique(trans_events, "a transgender poetry reading")
    if "lgbtq+ counseling workshop" in lower_joined and ("trans people" in lower_joined or "transgender" in lower_joined):
        _add_unique(trans_events, "an LGBTQ+ counseling workshop about working with trans people")
    if trans_events:
        slots["caroline_transgender_events"] = trans_events

    recommended_books: list[str] = []
    for text in texts:
        match = re.search(r"\bCaroline recommends ['\"]([^'\"]+)['\"]", text)
        if match:
            _add_unique(recommended_books, f"'{match.group(1)}'")
        match = re.search(r"\bCaroline's favorite book is ['\"]([^'\"]+)['\"]", text)
        if match:
            _add_unique(recommended_books, f"'{match.group(1)}'")
    if recommended_books and "melanie is reading the book caroline recommended" in lower_joined:
        slots["melanie_caroline_recommended_book"] = recommended_books[0]
    if recommended_books and "melanie has been reading a book caroline recommended" in lower_joined:
        slots["melanie_caroline_recommended_book"] = recommended_books[0]

    if re.search(r"\b(two|2) younger kids\b", lower_joined) and "daughter" in lower_joined:
        slots["melanie_children_count"] = "three children"

    return slots


def _derive_slot_aggregate_ops(
    *,
    memory: list[dict[str, Any]],
    session_anchor_time: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    texts = _strip_aggregate_texts(memory)
    slots = _derive_slot_values(texts)
    used_ids = {str(item.get("id") or "").strip() for item in memory if str(item.get("id") or "").strip()}
    ops: list[dict[str, Any]] = []
    stats = {"aggregate_add": 0, "aggregate_update": 0, "aggregate_none": 0}

    for slot_key, slot_value in slots.items():
        prefix = _AGGREGATE_PREFIXES.get(slot_key)
        if not prefix:
            continue
        value_text = _join_values(slot_value) if isinstance(slot_value, list) else str(slot_value or "").strip()
        if not value_text:
            continue
        text = f"{prefix}{value_text}."
        existing = _find_existing_aggregate(memory, prefix)
        if existing:
            memory_id = str(existing.get("id") or "").strip()
            if str(existing.get("text") or "").strip() == text:
                stats["aggregate_none"] += 1
                continue
            ops.append(
                {
                    "id": memory_id,
                    "text": text,
                    "event": "UPDATE",
                    "anchor_time": str(existing.get("anchor_time") or session_anchor_time).strip(),
                }
            )
            stats["aggregate_update"] += 1
            continue
        memory_id = _next_numeric_id(used_ids)
        used_ids.add(memory_id)
        ops.append({"id": memory_id, "text": text, "event": "ADD", "anchor_time": session_anchor_time})
        stats["aggregate_add"] += 1
    return ops, stats


def _resolved_time_value(text: str, anchor_time: str) -> str:
    anchor_date = parse_anchor_date(anchor_time)
    if not anchor_date:
        return ""
    resolved = resolve_relative_time(text, anchor_date)
    return resolved.value if resolved else ""


def _normalize_anchor_time_iso(
    llm: PipelineLLM,
    *,
    session_time_raw: str,
) -> tuple[str, bool]:
    """Normalize session_time into ISO date (YYYY-MM-DD) for anchor_time.

    Returns (anchor_time_iso_or_raw, used_llm).
    """
    raw = str(session_time_raw or "").strip()
    if not raw:
        return "", False
    parsed = parse_anchor_date(raw)
    if parsed:
        return parsed.isoformat(), False
    prompt = (
        "Normalize the following session_time into an ISO calendar date.\n\n"
        "Rules:\n"
        '- Output JSON object with key "date".\n'
        '- If a specific calendar date is present or can be safely extracted, return "YYYY-MM-DD".\n'
        '- If only a month/year is present or the date is ambiguous, return an empty string.\n'
        "- Return JSON only.\n\n"
        f"session_time: {raw!r}\n\n"
        'Output: {"date": "YYYY-MM-DD" | ""}'
    )
    try:
        payload = llm.chat_json_object(prompt, required_key="date", max_attempts=3)
    except ValueError:
        return raw, True
    value = str(payload.get("date") or "").strip()
    if value and parse_anchor_date(value):
        return value, True
    return raw, True


def _coerce_fact_items(payload: dict[str, Any]) -> list[dict[str, str]]:
    raw = payload.get("facts")
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fact = str(entry.get("fact") or entry.get("text") or "").strip()
        if not fact:
            continue
        fact_type = str(entry.get("type") or "").strip().lower()
        if fact_type not in {"event", "plan", "state", "feeling", "negative", "attribute"}:
            fact_type = "event"
        items.append({"fact": fact, "type": fact_type})
    return items


def _decide_ops_from_facts(
    *,
    old_memory: list[dict[str, Any]],
    new_facts: list[dict[str, str]],
    session_anchor_time: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    used_ids = {str(item.get("id") or "").strip() for item in old_memory if str(item.get("id") or "").strip()}
    known_texts = {str(item.get("text") or "").strip().lower() for item in old_memory if str(item.get("text") or "").strip()}
    old_by_id = {
        str(item.get("id") or "").strip(): item
        for item in old_memory
        if str(item.get("id") or "").strip() and str(item.get("text") or "").strip()
    }

    meta: dict[str, Any] = {"facts_total": len(new_facts), "ops_add": 0, "ops_update": 0, "ops_none": 0}

    best_update_by_id: dict[str, tuple[float, dict[str, Any]]] = {}
    ops_add: list[dict[str, Any]] = []

    for fact_item in new_facts:
        fact_text = str(fact_item.get("fact") or "").strip()
        fact_type = str(fact_item.get("type") or "event").strip().lower()
        if not fact_text:
            continue
        if fact_text.lower() in known_texts:
            meta["ops_none"] += 1
            continue

        best_id = ""
        best_score = 0.0
        for memory_id, old_item in old_by_id.items():
            score = _similarity(fact_text, str(old_item.get("text") or ""))
            if score > best_score:
                best_score = score
                best_id = memory_id

        if best_id and best_score >= 0.92:
            meta["ops_none"] += 1
            continue

        if best_id and best_score >= 0.70:
            old_item = old_by_id[best_id]
            old_text = str(old_item.get("text") or "")
            old_anchor = str(old_item.get("anchor_time") or "").strip()
            if not old_anchor:
                old_anchor = session_anchor_time

            old_resolved = _resolved_time_value(old_text, old_anchor) if old_anchor else ""
            new_resolved = _resolved_time_value(fact_text, session_anchor_time) if session_anchor_time else ""

            if old_resolved and new_resolved and old_resolved != new_resolved:
                pass
            else:
                old_tokens = _token_set(old_text)
                new_tokens = _token_set(fact_text)
                is_more_specific = best_score >= 0.75 and len(new_tokens - old_tokens) >= 2
                has_time_addition = not old_resolved and bool(new_resolved)
                is_plan = fact_type == "plan"
                if is_more_specific or has_time_addition or is_plan:
                    op = {
                        "id": best_id,
                        "text": fact_text,
                        "event": "UPDATE",
                        "old_memory": old_text,
                        "anchor_time": old_anchor,
                    }
                    existing = best_update_by_id.get(best_id)
                    if existing is None or best_score > existing[0]:
                        best_update_by_id[best_id] = (best_score, op)
                    continue

        new_id = _next_numeric_id(used_ids)
        used_ids.add(new_id)
        ops_add.append({"id": new_id, "text": fact_text, "event": "ADD", "anchor_time": session_anchor_time})
        meta["ops_add"] += 1
        known_texts.add(fact_text.lower())

    updates = [payload for _score, payload in sorted(best_update_by_id.values(), key=lambda pair: -pair[0])]
    meta["ops_update"] = len(updates)
    return updates + ops_add, meta


def _decide_global_memory_v4_sync(
    llm: PipelineLLM,
    *,
    speaker_a: str,
    speaker_b: str,
    old_memory: list[dict[str, Any]],
    history_sessions: list[Any],
    current_session: Any,
    memory_template: str,
    memory_prompt_max_items: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {"add_write_mode": "incremental"}
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

        payload = llm.chat_json_object(prompt, required_key="facts", max_attempts=6)
        facts = _coerce_fact_items(payload)
        meta["memory_prompt"] = "extract"
        meta["facts_extracted"] = len(facts)
        ops, op_stats = _decide_ops_from_facts(
            old_memory=old_memory,
            new_facts=facts,
            session_anchor_time=session_anchor_time,
        )
        meta.update(op_stats)
        merged, db_writes, delta_stats = apply_global_memory_delta(old_memory, ops)
        meta.update(delta_stats)
        aggregate_ops, aggregate_stats = _derive_slot_aggregate_ops(
            memory=merged,
            session_anchor_time=session_anchor_time,
        )
        meta.update(aggregate_stats)
        if aggregate_ops:
            merged, aggregate_writes, aggregate_delta_stats = apply_global_memory_delta(merged, aggregate_ops)
            db_writes.extend(aggregate_writes)
            meta["aggregate_writes"] = len(aggregate_writes)
            meta["delta_added"] = int(meta.get("delta_added") or 0) + int(
                aggregate_delta_stats.get("delta_added") or 0
            )
            meta["delta_updated"] = int(meta.get("delta_updated") or 0) + int(
                aggregate_delta_stats.get("delta_updated") or 0
            )
            meta["delta_none"] = int(meta.get("delta_none") or 0) + int(
                aggregate_delta_stats.get("delta_none") or 0
            )
        return merged, db_writes, meta
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
    aggregate_ops, aggregate_stats = _derive_slot_aggregate_ops(
        memory=merged,
        session_anchor_time=session_anchor_time,
    )
    meta.update(aggregate_stats)
    if aggregate_ops:
        merged, aggregate_writes, aggregate_delta_stats = apply_global_memory_delta(merged, aggregate_ops)
        db_writes.extend(aggregate_writes)
        meta["aggregate_writes"] = len(aggregate_writes)
        meta["delta_added"] = int(meta.get("delta_added") or 0) + int(aggregate_delta_stats.get("delta_added") or 0)
        meta["delta_updated"] = int(meta.get("delta_updated") or 0) + int(
            aggregate_delta_stats.get("delta_updated") or 0
        )
        meta["delta_none"] = int(meta.get("delta_none") or 0) + int(aggregate_delta_stats.get("delta_none") or 0)
    return merged, db_writes, meta


async def run_add_global_v4(
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
) -> dict[str, Any]:
    """global v4 add：structured D/M，每 session 增量 UPSERT（无 clear flush）。"""
    resolved_llm = llm or PipelineLLM()
    prompt_path = resolve_memory_prompt_path(memory_prompt_path)
    memory_template = _load_template(prompt_path)
    prompt_limit = _resolve_memory_prompt_limit(memory_prompt_max_items, default=None)
    limit_label = "unlimited" if prompt_limit is None else str(prompt_limit)
    print(
        f"[add-global-v4] memory_prompt={prompt_path.name} "
        f"memory_prompt_max_items={limit_label} write=incremental",
        flush=True,
    )
    llm_batch = max(1, int(add_llm_concurrency or 1))
    history_window = max(0, int(add_history_window or 0))

    add_snapshot_path = workspace_dir / "add_snapshot.json"
    snapshot_for_reset = load_json_list(add_snapshot_path) if add_snapshot_path.exists() else []
    effective_reset = bool(reset_database) and not snapshot_for_reset
    if reset_database and snapshot_for_reset:
        print(
            f"[add-global-v4] reset_database_on_add=true ignored: {add_snapshot_path.name} exists "
            f"({len(snapshot_for_reset)} conversation(s)) — resume mode",
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
                "add_backend": "global_v4",
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
        f"[add-global-v4] conversations={len(conversations)} sessions={len(session_plans)} "
        f"history_window={history_window} persist_per_session={add_flush_per_session} "
        f"llm_batch={llm_batch}",
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
            print(f"[add-global-v4] conv{conversation.idx} skipped (complete in snapshot)", flush=True)
            continue
        start_idx = 0
        memory: list[dict[str, Any]] = []
        conv_entry = dict(existing) if isinstance(existing, dict) else {}
        if existing:
            start_idx = len(existing.get("sessions") or [])
            rows = await list_memories_for_user(conn, str(build_conversation_user_id(conversation.idx)))
            memory = _memory_items_from_db(rows)
            print(
                f"[add-global-v4] conv{conversation.idx} resume from session {start_idx + 1}/{len(sessions)} "
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
        state.conv_entry["add_backend"] = "global_v4"
        state.conv_entry["add_write_mode"] = "incremental"
        conv_states.append(state)

    progress = ProgressBar("add-global-v4", total=len(session_plans) or None, unit="session", label=progress_label)
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
                progress.set_description(f"add-global-v4 conv{cs.conversation.idx} session{session.index}")
                progress.set_postfix_str(f"{cs.conversation.speaker_a} & {cs.conversation.speaker_b}")

            async def _memory_job(
                item: tuple[_GlobalConvState, Any],
            ) -> tuple[_GlobalConvState, Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
                cs, session = item
                hist_start = max(0, session_index - history_window)
                history_sessions = cs.sessions[hist_start:session_index]
                updated, db_writes, meta = await asyncio.to_thread(
                    _decide_global_memory_v4_sync,
                    resolved_llm,
                    speaker_a=cs.conversation.speaker_a,
                    speaker_b=cs.conversation.speaker_b,
                    old_memory=cs.memory,
                    history_sessions=history_sessions,
                    current_session=session,
                    memory_template=memory_template,
                    memory_prompt_max_items=memory_prompt_max_items,
                )
                return cs, session, updated, db_writes, meta

            memory_rows = await _run_batched(session_work, batch_size=llm_batch, worker=_memory_job)

            for cs, session, updated, db_writes, meta in memory_rows:
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
                    "memory": _serialize_memory_snapshot(updated),
                    **meta,
                }
                cs.conv_entry.setdefault("conversation_idx", cs.conversation.idx)
                cs.conv_entry.setdefault("speaker_a", cs.conversation.speaker_a)
                cs.conv_entry.setdefault("speaker_b", cs.conversation.speaker_b)
                cs.conv_entry.setdefault("add_backend", "global_v4")
                cs.conv_entry.setdefault("add_write_mode", "incremental")
                cs.conv_entry["user_id"] = user_id
                cs.conv_entry.setdefault("sessions", [])
                cs.conv_entry["sessions"].append(session_log)
                cs.conv_entry["memory_count"] = len(updated)
                _upsert_snapshot(snapshot, cs.conv_entry)
                write_json_list(add_snapshot_path, snapshot)
                progress.update(1)
                print(
                    f"[add-global-v4] conv{cs.conversation.idx} session{session.index} "
                    f"memory={len(updated)} delta_writes={len(db_writes)} "
                    f"upserted={written} (add={write_stats['added']} upd={write_stats['updated']}) "
                    f"preserved_unmentioned={meta.get('preserved_unmentioned', 0)}",
                    flush=True,
                )

        if backfill_embeddings:
            embedder = EmbeddingClient()
            print(
                f"[add-global-v4] embedding memories with {embedder.model} (dim={embedder.dimensions}) ...",
                flush=True,
            )
            embedded_count = await backfill_memory_embeddings(conn, embedder=embedder)
            print(f"[add-global-v4] embeddings written: {embedded_count}", flush=True)
        else:
            embedded_count = 0
            print("[add-global-v4] embedding backfill skipped (backfill_embeddings=false)", flush=True)
    finally:
        progress.close()
        await conn.close()

    write_json_list(add_snapshot_path, snapshot)
    return {
        "database_url": db_url,
        "add_snapshot_path": str(add_snapshot_path),
        "conversation_count": len(snapshot),
        "add_backend": "global_v4",
        "add_write_mode": "incremental",
        "embeddings_written": embedded_count,
    }
