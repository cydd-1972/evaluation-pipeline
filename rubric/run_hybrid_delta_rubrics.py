from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.infra.env import evaluator_settings, load_runtime_env
from core.infra.llm_client import PipelineLLM


RUBRIC_DIR = ROOT / "rubric"
DATASET_PATH = ROOT / "datasets" / "locomo_refined.json"
RUBRICS_DIR = RUBRIC_DIR / "rubrics"
PROMPTS_DIR = RUBRIC_DIR / "prompts"
ITEM_RUBRIC_PATH = RUBRICS_DIR / "item_precision_rubrics.json"
RECALL_RUBRIC_PATH = RUBRICS_DIR / "session_recall_rubrics.json"
AUX_RUBRIC_PATH = RUBRICS_DIR / "session_auxiliary_rubrics.json"
SALIENT_PROMPT_PATH = PROMPTS_DIR / "prompt_extract_salient_facts.txt"
EVAL_PROMPT_PATH = PROMPTS_DIR / "prompt_hybrid_delta_eval.txt"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"
HISTORY_WINDOW = 2

DEFAULT_WORKSPACES = {
    "minimax": ROOT / "v5" / "workspaces" / "allconv_v5_minimax",
    "deepseek": ROOT / "v5" / "workspaces" / "allconv_v5_deepseek",
    "gemini": ROOT / "v5" / "workspaces" / "allconv_v5_gemini",
}


@dataclass
class WorkspaceTarget:
    add_model: str
    workspace_dir: Path


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset() -> list[dict[str, Any]]:
    data = load_json(DATASET_PATH)
    if not isinstance(data, list):
        raise ValueError("dataset must be list")
    return data


def load_rubrics(path: Path) -> list[dict[str, str]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"rubrics must be list: {path}")
    return data


def load_existing_results(output_dir: Path) -> tuple[list[dict[str, Any]], set[tuple[int, int, str]]]:
    results_path = output_dir / "rubric_scores.json"
    if not results_path.exists():
        return [], set()
    payload = load_json(results_path)
    existing_results = payload.get("results") or []
    completed: set[tuple[int, int, str]] = set()
    for model_result in existing_results:
        model_name = str(model_result.get("add_model") or "")
        for session in model_result.get("sessions", []) or []:
            completed.add((int(session["conversation_idx"]), int(session["session_index"]), model_name))
    return existing_results, completed


def build_targets(selected_models: list[str] | None) -> tuple[list[WorkspaceTarget], list[dict[str, Any]]]:
    models = selected_models or list(DEFAULT_WORKSPACES.keys())
    targets: list[WorkspaceTarget] = []
    missing: list[dict[str, Any]] = []
    for model in models:
        workspace = DEFAULT_WORKSPACES.get(model)
        if workspace is None:
            missing.append({"add_model": model, "status": "unknown_model"})
            continue
        snapshot_path = workspace / "add_snapshot.json"
        if not snapshot_path.exists():
            missing.append({"add_model": model, "status": "missing", "snapshot_path": str(snapshot_path)})
            continue
        targets.append(WorkspaceTarget(add_model=model, workspace_dir=workspace))
    return targets, missing


def format_session_text(conversation_obj: dict[str, Any], session_index: int) -> str:
    session_time = conversation_obj.get(f"session_{session_index}_date_time", "")
    turns = conversation_obj.get(f"session_{session_index}", []) or []
    lines = [f"[Session {session_index}] {session_time}"]
    for turn in turns:
        dia_id = str(turn.get("dia_id") or "").strip()
        speaker = str(turn.get("speaker") or "").strip()
        text = " ".join(str(turn.get("text") or "").split())
        lines.append(f"{dia_id} | {speaker}: {text}")
    return "\n".join(lines).strip()


def format_history_window(conversation_obj: dict[str, Any], current_session_index: int, window: int) -> str:
    start = max(1, current_session_index - window)
    chunks = [
        format_session_text(conversation_obj, idx)
        for idx in range(start, current_session_index)
        if conversation_obj.get(f"session_{idx}") is not None
    ]
    return "\n\n".join(chunks).strip() or "(empty)"


def derive_delta_items(conv_entry: dict[str, Any], session_pos: int) -> list[dict[str, Any]]:
    sessions = conv_entry.get("sessions") or []
    session = sessions[session_pos]
    direct_ops = session.get("operations") or session.get("model_operations") or []
    if direct_ops:
        return direct_ops
    current_memory = session.get("memory") or []
    if session_pos == 0:
        return current_memory
    previous_memory = sessions[session_pos - 1].get("memory") or []
    prev_by_id = {str(item.get("id")): item for item in previous_memory}
    delta: list[dict[str, Any]] = []
    for item in current_memory:
        item_id = str(item.get("id"))
        prev = prev_by_id.get(item_id)
        if prev is None:
            delta.append(item)
            continue
        prev_key = (str(prev.get("text") or ""), str(prev.get("event") or ""), str(prev.get("anchor_time") or ""))
        curr_key = (str(item.get("text") or ""), str(item.get("event") or ""), str(item.get("anchor_time") or ""))
        if prev_key != curr_key:
            delta.append(item)
    return delta


def format_delta_text(delta_items: list[dict[str, Any]]) -> str:
    if not delta_items:
        return "(empty)"
    lines: list[str] = []
    for index, item in enumerate(delta_items, start=1):
        lines.append(
            f"{index}. id={item.get('id','')}; event={item.get('event','')}; "
            f"anchor_time={item.get('anchor_time','')}; text={' '.join(str(item.get('text') or '').split())}"
        )
    return "\n".join(lines)


def format_salient_facts_text(salient_facts: list[dict[str, Any]]) -> str:
    if not salient_facts:
        return "(empty)"
    lines = []
    for fact in salient_facts:
        lines.append(
            f"{fact.get('fact_id','')}. category={fact.get('category','')}; "
            f"importance={fact.get('importance','')}; text={' '.join(str(fact.get('text') or '').split())}"
        )
    return "\n".join(lines)


def build_salient_prompt(prompt_template: str, current_session_text: str) -> str:
    return prompt_template.format(current_session_text=current_session_text)


def build_eval_prompt(
    prompt_template: str,
    item_rubrics: list[dict[str, str]],
    recall_rubrics: list[dict[str, str]],
    aux_rubrics: list[dict[str, str]],
    history_sessions_text: str,
    current_session_text: str,
    delta_memory_text: str,
    salient_facts_text: str,
) -> str:
    return prompt_template.format(
        item_rubrics_json=json.dumps(item_rubrics, ensure_ascii=False, indent=2),
        recall_rubrics_json=json.dumps(recall_rubrics, ensure_ascii=False, indent=2),
        aux_rubrics_json=json.dumps(aux_rubrics, ensure_ascii=False, indent=2),
        history_sessions_text=history_sessions_text,
        current_session_text=current_session_text,
        delta_memory_text=delta_memory_text,
        salient_facts_text=salient_facts_text,
    )


def mean_binary_scores(values: list[int], default: float) -> float:
    return round(sum(values) / len(values), 4) if values else default


def normalize_salient_facts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    facts = payload.get("salient_facts") or []
    normalized: list[dict[str, Any]] = []
    for idx, fact in enumerate(facts, start=1):
        text = " ".join(str(fact.get("text") or "").split()).strip()
        if not text:
            continue
        normalized.append(
            {
                "fact_id": str(fact.get("fact_id") or f"F{idx}").strip() or f"F{idx}",
                "text": text,
                "category": str(fact.get("category") or "other").strip() or "other",
                "importance": str(fact.get("importance") or "medium").strip() or "medium",
            }
        )
    return normalized


def normalize_eval_result(
    payload: dict[str, Any],
    item_rubrics: list[dict[str, str]],
    aux_rubrics: list[dict[str, str]],
    delta_items: list[dict[str, Any]],
    salient_facts: list[dict[str, Any]],
    *,
    conversation_idx: int,
    session_index: int,
    add_model: str,
) -> dict[str, Any]:
    raw_item_scores = payload.get("item_scores") or []
    raw_fact_scores = payload.get("fact_scores") or []
    raw_aux_scores = payload.get("auxiliary_scores") or []

    item_by_index = {
        int(row.get("item_index")): row
        for row in raw_item_scores
        if str(row.get("item_index") or "").strip().isdigit()
    }
    fact_by_id = {
        str(row.get("fact_id") or "").strip(): row
        for row in raw_fact_scores
        if str(row.get("fact_id") or "").strip()
    }
    aux_by_id = {
        str(row.get("id") or "").strip(): row
        for row in raw_aux_scores
        if str(row.get("id") or "").strip()
    }

    normalized_items: list[dict[str, Any]] = []
    item_binary_scores: list[int] = []
    for item_index, delta_item in enumerate(delta_items, start=1):
        row = item_by_index.get(item_index, {})
        row_rubrics = {
            str(r.get("id") or "").strip(): r
            for r in (row.get("rubrics") or [])
            if str(r.get("id") or "").strip()
        }
        rubrics_out: list[dict[str, Any]] = []
        per_item_scores: list[int] = []
        for rubric in item_rubrics:
            rubric_row = row_rubrics.get(rubric["id"], {})
            score_raw = rubric_row.get("score", 0)
            score = 1 if str(score_raw).strip() in {"1", "1.0", "true", "True"} or score_raw == 1 else 0
            per_item_scores.append(score)
            item_binary_scores.append(score)
            rubrics_out.append(
                {
                    "id": rubric["id"],
                    "name": rubric["name"],
                    "question": rubric["question"],
                    "score": score,
                    "reason": str(rubric_row.get("reason") or "").strip(),
                }
            )
        normalized_items.append(
            {
                "item_index": item_index,
                "delta_item": delta_item,
                "rubrics": rubrics_out,
                "item_score": mean_binary_scores(per_item_scores, 0.0),
            }
        )

    normalized_facts: list[dict[str, Any]] = []
    recall_binary_scores: list[int] = []
    for fact in salient_facts:
        row = fact_by_id.get(fact["fact_id"], {})
        score_raw = row.get("score", 0)
        score = 1 if str(score_raw).strip() in {"1", "1.0", "true", "True"} or score_raw == 1 else 0
        recall_binary_scores.append(score)
        matched = []
        for item in row.get("matched_item_indices") or []:
            if str(item).strip().isdigit():
                matched.append(int(item))
        normalized_facts.append(
            {
                **fact,
                "score": score,
                "reason": str(row.get("reason") or "").strip(),
                "matched_item_indices": matched,
            }
        )

    normalized_aux: list[dict[str, Any]] = []
    aux_binary_scores: list[int] = []
    for rubric in aux_rubrics:
        row = aux_by_id.get(rubric["id"], {})
        score_raw = row.get("score", 0)
        score = 1 if str(score_raw).strip() in {"1", "1.0", "true", "True"} or score_raw == 1 else 0
        aux_binary_scores.append(score)
        normalized_aux.append(
            {
                "id": rubric["id"],
                "name": rubric["name"],
                "question": rubric["question"],
                "score": score,
                "reason": str(row.get("reason") or "").strip(),
            }
        )

    precision_score = mean_binary_scores(item_binary_scores, 0.0)
    recall_score = mean_binary_scores(recall_binary_scores, 1.0)
    auxiliary_score = mean_binary_scores(aux_binary_scores, 0.0)
    final_score = round((precision_score + recall_score + auxiliary_score) / 3, 4)

    return {
        "conversation_idx": conversation_idx,
        "session_index": session_index,
        "add_model": add_model,
        "delta_count": len(delta_items),
        "salient_fact_count": len(salient_facts),
        "item_scores": normalized_items,
        "fact_scores": normalized_facts,
        "auxiliary_scores": normalized_aux,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "auxiliary_score": auxiliary_score,
        "session_score": final_score,
        "summary": str(payload.get("summary") or "").strip(),
    }


async def extract_salient_facts(
    llm: PipelineLLM,
    prompt_template: str,
    current_session_text: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_salient_prompt(prompt_template, current_session_text)
    payload, meta = await asyncio.to_thread(
        llm.chat_json_object_with_meta,
        prompt,
        required_key="salient_facts",
        temperature=0.0,
        max_attempts=4,
    )
    return normalize_salient_facts(payload), meta


async def evaluate_one_session(
    *,
    llm: PipelineLLM,
    salient_prompt_template: str,
    eval_prompt_template: str,
    item_rubrics: list[dict[str, str]],
    recall_rubrics: list[dict[str, str]],
    aux_rubrics: list[dict[str, str]],
    conversation_idx: int,
    session_index: int,
    add_model: str,
    history_sessions_text: str,
    current_session_text: str,
    delta_items: list[dict[str, Any]],
) -> dict[str, Any]:
    salient_facts, salient_meta = await extract_salient_facts(llm, salient_prompt_template, current_session_text)
    eval_prompt = build_eval_prompt(
        eval_prompt_template,
        item_rubrics,
        recall_rubrics,
        aux_rubrics,
        history_sessions_text,
        current_session_text,
        format_delta_text(delta_items),
        format_salient_facts_text(salient_facts),
    )
    payload, eval_meta = await asyncio.to_thread(
        llm.chat_json_object_with_meta,
        eval_prompt,
        required_key="item_scores",
        temperature=0.0,
        max_attempts=4,
    )
    result = normalize_eval_result(
        payload,
        item_rubrics,
        aux_rubrics,
        delta_items,
        salient_facts,
        conversation_idx=conversation_idx,
        session_index=session_index,
        add_model=add_model,
    )
    result["evaluator_meta"] = {
        "salient_fact_extraction": salient_meta,
        "hybrid_evaluation": eval_meta,
    }
    return result


async def score_workspace(
    *,
    target: WorkspaceTarget,
    dataset: list[dict[str, Any]],
    item_rubrics: list[dict[str, str]],
    recall_rubrics: list[dict[str, str]],
    aux_rubrics: list[dict[str, str]],
    salient_prompt_template: str,
    eval_prompt_template: str,
    llm: PipelineLLM,
    logger: RunLogger,
    semaphore: asyncio.Semaphore,
    include_indices: set[int] | None,
    completed_keys: set[tuple[int, int, str]],
) -> dict[str, Any]:
    snapshot = load_json(target.workspace_dir / "add_snapshot.json")
    selected = [entry for entry in snapshot if include_indices is None or int(entry.get("conversation_idx")) in include_indices]
    session_results: list[dict[str, Any]] = []

    async def _run(conv_entry: dict[str, Any], session_pos: int) -> dict[str, Any]:
        conversation_idx = int(conv_entry.get("conversation_idx"))
        session_index = int((conv_entry.get("sessions") or [])[session_pos].get("session_index"))
        key = (conversation_idx, session_index, target.add_model)
        if key in completed_keys:
            logger.log(f"skip model={target.add_model} conv={conversation_idx} session={session_index} (already completed)")
            return {}
        conversation_obj = dataset[conversation_idx]["conversation"]
        history_text = format_history_window(conversation_obj, session_index, HISTORY_WINDOW)
        current_text = format_session_text(conversation_obj, session_index)
        delta_items = derive_delta_items(conv_entry, session_pos)
        async with semaphore:
            logger.log(f"score model={target.add_model} conv={conversation_idx} session={session_index} delta_items={len(delta_items)}")
            result = await evaluate_one_session(
                llm=llm,
                salient_prompt_template=salient_prompt_template,
                eval_prompt_template=eval_prompt_template,
                item_rubrics=item_rubrics,
                recall_rubrics=recall_rubrics,
                aux_rubrics=aux_rubrics,
                conversation_idx=conversation_idx,
                session_index=session_index,
                add_model=target.add_model,
                history_sessions_text=history_text,
                current_session_text=current_text,
                delta_items=delta_items,
            )
            result["workspace_dir"] = str(target.workspace_dir)
            logger.log(
                f"done model={target.add_model} conv={conversation_idx} session={session_index} "
                f"score={result['session_score']:.4f} precision={result['precision_score']:.4f} "
                f"recall={result['recall_score']:.4f} aux={result['auxiliary_score']:.4f}"
            )
            return result

    tasks = []
    for conv_entry in selected:
        for session_pos in range(len(conv_entry.get("sessions") or [])):
            tasks.append(_run(conv_entry, session_pos))
    for result in await asyncio.gather(*tasks):
        if result:
            session_results.append(result)
    session_results.sort(key=lambda item: (int(item["conversation_idx"]), int(item["session_index"])))
    model_score = round(sum(float(item["session_score"]) for item in session_results) / len(session_results), 4) if session_results else 0.0
    return {
        "add_model": target.add_model,
        "workspace_dir": str(target.workspace_dir),
        "session_count": len(session_results),
        "model_score": model_score,
        "sessions": session_results,
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_result in results:
        for session in model_result.get("sessions", []):
            row = {
                "add_model": model_result["add_model"],
                "workspace_dir": model_result["workspace_dir"],
                "conversation_idx": session["conversation_idx"],
                "session_index": session["session_index"],
                "delta_count": session["delta_count"],
                "salient_fact_count": session["salient_fact_count"],
                "precision_score": session["precision_score"],
                "recall_score": session["recall_score"],
                "auxiliary_score": session["auxiliary_score"],
                "session_score": session["session_score"],
            }
            rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def merge_model_results(existing_results: list[dict[str, Any]], new_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, Any]] = {}
    for model_result in existing_results + new_results:
        model_name = str(model_result.get("add_model") or "")
        if not model_name:
            continue
        current = by_model.get(model_name)
        if current is None:
            current = {
                "add_model": model_result["add_model"],
                "workspace_dir": model_result["workspace_dir"],
                "session_count": 0,
                "model_score": 0.0,
                "sessions": [],
            }
            by_model[model_name] = current
        session_map = {
            (int(item["conversation_idx"]), int(item["session_index"])): item
            for item in current.get("sessions", [])
        }
        for item in model_result.get("sessions", []) or []:
            session_map[(int(item["conversation_idx"]), int(item["session_index"]))] = item
        merged_sessions = sorted(session_map.values(), key=lambda item: (int(item["conversation_idx"]), int(item["session_index"])))
        current["sessions"] = merged_sessions
        current["session_count"] = len(merged_sessions)
        current["workspace_dir"] = model_result.get("workspace_dir") or current["workspace_dir"]
        current["model_score"] = round(sum(float(item["session_score"]) for item in merged_sessions) / len(merged_sessions), 4) if merged_sessions else 0.0
    return list(by_model.values())


async def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid delta rubric scoring")
    parser.add_argument("--models", type=str, default="", help="Comma-separated add models")
    parser.add_argument("--conversation-indices", type=str, default="", help="Comma-separated conversation indices")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="", help="Resume into a specific output directory")
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator config")
    llm = PipelineLLM(api_key=evaluator_key, api_base=evaluator_base, model=evaluator_model, max_tokens=12000)

    run_name = f"hybrid_delta_rubrics_{_now_tag()}"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir.strip() else (OUTPUTS_DIR / run_name)
    if not output_dir.is_absolute():
        output_dir = output_dir.resolve()
    run_name = output_dir.name
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")

    existing_results, completed_keys = load_existing_results(output_dir)
    if completed_keys:
        logger.log(f"resume_completed_sessions={len(completed_keys)}")

    dataset = load_dataset()
    item_rubrics = load_rubrics(ITEM_RUBRIC_PATH)
    recall_rubrics = load_rubrics(RECALL_RUBRIC_PATH)
    aux_rubrics = load_rubrics(AUX_RUBRIC_PATH)
    salient_prompt_template = SALIENT_PROMPT_PATH.read_text(encoding="utf-8")
    eval_prompt_template = EVAL_PROMPT_PATH.read_text(encoding="utf-8")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()] or None
    include_indices = {int(item.strip()) for item in args.conversation_indices.split(",") if item.strip()} if args.conversation_indices.strip() else None
    targets, missing = build_targets(selected_models)
    logger.log(f"targets={[item.add_model for item in targets]}")
    if missing:
        logger.log(f"missing={json.dumps(missing, ensure_ascii=False)}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: list[dict[str, Any]] = []
    if targets:
        results = list(
            await asyncio.gather(
                *[
                    score_workspace(
                        target=target,
                        dataset=dataset,
                        item_rubrics=item_rubrics,
                        recall_rubrics=recall_rubrics,
                        aux_rubrics=aux_rubrics,
                        salient_prompt_template=salient_prompt_template,
                        eval_prompt_template=eval_prompt_template,
                        llm=llm,
                        logger=logger,
                        semaphore=semaphore,
                        include_indices=include_indices,
                        completed_keys=completed_keys,
                    )
                    for target in targets
                ]
            )
        )

    merged_results = merge_model_results(existing_results, results)
    all_scores = [float(item["session_score"]) for result in merged_results for item in result.get("sessions", [])]
    summary = {
        "evaluator_model": evaluator_model,
        "final_score": round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0,
        "models": [
            {
                "add_model": result["add_model"],
                "workspace_dir": result["workspace_dir"],
                "session_count": result["session_count"],
                "model_score": result["model_score"],
            }
            for result in merged_results
        ],
        "missing": missing,
    }
    (output_dir / "rubric_scores.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "evaluator_model": evaluator_model,
                "results": merged_results,
                "missing": missing,
                "final_score": summary["final_score"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "rubric_scores.csv", merged_results)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"final_score={summary['final_score']:.4f}")
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
