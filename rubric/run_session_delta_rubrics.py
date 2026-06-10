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
RUBRIC_PATH = RUBRIC_DIR / "session_delta_rubrics.json"
PROMPT_PATH = RUBRIC_DIR / "prompt_session_delta.txt"
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


def load_rubrics() -> list[dict[str, str]]:
    data = load_json(RUBRIC_PATH)
    if not isinstance(data, list):
        raise ValueError("rubrics must be list")
    return data


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
        prev_text = str(prev.get("text") or "")
        curr_text = str(item.get("text") or "")
        prev_event = str(prev.get("event") or "")
        curr_event = str(item.get("event") or "")
        prev_time = str(prev.get("anchor_time") or "")
        curr_time = str(item.get("anchor_time") or "")
        if (prev_text, prev_event, prev_time) != (curr_text, curr_event, curr_time):
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


def build_prompt(
    *,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    history_sessions_text: str,
    current_session_text: str,
    delta_memory_text: str,
) -> str:
    return prompt_template.format(
        rubrics_json=json.dumps(rubrics, ensure_ascii=False, indent=2),
        history_sessions_text=history_sessions_text,
        current_session_text=current_session_text,
        delta_memory_text=delta_memory_text,
    )


def normalize_result(
    payload: dict[str, Any],
    rubrics: list[dict[str, str]],
    *,
    conversation_idx: int,
    session_index: int,
    add_model: str,
) -> dict[str, Any]:
    raw_items = payload.get("rubrics") or []
    by_id = {str(item.get("id") or "").strip(): item for item in raw_items if str(item.get("id") or "").strip()}
    normalized: list[dict[str, Any]] = []
    scores: list[int] = []
    for rubric in rubrics:
        rubric_id = rubric["id"]
        row = by_id.get(rubric_id, {})
        score_raw = row.get("score", 0)
        score = 1 if str(score_raw).strip() in {"1", "1.0", "true", "True"} or score_raw == 1 else 0
        scores.append(score)
        normalized.append(
            {
                "id": rubric_id,
                "name": rubric["name"],
                "question": rubric["question"],
                "score": score,
                "reason": str(row.get("reason") or "").strip(),
            }
        )
    session_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "conversation_idx": conversation_idx,
        "session_index": session_index,
        "add_model": add_model,
        "rubrics": normalized,
        "session_score": session_score,
        "summary": str(payload.get("summary") or "").strip(),
    }


async def score_one_session(
    *,
    llm: PipelineLLM,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    session_index: int,
    add_model: str,
    history_sessions_text: str,
    current_session_text: str,
    delta_memory_text: str,
) -> dict[str, Any]:
    prompt = build_prompt(
        prompt_template=prompt_template,
        rubrics=rubrics,
        history_sessions_text=history_sessions_text,
        current_session_text=current_session_text,
        delta_memory_text=delta_memory_text,
    )
    payload, meta = await asyncio.to_thread(
        llm.chat_json_object_with_meta,
        prompt,
        required_key="rubrics",
        temperature=0.0,
        max_attempts=4,
    )
    result = normalize_result(
        payload,
        rubrics,
        conversation_idx=conversation_idx,
        session_index=session_index,
        add_model=add_model,
    )
    result["evaluator_meta"] = meta
    return result


async def score_workspace(
    *,
    target: WorkspaceTarget,
    dataset: list[dict[str, Any]],
    rubrics: list[dict[str, str]],
    prompt_template: str,
    llm: PipelineLLM,
    logger: RunLogger,
    semaphore: asyncio.Semaphore,
    include_indices: set[int] | None,
) -> dict[str, Any]:
    snapshot = load_json(target.workspace_dir / "add_snapshot.json")
    selected = [
        entry for entry in snapshot
        if include_indices is None or int(entry.get("conversation_idx")) in include_indices
    ]
    session_results: list[dict[str, Any]] = []

    async def _run(conv_entry: dict[str, Any], session_pos: int) -> dict[str, Any]:
        conversation_idx = int(conv_entry.get("conversation_idx"))
        session_index = int((conv_entry.get("sessions") or [])[session_pos].get("session_index"))
        conversation_obj = dataset[conversation_idx]["conversation"]
        history_text = format_history_window(conversation_obj, session_index, HISTORY_WINDOW)
        current_text = format_session_text(conversation_obj, session_index)
        delta_items = derive_delta_items(conv_entry, session_pos)
        delta_text = format_delta_text(delta_items)
        async with semaphore:
            logger.log(
                f"score model={target.add_model} conv={conversation_idx} "
                f"session={session_index} delta_items={len(delta_items)}"
            )
            result = await score_one_session(
                llm=llm,
                prompt_template=prompt_template,
                rubrics=rubrics,
                conversation_idx=conversation_idx,
                session_index=session_index,
                add_model=target.add_model,
                history_sessions_text=history_text,
                current_session_text=current_text,
                delta_memory_text=delta_text,
            )
            result["workspace_dir"] = str(target.workspace_dir)
            result["delta_count"] = len(delta_items)
            logger.log(
                f"done model={target.add_model} conv={conversation_idx} "
                f"session={session_index} score={result['session_score']:.4f}"
            )
            return result

    tasks = []
    for conv_entry in selected:
        sessions = conv_entry.get("sessions") or []
        for session_pos in range(len(sessions)):
            tasks.append(_run(conv_entry, session_pos))
    for result in await asyncio.gather(*tasks):
        session_results.append(result)
    session_results.sort(key=lambda item: (int(item["conversation_idx"]), int(item["session_index"])))
    model_score = round(
        sum(float(item["session_score"]) for item in session_results) / len(session_results),
        4,
    ) if session_results else 0.0
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
                "session_score": session["session_score"],
            }
            for rubric in session.get("rubrics", []):
                row[f"rubric_{rubric['id']}"] = rubric["score"]
            rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Session delta rubric scoring")
    parser.add_argument("--models", type=str, default="", help="Comma-separated add models")
    parser.add_argument("--conversation-indices", type=str, default="", help="Comma-separated conversation indices")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator config")
    llm = PipelineLLM(api_key=evaluator_key, api_base=evaluator_base, model=evaluator_model, max_tokens=4096)

    run_name = f"session_delta_rubrics_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")

    dataset = load_dataset()
    rubrics = load_rubrics()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()] or None
    include_indices = {int(item.strip()) for item in args.conversation_indices.split(",") if item.strip()} if args.conversation_indices.strip() else None
    targets, missing = build_targets(selected_models)
    logger.log(f"targets={[item.add_model for item in targets]}")
    if missing:
        logger.log(f"missing={json.dumps(missing, ensure_ascii=False)}")
    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    results: list[dict[str, Any]] = []
    for target in targets:
        results.append(
            await score_workspace(
                target=target,
                dataset=dataset,
                rubrics=rubrics,
                prompt_template=prompt_template,
                llm=llm,
                logger=logger,
                semaphore=semaphore,
                include_indices=include_indices,
            )
        )

    final_scores = [float(item["session_score"]) for result in results for item in result.get("sessions", [])]
    summary = {
        "evaluator_model": evaluator_model,
        "final_score": round(sum(final_scores) / len(final_scores), 4) if final_scores else 0.0,
        "models": [
            {
                "add_model": result["add_model"],
                "workspace_dir": result["workspace_dir"],
                "session_count": result["session_count"],
                "model_score": result["model_score"],
            }
            for result in results
        ],
        "missing": missing,
    }
    (output_dir / "rubric_scores.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "evaluator_model": evaluator_model,
                "results": results,
                "missing": missing,
                "final_score": summary["final_score"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "rubric_scores.csv", results)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"final_score={summary['final_score']:.4f}")
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
