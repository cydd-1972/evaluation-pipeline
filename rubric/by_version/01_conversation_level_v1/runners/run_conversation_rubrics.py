from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
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
RUBRIC_PATH = RUBRICS_DIR / "conversation_level_rubrics.json"
PROMPT_PATH = PROMPTS_DIR / "prompt_conversation_level.txt"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"

DEFAULT_WORKSPACES = {
    "minimax": ROOT / "v5" / "workspaces" / "allconv_v5_minimax",
    "deepseek": ROOT / "v5" / "workspaces" / "allconv_v5_deepseek",
    "gemini": ROOT / "v5" / "workspaces" / "allconv_v5_gemini",
    "qwen3-14b": ROOT / "v5" / "workspaces" / "allconv_v5_qwen3_14b",
    "qwen3-4b": ROOT / "v5" / "workspaces" / "allconv_v5_qwen3_4b",
}


@dataclass
class WorkspaceTarget:
    add_model: str
    workspace_dir: Path


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset() -> list[dict[str, Any]]:
    data = load_json(DATASET_PATH)
    if not isinstance(data, list):
        raise ValueError(f"dataset must be a list: {DATASET_PATH}")
    return data


def load_rubrics() -> list[dict[str, str]]:
    rubrics = load_json(RUBRIC_PATH)
    if not isinstance(rubrics, list):
        raise ValueError(f"rubrics must be a list: {RUBRIC_PATH}")
    return rubrics


def format_conversation_text(item: dict[str, Any]) -> str:
    conversation = item.get("conversation") or {}
    lines: list[str] = []
    speaker_a = conversation.get("speaker_a", "")
    speaker_b = conversation.get("speaker_b", "")
    lines.append(f"Participants: {speaker_a} / {speaker_b}")
    session_ids = sorted(
        {
            int(key.split("_")[1])
            for key in conversation.keys()
            if key.startswith("session_") and key.endswith("_date_time")
        }
    )
    for session_idx in session_ids:
        session_time = conversation.get(f"session_{session_idx}_date_time", "")
        turns = conversation.get(f"session_{session_idx}", []) or []
        lines.append(f"\n[Session {session_idx}] {session_time}")
        for turn in turns:
            dia_id = str(turn.get("dia_id") or "").strip()
            speaker = str(turn.get("speaker") or "").strip()
            text = " ".join(str(turn.get("text") or "").split())
            lines.append(f"{dia_id} | {speaker}: {text}")
    return "\n".join(lines).strip()


def _session_final_memory(conv_entry: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = conv_entry.get("sessions") or []
    if not sessions:
        return []
    last_session = sessions[-1]
    memory = last_session.get("memory") or conv_entry.get("memory") or []
    if not isinstance(memory, list):
        return []
    return memory


def format_final_memory_text(memory: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, item in enumerate(memory, start=1):
        memory_id = str(item.get("id") or "")
        text = " ".join(str(item.get("text") or "").split())
        event = str(item.get("event") or "")
        anchor_time = str(item.get("anchor_time") or "")
        lines.append(
            f"{index}. id={memory_id}; event={event}; anchor_time={anchor_time}; text={text}"
        )
    return "\n".join(lines).strip()


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
            missing.append(
                {
                    "add_model": model,
                    "status": "missing",
                    "workspace_dir": str(workspace),
                    "snapshot_path": str(snapshot_path),
                }
            )
            continue
        targets.append(WorkspaceTarget(add_model=model, workspace_dir=workspace))
    return targets, missing


def should_keep_conversation(
    conversation_idx: int,
    *,
    include_indices: set[int] | None,
    max_conversations: int | None,
) -> bool:
    if include_indices is not None and conversation_idx not in include_indices:
        return False
    if max_conversations is not None and conversation_idx >= max_conversations:
        return False
    return True


def build_prompt(
    *,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    add_model: str,
    conversation_text: str,
    final_memory_text: str,
) -> str:
    return prompt_template.format(
        rubrics_json=json.dumps(rubrics, ensure_ascii=False, indent=2),
        conversation_idx=conversation_idx,
        add_model=add_model,
        conversation_text=conversation_text,
        final_memory_text=final_memory_text or "(empty)",
    )


def normalize_result(
    *,
    payload: dict[str, Any],
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    add_model: str,
) -> dict[str, Any]:
    rubric_defs = {item["id"]: item for item in rubrics}
    raw_items = payload.get("rubrics") or []
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        rubric_id = str(item.get("id") or "").strip()
        if rubric_id:
            by_id[rubric_id] = item
    normalized_items: list[dict[str, Any]] = []
    scores: list[int] = []
    for rubric in rubrics:
        rubric_id = rubric["id"]
        row = by_id.get(rubric_id, {})
        score_raw = row.get("score", 0)
        score = 1 if str(score_raw).strip() in {"1", "1.0", "true", "True"} or score_raw == 1 else 0
        scores.append(score)
        normalized_items.append(
            {
                "id": rubric_id,
                "name": rubric["name"],
                "question": rubric["question"],
                "score": score,
                "reason": str(row.get("reason") or "").strip(),
            }
        )
    conversation_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "conversation_idx": conversation_idx,
        "add_model": add_model,
        "rubrics": normalized_items,
        "conversation_score": conversation_score,
        "summary": str(payload.get("summary") or "").strip(),
    }


async def score_one_conversation(
    *,
    llm: PipelineLLM,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    add_model: str,
    conversation_text: str,
    final_memory_text: str,
) -> dict[str, Any]:
    prompt = build_prompt(
        prompt_template=prompt_template,
        rubrics=rubrics,
        conversation_idx=conversation_idx,
        add_model=add_model,
        conversation_text=conversation_text,
        final_memory_text=final_memory_text,
    )
    payload, meta = await asyncio.to_thread(
        llm.chat_json_object_with_meta,
        prompt,
        required_key="rubrics",
        temperature=0.0,
        max_attempts=4,
    )
    result = normalize_result(
        payload=payload,
        rubrics=rubrics,
        conversation_idx=conversation_idx,
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
    max_conversations: int | None,
) -> dict[str, Any]:
    snapshot_path = target.workspace_dir / "add_snapshot.json"
    snapshot = load_json(snapshot_path)
    if not isinstance(snapshot, list):
        raise ValueError(f"snapshot must be a list: {snapshot_path}")
    logger.log(f"start model={target.add_model} workspace={target.workspace_dir}")
    conversations: list[dict[str, Any]] = []

    async def _run(conv_entry: dict[str, Any]) -> dict[str, Any]:
        conversation_idx = int(conv_entry.get("conversation_idx"))
        dataset_item = dataset[conversation_idx]
        conversation_text = format_conversation_text(dataset_item)
        final_memory = _session_final_memory(conv_entry)
        final_memory_text = format_final_memory_text(final_memory)
        async with semaphore:
            logger.log(
                f"score model={target.add_model} conv={conversation_idx} "
                f"memory_items={len(final_memory)}"
            )
            result = await score_one_conversation(
                llm=llm,
                prompt_template=prompt_template,
                rubrics=rubrics,
                conversation_idx=conversation_idx,
                add_model=target.add_model,
                conversation_text=conversation_text,
                final_memory_text=final_memory_text,
            )
            logger.log(
                f"done model={target.add_model} conv={conversation_idx} "
                f"score={result['conversation_score']:.4f}"
            )
            result["workspace_dir"] = str(target.workspace_dir)
            result["memory_count"] = len(final_memory)
            return result

    filtered_snapshot = [
        conv_entry
        for conv_entry in snapshot
        if should_keep_conversation(
            int(conv_entry.get("conversation_idx")),
            include_indices=include_indices,
            max_conversations=max_conversations,
        )
    ]
    tasks = [_run(conv_entry) for conv_entry in filtered_snapshot]
    for result in await asyncio.gather(*tasks):
        conversations.append(result)
    conversations.sort(key=lambda item: int(item["conversation_idx"]))
    model_score = round(
        sum(float(item["conversation_score"]) for item in conversations) / len(conversations),
        4,
    ) if conversations else 0.0
    return {
        "add_model": target.add_model,
        "workspace_dir": str(target.workspace_dir),
        "snapshot_path": str(snapshot_path),
        "conversation_count": len(conversations),
        "model_score": model_score,
        "conversations": conversations,
    }


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for model_result in results:
        for conversation in model_result.get("conversations", []):
            row: dict[str, Any] = {
                "add_model": model_result["add_model"],
                "workspace_dir": model_result["workspace_dir"],
                "conversation_idx": conversation["conversation_idx"],
                "memory_count": conversation["memory_count"],
                "conversation_score": conversation["conversation_score"],
            }
            for rubric in conversation.get("rubrics", []):
                row[f"rubric_{rubric['id']}"] = rubric["score"]
            rows.append(row)
    if not rows:
        return
    fieldnames: list[str] = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    results: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    evaluator_model: str,
) -> dict[str, Any]:
    all_conv_scores = [
        float(conversation["conversation_score"])
        for model_result in results
        for conversation in model_result.get("conversations", [])
    ]
    final_score = round(sum(all_conv_scores) / len(all_conv_scores), 4) if all_conv_scores else 0.0
    models = [
        {
            "add_model": model_result["add_model"],
            "workspace_dir": model_result["workspace_dir"],
            "conversation_count": model_result["conversation_count"],
            "model_score": model_result["model_score"],
        }
        for model_result in results
    ]
    return {
        "evaluator_model": evaluator_model,
        "final_score": final_score,
        "models": models,
        "missing": missing,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Conversation-level rubric scoring for final memory.")
    parser.add_argument("--models", type=str, default="", help="Comma-separated add models to evaluate.")
    parser.add_argument("--concurrency", type=int, default=1, help="LLM scoring concurrency.")
    parser.add_argument("--max-conversations", type=int, default=None, help="Only score the first N conversation indices.")
    parser.add_argument("--conversation-indices", type=str, default="", help="Comma-separated conversation indices to score.")
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator model/base/key configuration")
    llm = PipelineLLM(
        api_key=evaluator_key,
        api_base=evaluator_base,
        model=evaluator_model,
        max_tokens=4096,
    )

    run_name = f"conversation_rubrics_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")

    dataset = load_dataset()
    rubrics = load_rubrics()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()] or None
    include_indices = (
        {int(item.strip()) for item in args.conversation_indices.split(",") if item.strip()}
        if args.conversation_indices.strip()
        else None
    )
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
                max_conversations=args.max_conversations,
            )
        )

    summary = build_summary(results=results, missing=missing, evaluator_model=evaluator_model)
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
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log(f"final_score={summary['final_score']:.4f}")
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
