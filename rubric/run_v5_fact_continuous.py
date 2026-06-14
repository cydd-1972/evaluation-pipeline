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
DEFAULT_RUBRIC_PATH = RUBRICS_DIR / "conversation_fact_rubrics_continuous.json"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "prompt_conversation_fact_continuous.txt"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"

DEFAULT_WORKSPACES = {
    "minimax": ROOT / "v5" / "workspaces" / "allconv_v5_minimax",
    "deepseek": ROOT / "v5" / "workspaces" / "allconv_v5_deepseek",
    "gemini": ROOT / "v5" / "workspaces" / "allconv_v5_gemini",
}

WEIGHTS = {
    "fact_sourceability": 3.0,
    "fact_factual_correctness": 3.0,
    "fact_temporal_correctness": 2.0,
    "fact_atomicity": 1.0,
    "fact_nonredundancy": 1.0,
    "fact_coverage": 2.0,
    "fact_conciseness": 1.0,
    "fact_structural_consistency": 1.0,
}
HARD_CAP_RUBRICS = (
    "fact_sourceability",
    "fact_factual_correctness",
)
HARD_CAP_THRESHOLD = 0.3
HARD_CAP_VALUE = 0.4


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
        raise ValueError("dataset must be a list")
    return data


def load_rubrics(path: Path) -> list[dict[str, str]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("rubrics must be a list")
    return data


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


def final_memory_from_entry(conv_entry: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = conv_entry.get("sessions") or []
    if not sessions:
        return []
    last_session = sessions[-1]
    memory = last_session.get("memory") or conv_entry.get("memory") or []
    return memory if isinstance(memory, list) else []


def format_memory_text(memory: list[dict[str, Any]]) -> str:
    if not memory:
        return "(empty)"
    lines: list[str] = []
    for index, item in enumerate(memory, start=1):
        lines.append(
            f"{index}. id={item.get('id','')}; event={item.get('event','')}; "
            f"anchor_time={item.get('anchor_time','')}; text={' '.join(str(item.get('text') or '').split())}"
        )
    return "\n".join(lines)


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


def build_prompt(
    *,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_text: str,
    fact_memory_text: str,
) -> str:
    return prompt_template.format(
        rubrics_json=json.dumps(rubrics, ensure_ascii=False, indent=2),
        conversation_text=conversation_text,
        fact_memory_text=fact_memory_text,
    )


def normalize_score(raw: Any) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return round(value, 4)


def weighted_score(rubric_rows: list[dict[str, Any]]) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    by_id: dict[str, float] = {}
    for row in rubric_rows:
        rubric_id = row["id"]
        score = float(row["score"])
        weight = float(WEIGHTS.get(rubric_id, 1.0))
        by_id[rubric_id] = score
        total_weight += weight
        weighted_sum += score * weight
    score = round(weighted_sum / total_weight, 4) if total_weight else 0.0
    if any(by_id.get(rubric_id, 1.0) <= HARD_CAP_THRESHOLD for rubric_id in HARD_CAP_RUBRICS):
        score = min(score, HARD_CAP_VALUE)
    return score


def normalize_result(
    *,
    payload: dict[str, Any],
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    add_model: str,
    workspace_dir: Path,
    memory_count: int,
) -> dict[str, Any]:
    raw_items = payload.get("rubrics") or []
    by_id = {
        str(item.get("id") or "").strip(): item
        for item in raw_items
        if str(item.get("id") or "").strip()
    }
    normalized_items: list[dict[str, Any]] = []
    for rubric in rubrics:
        rubric_id = rubric["id"]
        row = by_id.get(rubric_id, {})
        normalized_items.append(
            {
                "id": rubric_id,
                "name": rubric["name"],
                "question": rubric["question"],
                "score": normalize_score(row.get("score", 0.0)),
                "reason": str(row.get("reason") or "").strip(),
            }
        )
    arithmetic_mean = round(sum(float(row["score"]) for row in normalized_items) / len(normalized_items), 4) if normalized_items else 0.0
    final_score = weighted_score(normalized_items)
    return {
        "add_model": add_model,
        "workspace_dir": str(workspace_dir),
        "conversation_idx": conversation_idx,
        "memory_count": memory_count,
        "rubrics": normalized_items,
        "conversation_score": final_score,
        "arithmetic_mean_score": arithmetic_mean,
        "summary": str(payload.get("summary") or "").strip(),
    }


async def score_one(
    *,
    llm: PipelineLLM,
    prompt: str,
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    add_model: str,
    workspace_dir: Path,
    memory_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        workspace_dir=workspace_dir,
        memory_count=memory_count,
    )
    result["evaluator_meta"] = meta
    return result, meta


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        row = {
            "add_model": result["add_model"],
            "workspace_dir": result["workspace_dir"],
            "conversation_idx": result["conversation_idx"],
            "memory_count": result["memory_count"],
            "conversation_score": result["conversation_score"],
            "arithmetic_mean_score": result["arithmetic_mean_score"],
        }
        for rubric in result["rubrics"]:
            row[rubric["id"]] = rubric["score"]
        rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Score v5 final add memories with fact-only continuous conversation rubric")
    parser.add_argument("--models", type=str, default="minimax,deepseek,gemini")
    parser.add_argument("--conversation-indices", type=str, default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--rubric-path", type=str, default=str(DEFAULT_RUBRIC_PATH))
    parser.add_argument("--prompt-path", type=str, default=str(DEFAULT_PROMPT_PATH))
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator config")
    llm = PipelineLLM(api_key=evaluator_key, api_base=evaluator_base, model=evaluator_model, max_tokens=12000)

    run_name = f"v5_fact_continuous_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")

    dataset = load_dataset()
    rubric_path = Path(args.rubric_path)
    if not rubric_path.is_absolute():
        rubric_path = (ROOT / rubric_path).resolve()
    prompt_path = Path(args.prompt_path)
    if not prompt_path.is_absolute():
        prompt_path = (ROOT / prompt_path).resolve()
    rubrics = load_rubrics(rubric_path)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    include_indices = (
        {int(item.strip()) for item in args.conversation_indices.split(",") if item.strip()}
        if args.conversation_indices.strip()
        else None
    )
    targets, missing = build_targets(selected_models)
    logger.log(f"models={selected_models}")
    logger.log(f"conversation_indices={sorted(include_indices) if include_indices else 'all'}")
    logger.log(f"rubric_path={rubric_path}")
    logger.log(f"prompt_path={prompt_path}")

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: list[dict[str, Any]] = []

    async def _run_one(target: WorkspaceTarget, conversation_idx: int, conv_entry: dict[str, Any]) -> dict[str, Any]:
        conversation_text = format_conversation_text(dataset[conversation_idx])
        final_memory = final_memory_from_entry(conv_entry)
        prompt = build_prompt(
            prompt_template=prompt_template,
            rubrics=rubrics,
            conversation_text=conversation_text,
            fact_memory_text=format_memory_text(final_memory),
        )
        async with semaphore:
            logger.log(
                f"score model={target.add_model} conv={conversation_idx} memory_items={len(final_memory)}"
            )
            result, _meta = await score_one(
                llm=llm,
                prompt=prompt,
                rubrics=rubrics,
                conversation_idx=conversation_idx,
                add_model=target.add_model,
                workspace_dir=target.workspace_dir,
                memory_count=len(final_memory),
            )
            logger.log(
                f"done model={target.add_model} conv={conversation_idx} "
                f"score={result['conversation_score']:.4f} mean={result['arithmetic_mean_score']:.4f}"
            )
            return result

    coroutines: list[Any] = []
    for target in targets:
        snapshot = load_json(target.workspace_dir / "add_snapshot.json")
        if not isinstance(snapshot, list):
            raise ValueError(f"snapshot must be a list: {target.workspace_dir / 'add_snapshot.json'}")
        index_to_entry = {int(entry.get("conversation_idx")): entry for entry in snapshot}
        for conversation_idx in range(len(dataset)):
            if include_indices is not None and conversation_idx not in include_indices:
                continue
            conv_entry = index_to_entry.get(conversation_idx)
            if conv_entry is None:
                missing.append(
                    {
                        "add_model": target.add_model,
                        "conversation_idx": conversation_idx,
                        "status": "missing_conversation",
                    }
                )
                continue
            coroutines.append(_run_one(target, conversation_idx, conv_entry))

    for result in await asyncio.gather(*coroutines):
        results.append(result)

    results.sort(key=lambda item: (str(item["add_model"]), int(item["conversation_idx"])))
    by_model: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_model.setdefault(str(result["add_model"]), []).append(result)

    summary_models: dict[str, Any] = {}
    for model, rows in by_model.items():
        scores = [float(row["conversation_score"]) for row in rows]
        means = [float(row["arithmetic_mean_score"]) for row in rows]
        summary_models[model] = {
            "conversation_count": len(rows),
            "fact_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "arithmetic_mean_score": round(sum(means) / len(means), 4) if means else 0.0,
            "workspace_dir": rows[0]["workspace_dir"] if rows else "",
        }

    summary = {
        "evaluator_model": evaluator_model,
        "rubric_path": str(rubric_path),
        "prompt_path": str(prompt_path),
        "models": summary_models,
        "missing": missing,
        "weights": WEIGHTS,
        "hard_cap_threshold": HARD_CAP_THRESHOLD,
        "hard_cap_value": HARD_CAP_VALUE,
    }

    (output_dir / "rubric_scores.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "evaluator_model": evaluator_model,
                "results": results,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(output_dir / "rubric_scores.csv", results)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for model, model_summary in summary_models.items():
        logger.log(
            f"final model={model} fact_score={model_summary['fact_score']:.4f} "
            f"mean={model_summary['arithmetic_mean_score']:.4f}"
        )
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
