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
SNAPSHOT_PATH = ROOT / "datasets" / "add_snapshot_locomo.json"
RUBRICS_DIR = RUBRIC_DIR / "rubrics"
PROMPTS_DIR = RUBRIC_DIR / "prompts"
RUBRIC_PATH = RUBRICS_DIR / "conversation_bucket_summary_rubrics.json"
PROMPT_PATH = PROMPTS_DIR / "prompt_conversation_bucket_summary_continuous.txt"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"

WEIGHTS = {
    "summary_sourceability": 3.0,
    "summary_factual_correctness": 3.0,
    "summary_temporal_consistency": 2.0,
    "bucket_schema_fit": 2.0,
    "summary_abstraction_quality": 2.0,
    "cross_fact_integration": 2.0,
    "nonredundancy_across_summaries": 1.0,
    "conciseness_and_density": 1.0,
    "structural_consistency": 1.0,
    "coverage_of_summary_targets": 2.0,
}
HARD_CAP_RUBRICS = (
    "summary_sourceability",
    "summary_factual_correctness",
)
HARD_CAP_THRESHOLD = 0.3
HARD_CAP_VALUE = 0.4


@dataclass
class ConversationTarget:
    conversation_idx: int
    conversation_entry: dict[str, Any]


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


def load_snapshot() -> list[dict[str, Any]]:
    data = load_json(SNAPSHOT_PATH)
    if not isinstance(data, list):
        raise ValueError("snapshot must be a list")
    return data


def load_rubrics() -> list[dict[str, str]]:
    data = load_json(RUBRIC_PATH)
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


def final_session_memory(conversation_entry: dict[str, Any]) -> list[dict[str, Any]]:
    sessions = conversation_entry.get("sessions") or []
    if not sessions:
        return []
    last_session = sessions[-1]
    memory = last_session.get("memory") or []
    return memory if isinstance(memory, list) else []


def split_memory_by_bucket(memory: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    facts: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for item in memory:
        if str(item.get("bucket") or "").strip() == "fact":
            facts.append(item)
        else:
            summaries.append(item)
    return facts, summaries


def format_memory_text(memory: list[dict[str, Any]]) -> str:
    if not memory:
        return "(empty)"
    lines: list[str] = []
    for index, item in enumerate(memory, start=1):
        lines.append(
            f"{index}. id={item.get('id','')}; bucket={item.get('bucket','')}; event={item.get('event','')}; "
            f"anchor_time={item.get('anchor_time','')}; text={' '.join(str(item.get('text') or '').split())}"
        )
    return "\n".join(lines)


def build_prompt(
    *,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_text: str,
    fact_memory_text: str,
    summary_memory_text: str,
) -> str:
    return prompt_template.format(
        rubrics_json=json.dumps(rubrics, ensure_ascii=False, indent=2),
        conversation_text=conversation_text,
        fact_memory_text=fact_memory_text,
        summary_memory_text=summary_memory_text,
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
    by_id = {}
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
        "conversation_idx": conversation_idx,
        "rubrics": normalized_items,
        "conversation_score": final_score,
        "arithmetic_mean_score": arithmetic_mean,
        "summary": str(payload.get("summary") or "").strip(),
    }


async def score_one_conversation(
    *,
    llm: PipelineLLM,
    prompt_template: str,
    rubrics: list[dict[str, str]],
    conversation_idx: int,
    conversation_text: str,
    fact_memory_text: str,
    summary_memory_text: str,
) -> dict[str, Any]:
    prompt = build_prompt(
        prompt_template=prompt_template,
        rubrics=rubrics,
        conversation_text=conversation_text,
        fact_memory_text=fact_memory_text,
        summary_memory_text=summary_memory_text,
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
    )
    result["evaluator_meta"] = meta
    return result


def write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for conversation in results:
        row = {
            "conversation_idx": conversation["conversation_idx"],
            "fact_count": conversation["fact_count"],
            "summary_count": conversation["summary_count"],
            "conversation_score": conversation["conversation_score"],
            "arithmetic_mean_score": conversation["arithmetic_mean_score"],
        }
        for rubric in conversation.get("rubrics", []):
            row[f"rubric_{rubric['id']}"] = rubric["score"]
        rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Conversation-level continuous bucket-summary rubric scoring")
    parser.add_argument("--conversation-indices", type=str, default="", help="Comma-separated conversation indices to score")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator config")
    llm = PipelineLLM(api_key=evaluator_key, api_base=evaluator_base, model=evaluator_model, max_tokens=12000)

    run_name = f"conversation_bucket_summary_continuous_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")

    dataset = load_dataset()
    snapshot = load_snapshot()
    rubrics = load_rubrics()
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8")
    include_indices = (
        {int(item.strip()) for item in args.conversation_indices.split(",") if item.strip()}
        if args.conversation_indices.strip()
        else None
    )
    logger.log(f"conversation_indices={sorted(include_indices) if include_indices else 'all'}")

    index_to_snapshot = {int(entry.get("conversation_idx")): entry for entry in snapshot}
    targets: list[ConversationTarget] = []
    missing: list[int] = []
    for conversation_idx, _dataset_item in enumerate(dataset):
        if include_indices is not None and conversation_idx not in include_indices:
            continue
        entry = index_to_snapshot.get(conversation_idx)
        if entry is None:
            missing.append(conversation_idx)
            continue
        targets.append(ConversationTarget(conversation_idx=conversation_idx, conversation_entry=entry))

    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    results: list[dict[str, Any]] = []

    async def _run(target: ConversationTarget) -> dict[str, Any]:
        conversation_idx = target.conversation_idx
        conversation_text = format_conversation_text(dataset[conversation_idx])
        memory = final_session_memory(target.conversation_entry)
        fact_memory, summary_memory = split_memory_by_bucket(memory)
        async with semaphore:
            logger.log(
                f"score conv={conversation_idx} fact_items={len(fact_memory)} summary_items={len(summary_memory)}"
            )
            result = await score_one_conversation(
                llm=llm,
                prompt_template=prompt_template,
                rubrics=rubrics,
                conversation_idx=conversation_idx,
                conversation_text=conversation_text,
                fact_memory_text=format_memory_text(fact_memory),
                summary_memory_text=format_memory_text(summary_memory),
            )
            result["fact_count"] = len(fact_memory)
            result["summary_count"] = len(summary_memory)
            logger.log(
                f"done conv={conversation_idx} score={result['conversation_score']:.4f} mean={result['arithmetic_mean_score']:.4f}"
            )
            return result

    for result in await asyncio.gather(*[_run(target) for target in targets]):
        results.append(result)
    results.sort(key=lambda item: int(item["conversation_idx"]))
    final_scores = [float(item["conversation_score"]) for item in results]
    summary = {
        "evaluator_model": evaluator_model,
        "final_score": round(sum(final_scores) / len(final_scores), 4) if final_scores else 0.0,
        "conversation_count": len(results),
        "missing_conversation_indices": missing,
        "weights": WEIGHTS,
        "hard_cap_rubrics": list(HARD_CAP_RUBRICS),
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
    logger.log(f"final_score={summary['final_score']:.4f}")
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
