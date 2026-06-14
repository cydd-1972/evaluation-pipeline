from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.infra.env import evaluator_settings, load_runtime_env
from core.infra.llm_client import PipelineLLM


RUBRIC_DIR = ROOT / "rubric"
PROMPTS_DIR = RUBRIC_DIR / "prompts"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"
PROMPT_PATH = PROMPTS_DIR / "prompt_deduplicate_generated_fact_rubrics_conservative_simplejson.txt"

DEFAULT_INPUTS = {
    "gemini": OUTPUTS_DIR / "generate_finer_fact_rubrics_20260613_122735" / "gemini_finer_fact_rubrics.json",
    "deepseek": OUTPUTS_DIR / "generate_finer_fact_rubrics_20260613_122735" / "deepseek_finer_fact_rubrics.json",
    "minimax": OUTPUTS_DIR / "generate_finer_fact_rubrics_20260613_122952" / "minimax_finer_fact_rubrics.json",
}


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


def build_candidate_payload(models: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"candidate_sets": []}
    for model in models:
        path = DEFAULT_INPUTS[model]
        data = load_json(path)
        payload["candidate_sets"].append(
            {
                "model": model,
                "generator_model_name": data.get("generator_model_name"),
                "rubrics": (data.get("result") or {}).get("rubrics", []),
            }
        )
    return payload


def build_prompt(models: list[str]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    candidates = build_candidate_payload(models)
    return template + "\n\nCandidate rubric sets:\n" + json.dumps(candidates, ensure_ascii=False, indent=2)


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be object")
    for key in ("merged_groups", "kept_separate_notes", "consolidated_rubrics"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{key} must be list")
    for index, item in enumerate(payload["merged_groups"]):
        if not isinstance(item, dict):
            raise ValueError(f"merged_groups[{index}] must be object")
        if not isinstance(item.get("source_refs"), list):
            raise ValueError(f"merged_groups[{index}].source_refs must be list")
    for index, item in enumerate(payload["kept_separate_notes"]):
        if not isinstance(item, dict):
            raise ValueError(f"kept_separate_notes[{index}] must be object")
        if not isinstance(item.get("source_refs"), list):
            raise ValueError(f"kept_separate_notes[{index}].source_refs must be list")
    for index, item in enumerate(payload["consolidated_rubrics"]):
        if not isinstance(item, dict):
            raise ValueError(f"consolidated_rubrics[{index}] must be object")
        for key in ("id", "name", "question", "why_it_matters"):
            if not isinstance(item.get(key), str):
                raise ValueError(f"consolidated_rubrics[{index}].{key} must be string")
        if not isinstance(item.get("failure_examples"), list):
            raise ValueError(f"consolidated_rubrics[{index}].failure_examples must be list")
        if not isinstance(item.get("source_refs"), list):
            raise ValueError(f"consolidated_rubrics[{index}].source_refs must be list")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservatively deduplicate generated fact rubrics with qwen-friendly schema")
    parser.add_argument("--models", type=str, default="gemini,deepseek,minimax")
    parser.add_argument("--evaluator-model", type=str, default="")
    parser.add_argument("--evaluator-api-base", type=str, default="")
    parser.add_argument("--evaluator-api-key", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=7000)
    args = parser.parse_args()

    load_runtime_env()
    evaluator_model, evaluator_base, evaluator_key = evaluator_settings()
    if args.evaluator_model.strip():
        evaluator_model = args.evaluator_model.strip()
    if args.evaluator_api_base.strip():
        evaluator_base = args.evaluator_api_base.strip()
    if args.evaluator_api_key.strip():
        evaluator_key = args.evaluator_api_key.strip()
    if not (evaluator_model and evaluator_base and evaluator_key):
        raise RuntimeError("missing evaluator config")

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    run_name = f"deduplicate_generated_fact_rubrics_qwen_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"evaluator_model={evaluator_model}")
    logger.log(f"models={models}")

    prompt = build_prompt(models)
    llm = PipelineLLM(
        api_key=evaluator_key,
        api_base=evaluator_base,
        model=evaluator_model,
        max_tokens=max(1, int(args.max_tokens)),
    )

    last_error: Exception | None = None
    validated: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None
    current_prompt = prompt
    for retry_index in range(3):
        payload, meta = llm.chat_json_object_with_meta(
            current_prompt,
            required_key="consolidated_rubrics",
            temperature=0.0,
            max_attempts=4,
        )
        try:
            validated = validate_payload(payload)
            break
        except Exception as exc:
            last_error = exc
            logger.log(f"validation_retry={retry_index + 1} error={exc}")
            current_prompt = (
                prompt
                + "\n\nIMPORTANT CORRECTION:\n"
                + "All fields that end with `_refs` must be JSON arrays of strings. "
                + "`failure_examples` must be a JSON array of strings. "
                + "Do not replace arrays with plain strings."
            )
    if validated is None:
        raise ValueError(f"dedup payload validation failed after retries: {last_error}")

    result = {
        "run_name": run_name,
        "evaluator_model": evaluator_model,
        "prompt_path": str(PROMPT_PATH),
        "input_files": {model: str(DEFAULT_INPUTS[model]) for model in models},
        "result": validated,
        "meta": meta,
    }
    (output_dir / "deduplicated_fact_rubrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "run_name": run_name,
        "evaluator_model": evaluator_model,
        "models": models,
        "consolidated_rubric_count": len(validated["consolidated_rubrics"]),
        "merged_group_count": len(validated["merged_groups"]),
        "kept_separate_note_count": len(validated["kept_separate_notes"]),
        "output_file": str(output_dir / "deduplicated_fact_rubrics.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(f"consolidated_rubric_count={summary['consolidated_rubric_count']}")
    logger.log(f"merged_group_count={summary['merged_group_count']}")
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
