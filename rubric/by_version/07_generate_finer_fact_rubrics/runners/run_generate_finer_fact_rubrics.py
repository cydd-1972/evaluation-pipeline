from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.infra.env import load_runtime_env
from core.infra.llm_client import PipelineLLM


RUBRIC_DIR = ROOT / "rubric"
PROMPTS_DIR = RUBRIC_DIR / "prompts"
RUBRICS_DIR = RUBRIC_DIR / "rubrics"
OUTPUTS_DIR = RUBRIC_DIR / "outputs"
PROMPT_PATH = PROMPTS_DIR / "prompt_generate_finer_fact_rubrics.txt"
BASELINE_RUBRIC_PATH = RUBRICS_DIR / "conversation_fact_rubrics_continuous.json"
SECRETS_PATH = ROOT / "configs" / "matrix_secrets.yaml"

DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "gemini": {
        "model": "gemini-3.1-flash-lite-preview",
        "llm_thinking_mode": "",
    },
    "minimax": {
        "model": "MiniMax-M2.7",
        "llm_thinking_mode": "split",
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "llm_thinking_mode": "",
    },
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


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"yaml must be a mapping: {path}")
    return payload


def resolve_model_spec(model_id: str) -> dict[str, str]:
    normalized = str(model_id or "").strip().lower()
    if normalized not in DEFAULT_MODELS:
        raise ValueError(f"unsupported model_id: {model_id}")
    secrets = load_yaml(SECRETS_PATH)
    secret_block = secrets.get(normalized) or {}
    if not isinstance(secret_block, dict):
        raise ValueError(f"invalid secret block for {normalized}")
    api_key = str(secret_block.get("api_key") or "").strip()
    api_base = str(secret_block.get("api_base") or "").strip()
    if not (api_key and api_base):
        raise ValueError(f"model {normalized} missing api_key/api_base in {SECRETS_PATH}")
    defaults = DEFAULT_MODELS[normalized]
    return {
        "id": normalized,
        "model": defaults["model"],
        "api_key": api_key,
        "api_base": api_base,
        "llm_thinking_mode": defaults.get("llm_thinking_mode", ""),
    }


def build_prompt() -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    baseline_rubrics = load_json(BASELINE_RUBRIC_PATH)
    return template.format(
        baseline_rubrics_json=json.dumps(baseline_rubrics, ensure_ascii=False, indent=2)
    )


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    rubrics = payload.get("rubrics")
    if not isinstance(rubrics, list) or not rubrics:
        raise ValueError("payload.rubrics must be a non-empty list")
    normalized_rubrics: list[dict[str, Any]] = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, dict):
            raise ValueError(f"rubric[{index}] must be an object")
        rubric_id = str(rubric.get("id") or "").strip()
        question = str(rubric.get("question") or "").strip()
        if not rubric_id:
            raise ValueError(f"rubric[{index}] missing key: id")
        if not question:
            raise ValueError(f"rubric[{index}] missing key: question")
        name = str(rubric.get("name") or "").strip()
        if not name:
            name = rubric_id.replace("_", " ").title()
        why_it_matters = str(rubric.get("why_it_matters") or "").strip()
        if not why_it_matters:
            why_it_matters = "Model did not provide an explicit explanation."
        failure_examples = rubric.get("failure_examples")
        if not isinstance(failure_examples, list):
            raise ValueError(f"rubric[{index}].failure_examples must be a list")
        normalized_rubrics.append(
            {
                "id": rubric_id,
                "name": name,
                "question": question,
                "why_it_matters": why_it_matters,
                "failure_examples": [str(item).strip() for item in failure_examples if str(item).strip()],
            }
        )
    payload["task_summary"] = str(payload.get("task_summary") or "").strip()
    payload["design_rationale"] = str(payload.get("design_rationale") or "").strip()
    payload["rubrics"] = normalized_rubrics
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate finer-grained fact rubrics with multiple models")
    parser.add_argument("--models", type=str, default="gemini,deepseek,minimax")
    args = parser.parse_args()

    load_runtime_env()
    prompt = build_prompt()
    models = [item.strip() for item in args.models.split(",") if item.strip()]

    run_name = f"generate_finer_fact_rubrics_{_now_tag()}"
    output_dir = OUTPUTS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir / "run.log")
    logger.log(f"run_name={run_name}")
    logger.log(f"models={models}")

    outputs: dict[str, Any] = {}

    for model_id in models:
        spec = resolve_model_spec(model_id)
        logger.log(f"generate model={model_id} api_base={spec['api_base']} llm_model={spec['model']}")
        llm = PipelineLLM(
            api_key=spec["api_key"],
            api_base=spec["api_base"],
            model=spec["model"],
            max_tokens=12000,
        )
        if spec.get("llm_thinking_mode"):
            import os

            os.environ["PIPELINE_LLM_THINKING_MODE"] = str(spec["llm_thinking_mode"])
        else:
            import os

            os.environ.pop("PIPELINE_LLM_THINKING_MODE", None)

        payload, meta = llm.chat_json_object_with_meta(
            prompt,
            required_key="rubrics",
            temperature=0.0,
            max_attempts=4,
        )
        validated = validate_payload(payload)
        result = {
            "generator_model_id": model_id,
            "generator_model_name": spec["model"],
            "api_base": spec["api_base"],
            "prompt_path": str(PROMPT_PATH),
            "baseline_rubric_path": str(BASELINE_RUBRIC_PATH),
            "result": validated,
            "meta": meta,
        }
        outputs[model_id] = result
        (output_dir / f"{model_id}_finer_fact_rubrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.log(f"done model={model_id} rubric_count={len(validated['rubrics'])}")

    summary = {
        "run_name": run_name,
        "models": {
            model_id: {
                "generator_model_name": payload["generator_model_name"],
                "rubric_count": len(payload["result"]["rubrics"]),
                "output_file": str(output_dir / f"{model_id}_finer_fact_rubrics.json"),
            }
            for model_id, payload in outputs.items()
        },
        "prompt_path": str(PROMPT_PATH),
        "baseline_rubric_path": str(BASELINE_RUBRIC_PATH),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
