"""v6 entry: add model selectable (qwen3-4b / qwen3-14b / dpsk-flash), downstream search/answer fixed to DeepSeek."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

VERSION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = VERSION_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from core.infra.env import evaluator_dashscope_keys, load_runtime_env
from core.pipeline.runner import PIPELINE_STEPS, run_pipeline_from_config

DEFAULT_CONFIG_NAME = "config.yaml"
SECRETS_PATH = PIPELINE_DIR / "configs" / "matrix_secrets.yaml"
FIXED_DOWNSTREAM_MODEL_ID = "deepseek"


class _TimestampedWriter:
    def __init__(self, wrapped: Any) -> None:
        self._wrapped = wrapped
        self._buffer = ""
        self._at_line_start = True

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += str(text)
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index < 0:
                break
            line = self._buffer[: newline_index + 1]
            self._buffer = self._buffer[newline_index + 1 :]
            self._emit(line)
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit(self._buffer)
            self._buffer = ""
        self._wrapped.flush()

    def _emit(self, text: str) -> None:
        payload = text
        if self._at_line_start and payload:
            payload = f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {payload}"
        self._wrapped.write(payload)
        self._at_line_start = payload.endswith("\n")

    def isatty(self) -> bool:
        return bool(getattr(self._wrapped, "isatty", lambda: False)())


def _install_timestamped_logging() -> None:
    if not isinstance(sys.stdout, _TimestampedWriter):
        sys.stdout = _TimestampedWriter(sys.stdout)
    if not isinstance(sys.stderr, _TimestampedWriter):
        sys.stderr = _TimestampedWriter(sys.stderr)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _load_model_secrets(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing model secrets file: {path}")
    payload = _load_yaml(path)
    secrets: dict[str, dict[str, str]] = {}
    for model_id, spec in payload.items():
        if not isinstance(spec, dict):
            continue
        secrets[str(model_id).strip().lower()] = {
            "api_key": str(spec.get("api_key") or "").strip(),
            "api_base": str(spec.get("api_base") or "").strip(),
        }
    return secrets


def _dashscope_model_spec(model_name: str) -> dict[str, str]:
    load_runtime_env()
    keys = evaluator_dashscope_keys()
    api_key = keys[0] if keys else ""
    api_base = os.getenv("EVALUATOR_DASHSCOPE_API_BASE", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not (api_key and api_base and model_name):
        raise ValueError("missing DashScope config for qwen add model")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model_name,
        "llm_thinking_mode": "",
    }


def _siliconflow_model_spec(model_name: str, *, purpose: str) -> dict[str, str]:
    load_runtime_env()
    api_key = os.getenv("EVALUATOR_API_KEY", "").strip()
    api_base = os.getenv("EVALUATOR_API_BASE", "").strip() or "https://api.siliconflow.cn/v1"
    if not (api_key and api_base and model_name):
        raise ValueError(f"missing SiliconFlow config for {purpose}")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model_name,
        "llm_thinking_mode": "",
    }


def _deepseek_env_spec(model_name: str) -> dict[str, str]:
    api_key = str(os.getenv("DEEPSEEK_API_KEY") or "").strip()
    api_base = str(os.getenv("DEEPSEEK_API_BASE") or "").strip() or "https://api.deepseek.com"
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if not (api_key and api_base and model_name):
        raise ValueError("missing DeepSeek config for downstream model")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model_name,
        "llm_thinking_mode": "",
    }


def _resolve_model_spec(
    *,
    model_id: str,
    model_name_override: str | None,
    secrets_path: Path,
) -> dict[str, str]:
    normalized = str(model_id or "").strip().lower()
    if normalized == "qwen3-4b":
        spec = _siliconflow_model_spec(
            model_name_override or "Qwen/Qwen3-4B",
            purpose="qwen add model",
        )
        return {"id": normalized, **spec}
    if normalized == "qwen3-14b":
        spec = _siliconflow_model_spec(
            model_name_override or "Qwen/Qwen3-14B",
            purpose="qwen add model",
        )
        return {"id": normalized, **spec}
    if normalized in {"dpsk-flash", "deepseek-flash", "deepseek"}:
        requested_model = str(model_name_override or "").strip()
        if normalized == "dpsk-flash" and not requested_model:
            spec = _deepseek_env_spec("deepseek-ai/DeepSeek-V4-Flash")
        else:
            siliconflow_model = requested_model or "deepseek-ai/DeepSeek-V4-Flash"
            if siliconflow_model == "deepseek-v4-flash":
                siliconflow_model = "deepseek-ai/DeepSeek-V4-Flash"
            spec = _siliconflow_model_spec(
                siliconflow_model,
                purpose="deepseek downstream model",
            )
        return {"id": "dpsk-flash", **spec}
    raise ValueError("unsupported model_id: use qwen3-4b | qwen3-14b | dpsk-flash")


def _apply_model_env(model_spec: dict[str, str]) -> None:
    os.environ["OPENAI_API_KEY"] = model_spec["api_key"]
    os.environ["OPENAI_API_BASE"] = model_spec["api_base"]
    os.environ["OPENAI_MODEL"] = model_spec["model"]
    os.environ["key"] = model_spec["api_key"]
    os.environ["api_base"] = model_spec["api_base"]
    os.environ["model_name"] = model_spec["model"]
    thinking_mode = str(model_spec.get("llm_thinking_mode") or "").strip()
    if thinking_mode:
        os.environ["PIPELINE_LLM_THINKING_MODE"] = thinking_mode
    else:
        os.environ.pop("PIPELINE_LLM_THINKING_MODE", None)


def _client_config(model_spec: dict[str, str]) -> dict[str, str]:
    payload = {
        "api_key": model_spec["api_key"],
        "api_base": model_spec["api_base"],
        "model": model_spec["model"],
    }
    thinking_mode = str(model_spec.get("llm_thinking_mode") or "").strip()
    if thinking_mode:
        payload["llm_thinking_mode"] = thinking_mode
    return payload


def _resolve_config_path(raw: Path, *, version_dir: Path) -> Path:
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    candidate = version_dir / raw
    if candidate.exists():
        return candidate
    pipeline_candidate = version_dir.parent / raw
    if pipeline_candidate.exists():
        return pipeline_candidate
    return candidate


def _materialize_templates(
    config: dict[str, Any],
    *,
    add_model_spec: dict[str, str],
    downstream_model_spec: dict[str, str],
) -> dict[str, Any]:
    resolved = dict(config)
    format_vars = {
        "model_id": add_model_spec["id"],
        "model_name": add_model_spec["model"],
    }
    for key in ("workspace_name", "workspace_db_name", "database_prefix", "progress_label"):
        raw = resolved.get(key)
        if isinstance(raw, str) and "{" in raw:
            resolved[key] = raw.format(**format_vars)
    resolved["selected_model_id"] = add_model_spec["id"]
    resolved["selected_model_name"] = add_model_spec["model"]
    resolved["selected_api_base"] = add_model_spec["api_base"]
    resolved["add_llm_client"] = _client_config(add_model_spec)
    resolved["search_llm_client"] = _client_config(downstream_model_spec)
    resolved["answer_llm_client"] = _client_config(downstream_model_spec)
    resolved["fixed_downstream_model_id"] = downstream_model_spec["id"]
    resolved["fixed_downstream_model_name"] = downstream_model_spec["model"]
    return resolved


def _workspace_dir_from_config(config: dict[str, Any]) -> Path:
    base_raw = str(config.get("workspace_base_dir") or "workspaces")
    base_path = Path(base_raw)
    if not base_path.is_absolute():
        base_path = VERSION_DIR / base_path
    return base_path / str(config.get("workspace_name") or "smoke")


def main() -> None:
    _install_timestamped_logging()
    parser = argparse.ArgumentParser(description="LoCoMo pipeline (v6 summary add)")
    parser.add_argument("--config", type=Path, default=VERSION_DIR / DEFAULT_CONFIG_NAME)
    parser.add_argument("--model-id", required=True, choices=("qwen3-4b", "qwen3-14b", "dpsk-flash"))
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--secrets", type=Path, default=SECRETS_PATH)
    parser.add_argument("--start-from-step", "--from", dest="start_from_step", default="add", choices=PIPELINE_STEPS)
    parser.add_argument("--end-at-step", "--only", dest="end_at_step", default=None, choices=PIPELINE_STEPS)
    parser.add_argument("--no-tee-log", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config, version_dir=VERSION_DIR)
    config = _load_yaml(config_path)
    load_runtime_env()

    secrets_path = args.secrets if args.secrets.is_absolute() else PIPELINE_DIR / args.secrets
    add_model_spec = _resolve_model_spec(
        model_id=args.model_id,
        model_name_override=args.model_name,
        secrets_path=secrets_path,
    )
    downstream_model_spec = _resolve_model_spec(
        model_id=FIXED_DOWNSTREAM_MODEL_ID,
        model_name_override="deepseek-v4-flash",
        secrets_path=secrets_path,
    )

    os.environ["PIPELINE_LLM_API_RETRY_ATTEMPTS"] = "8"
    os.environ["PIPELINE_API_FAILURE_MAX"] = "8"

    _apply_model_env(add_model_spec)
    config = _materialize_templates(
        config,
        add_model_spec=add_model_spec,
        downstream_model_spec=downstream_model_spec,
    )
    workspace_dir = _workspace_dir_from_config(config)
    trace_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.environ["PIPELINE_LLM_TRACE_PATH"] = str(workspace_dir / f"llm_trace_{trace_stamp}.jsonl")

    if args.print_config:
        payload = {
            "add_model": add_model_spec,
            "downstream_model": downstream_model_spec,
            "config_path": str(config_path),
            "config": config,
            "llm_trace_path": os.environ["PIPELINE_LLM_TRACE_PATH"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(
        f"[v6] add_model_id={add_model_spec['id']} add_model={add_model_spec['model']} "
        f"api_base={add_model_spec['api_base']}",
        flush=True,
    )
    print(
        f"[v6] search_answer_model_id={downstream_model_spec['id']} "
        f"search_answer_model={downstream_model_spec['model']}",
        flush=True,
    )
    print(f"[v6] llm_trace={os.environ['PIPELINE_LLM_TRACE_PATH']}", flush=True)
    print("[v6] retry_policy=8 attempts then stop", flush=True)

    import asyncio

    asyncio.run(
        run_pipeline_from_config(
            config,
            start_from_step=args.start_from_step,
            end_at_step=args.end_at_step,
            config_path=config_path,
            load_env=False,
            pipeline_dir=PIPELINE_DIR,
            version_dir=VERSION_DIR,
            tee_log=not args.no_tee_log,
        )
    )


if __name__ == "__main__":
    main()
