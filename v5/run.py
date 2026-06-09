"""v5 entry: add model is selectable; search/answer stay on MiniMax; eval judge stays unchanged."""
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

from core.infra.env import load_runtime_env
from core.infra.flat_export import flattened_eval_output_path
from core.pipeline.runner import PIPELINE_STEPS, run_pipeline_from_config

DEFAULT_CONFIG_NAME = "config.yaml"
SECRETS_PATH = PIPELINE_DIR / "configs" / "matrix_secrets.yaml"
RESETTABLE_STEPS = tuple(step for step in PIPELINE_STEPS if step != "add")
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
FIXED_DOWNSTREAM_MODEL_ID = "minimax"


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


def _resolve_model_spec(
    *,
    model_id: str,
    model_name_override: str | None,
    secrets_path: Path,
) -> dict[str, str]:
    normalized = str(model_id or "").strip().lower()
    if normalized not in DEFAULT_MODELS:
        raise ValueError(f"unsupported model_id: {model_id} (use gemini|minimax|deepseek)")
    secrets = _load_model_secrets(secrets_path)
    secret_block = secrets.get(normalized) or {}
    api_key = str(secret_block.get("api_key") or "").strip()
    api_base = str(secret_block.get("api_base") or "").strip()
    if not (api_key and api_base):
        raise ValueError(f"model {normalized} missing api_key/api_base in {secrets_path}")
    defaults = DEFAULT_MODELS[normalized]
    return {
        "id": normalized,
        "model": str(model_name_override or defaults["model"]).strip(),
        "api_key": api_key,
        "api_base": api_base,
        "llm_thinking_mode": str(defaults.get("llm_thinking_mode") or "").strip().lower(),
    }


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


def _resolve_workspace_dir(config: dict[str, Any], *, version_dir: Path) -> Path:
    base_raw = str(config.get("workspace_base_dir") or "workspaces")
    base_path = Path(base_raw)
    base_dir = base_path if base_path.is_absolute() else version_dir / base_path
    return base_dir / str(config.get("workspace_name") or "smoke")


def _reset_output_paths(workspace_dir: Path, *, answer_mode: str) -> dict[str, list[Path]]:
    answer_output = workspace_dir / f"search_results_answer{answer_mode}.json"
    eval_output = workspace_dir / f"evaluation_metrics_answer{answer_mode}.json"
    flattened_eval = flattened_eval_output_path(eval_output)
    score_output = workspace_dir / f"score_summary_answer{answer_mode}.json"
    return {
        "search": [
            workspace_dir / "search_results.json",
            answer_output,
            eval_output,
            flattened_eval,
            score_output,
        ],
        "answer": [
            answer_output,
            eval_output,
            flattened_eval,
            score_output,
        ],
        "eval": [
            eval_output,
            flattened_eval,
            score_output,
        ],
        "score": [
            score_output,
        ],
    }


def _archive_step_outputs(
    *,
    workspace_dir: Path,
    step: str,
    answer_mode: str,
    archive_dir_name: str,
) -> list[Path]:
    output_paths = _reset_output_paths(workspace_dir, answer_mode=answer_mode).get(step, [])
    existing = [path for path in output_paths if path.exists()]
    if not existing:
        return []
    archive_root = workspace_dir / archive_dir_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = archive_root / f"{stamp}_{step}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived_to: list[Path] = []
    manifest: list[dict[str, str]] = []
    for source in existing:
        target = archive_dir / source.name
        suffix_index = 1
        while target.exists():
            target = archive_dir / f"{source.stem}_{suffix_index}{source.suffix}"
            suffix_index += 1
        source.rename(target)
        archived_to.append(target)
        manifest.append({"source": str(source), "archived_to": str(target)})
    (archive_dir / "manifest.json").write_text(
        json.dumps(
            {
                "step": step,
                "answer_mode": answer_mode,
                "archived_at": datetime.now().isoformat(timespec="seconds"),
                "files": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return archived_to


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


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo pipeline (v5)")
    parser.add_argument("--config", type=Path, default=VERSION_DIR / DEFAULT_CONFIG_NAME)
    parser.add_argument("--model-id", required=True, choices=sorted(DEFAULT_MODELS.keys()))
    parser.add_argument("--model-name", default=None, help="override default model name for the selected model id")
    parser.add_argument("--secrets", type=Path, default=SECRETS_PATH)
    parser.add_argument("--start-from-step", "--from", dest="start_from_step", default="add", choices=PIPELINE_STEPS)
    parser.add_argument("--end-at-step", "--only", dest="end_at_step", default=None, choices=PIPELINE_STEPS)
    parser.add_argument(
        "--reset-outputs-from-step",
        default=None,
        choices=RESETTABLE_STEPS,
        help="archive old outputs for this step and downstream files before rerunning; never deletes directly",
    )
    parser.add_argument("--no-tee-log", action="store_true")
    parser.add_argument("--print-config", action="store_true", help="print resolved config and exit")
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config, version_dir=VERSION_DIR)
    config = _load_yaml(config_path)
    load_runtime_env()
    add_model_spec = _resolve_model_spec(
        model_id=args.model_id,
        model_name_override=args.model_name,
        secrets_path=args.secrets if args.secrets.is_absolute() else PIPELINE_DIR / args.secrets,
    )
    downstream_model_spec = _resolve_model_spec(
        model_id=FIXED_DOWNSTREAM_MODEL_ID,
        model_name_override=None,
        secrets_path=args.secrets if args.secrets.is_absolute() else PIPELINE_DIR / args.secrets,
    )
    _apply_model_env(add_model_spec)
    config = _materialize_templates(
        config,
        add_model_spec=add_model_spec,
        downstream_model_spec=downstream_model_spec,
    )
    workspace_dir = _resolve_workspace_dir(config, version_dir=VERSION_DIR)
    archive_dir_name = str(config.get("reset_archive_dir") or "archived_outputs").strip() or "archived_outputs"

    if args.print_config:
        payload = {
            "add_model": add_model_spec,
            "downstream_model": downstream_model_spec,
            "config_path": str(config_path),
            "config": config,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(
        f"[v5] add_model_id={add_model_spec['id']} add_model={add_model_spec['model']} "
        f"api_base={add_model_spec['api_base']}",
        flush=True,
    )
    print(
        f"[v5] search_answer_model_id={downstream_model_spec['id']} "
        f"search_answer_model={downstream_model_spec['model']}",
        flush=True,
    )
    if add_model_spec.get("llm_thinking_mode"):
        print(f"[v5] add_llm_thinking_mode={add_model_spec['llm_thinking_mode']}", flush=True)
    if args.reset_outputs_from_step:
        archived = _archive_step_outputs(
            workspace_dir=workspace_dir,
            step=args.reset_outputs_from_step,
            answer_mode=str(config.get("answer_prompt_mode") or "history"),
            archive_dir_name=archive_dir_name,
        )
        if archived:
            print(
                f"[v5] archived {len(archived)} file(s) before rerun: "
                + ", ".join(str(path) for path in archived),
                flush=True,
            )
        else:
            print(f"[v5] no existing outputs to archive for step={args.reset_outputs_from_step}", flush=True)

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
