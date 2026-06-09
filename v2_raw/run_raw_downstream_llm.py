from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

VERSION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = VERSION_DIR.parent
WORKSPACES_DIR = PIPELINE_DIR / "workspaces"
LOGS_DIR = VERSION_DIR / "workspaces" / "logs"
SOURCE_RAW_ADD_DIR = WORKSPACES_DIR / "matrix" / "raw" / "add"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from core.infra.env import evaluator_dashscope_keys, load_runtime_env
from core.matrix.matrix import link_add_database_to_search_run
from core.pipeline.runner import run_pipeline_from_config


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _load_matrix_secrets(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    payload = _load_yaml(path)
    secrets: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        secrets[str(key).strip().lower()] = {
            "api_key": str(value.get("api_key") or "").strip(),
            "api_base": str(value.get("api_base") or "").strip(),
        }
    return secrets


def _copy_required_workspace_files(*, source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("workspace.json", "add_snapshot.json", "pipeline_config.json"):
        source = source_dir / name
        if not source.exists():
            if name == "pipeline_config.json":
                continue
            raise FileNotFoundError(f"missing source raw-add file: {source}")
        shutil.copy2(source, target_dir / name)
    manifest = {
        "source_workspace": str(source_dir),
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target_dir / "reused_add_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _qwen_model_spec() -> dict[str, str]:
    load_runtime_env()
    keys = evaluator_dashscope_keys()
    api_key = keys[0] if keys else ""
    api_base = os.getenv("EVALUATOR_DASHSCOPE_API_BASE", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.getenv("EVALUATOR_DASHSCOPE_MODEL", "").strip() or "qwen3-14b"
    if not (api_key and api_base and model):
        raise ValueError("missing DashScope qwen3-14b config in .env")
    return {
        "id": "qwen3",
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "llm_thinking_mode": "",
    }


def _minimax_model_spec(secrets_path: Path) -> dict[str, str]:
    load_runtime_env()
    secrets = _load_matrix_secrets(secrets_path)
    minimax = secrets.get("minimax") or {}
    api_key = str(minimax.get("api_key") or "").strip()
    api_base = str(minimax.get("api_base") or "").strip()
    model = os.getenv("OPENAI_MODEL", "").strip() or "MiniMax-M2.7"
    if not (api_key and api_base):
        raise ValueError(f"missing minimax api_key/api_base in {secrets_path}")
    return {
        "id": "minimax",
        "model": model,
        "api_key": api_key,
        "api_base": api_base,
        "llm_thinking_mode": "split",
    }


def _model_spec(model_id: str, *, secrets_path: Path) -> dict[str, str]:
    normalized = str(model_id or "").strip().lower()
    if normalized == "minimax":
        return _minimax_model_spec(secrets_path)
    if normalized in {"qwen", "qwen3", "qwen3-14b"}:
        return _qwen_model_spec()
    raise ValueError(f"unsupported model_id: {model_id} (use minimax|qwen3)")


def _pipeline_config_payload(
    *,
    target_root: Path,
    root_name: str,
    pipeline_model: dict[str, str],
    search_llm_concurrency: int,
    answer_concurrency: int,
    eval_concurrency: int,
) -> dict[str, Any]:
    raw_root = target_root / "raw"
    return {
        "dataset_path": "datasets/locomo_refined.json",
        "max_conversations": None,
        "max_questions_per_conversation": None,
        "max_sessions_per_conversation": None,
        "workspace_base_dir": str(raw_root),
        "workspace_name": "llm",
        "workspace_db_name": "matrix_raw_add",
        "database_prefix": "eval_v2_raw",
        "reset_database_on_add": False,
        "add_backend": "raw",
        "search_backend": "llm",
        "search_mode": "per_speaker",
        "answer_prompt_mode": "history",
        "search_top_k": 30,
        "concurrency": answer_concurrency,
        "search_llm_concurrency": search_llm_concurrency,
        "search_llm_client": {
            "api_key": pipeline_model["api_key"],
            "api_base": pipeline_model["api_base"],
            "model": pipeline_model["model"],
            "llm_thinking_mode": pipeline_model.get("llm_thinking_mode") or "",
        },
        "answer_llm_client": {
            "api_key": pipeline_model["api_key"],
            "api_base": pipeline_model["api_base"],
            "model": pipeline_model["model"],
            "llm_thinking_mode": pipeline_model.get("llm_thinking_mode") or "",
        },
        "parent_add_workspace": str(raw_root / "add"),
        "progress_label": f"raw/{pipeline_model['id']}-llm-full",
        "eval_concurrency": eval_concurrency,
        "eval": {
            "metrics": ["llm", "f1", "bleu"],
            "dataset_name": "locomo",
        },
        "selected_model_id": pipeline_model["id"],
        "selected_model_name": pipeline_model["model"],
        "experiment_name": root_name,
    }


def _write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _launch_background(
    *,
    script_path: Path,
    model_id: str,
    root_name: str,
    source_add_dir: Path,
    secrets_path: Path,
    search_llm_concurrency: int,
    answer_concurrency: int,
    eval_concurrency: int,
    start_from_step: str,
    end_at_step: str | None,
) -> tuple[int, Path, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOGS_DIR / f"launch_raw_downstream_{model_id}_{stamp}.out.txt"
    stderr_path = LOGS_DIR / f"launch_raw_downstream_{model_id}_{stamp}.err.txt"
    command = [
        sys.executable,
        str(script_path),
        "--model-id",
        model_id,
        "--root-name",
        root_name,
        "--source-add-dir",
        str(source_add_dir),
        "--secrets",
        str(secrets_path),
        "--search-llm-concurrency",
        str(search_llm_concurrency),
        "--answer-concurrency",
        str(answer_concurrency),
        "--eval-concurrency",
        str(eval_concurrency),
        "--start-from-step",
        start_from_step,
        "--foreground",
    ]
    if end_at_step:
        command.extend(["--end-at-step", end_at_step])
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(PIPELINE_DIR),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    time.sleep(2)
    return proc.pid, stdout_path, stderr_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse v2 raw add outputs, then run full-data llm downstream with a chosen model.")
    parser.add_argument("--model-id", required=True, choices=("minimax", "qwen3"))
    parser.add_argument("--root-name", default=None)
    parser.add_argument("--source-add-dir", type=Path, default=SOURCE_RAW_ADD_DIR)
    parser.add_argument("--secrets", type=Path, default=PIPELINE_DIR / "configs" / "matrix_secrets.yaml")
    parser.add_argument("--search-llm-concurrency", type=int, default=4)
    parser.add_argument("--answer-concurrency", type=int, default=1)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    parser.add_argument("--start-from-step", default="search", choices=("search", "answer", "eval", "score"))
    parser.add_argument("--end-at-step", default=None, choices=("search", "answer", "eval", "score"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    model_spec = _model_spec(args.model_id, secrets_path=args.secrets)
    root_name = args.root_name or f"matrix_raw_downstream_{model_spec['id']}"
    target_root = WORKSPACES_DIR / root_name
    target_add_dir = target_root / "raw" / "add"
    target_search_dir = target_root / "raw" / "llm"
    config_path = target_root / f"config.pipeline.{model_spec['id']}.yaml"
    payload = _pipeline_config_payload(
        target_root=target_root,
        root_name=root_name,
        pipeline_model=model_spec,
        search_llm_concurrency=max(1, int(args.search_llm_concurrency)),
        answer_concurrency=max(1, int(args.answer_concurrency)),
        eval_concurrency=max(1, int(args.eval_concurrency)),
    )

    if not args.foreground:
        _copy_required_workspace_files(source_dir=args.source_add_dir, target_dir=target_add_dir)
        _write_config(config_path, payload)
        print(f"[raw-downstream] prepared add workspace: {target_add_dir}")
        print(f"[raw-downstream] matrix_root: {target_root}")
        print(f"[raw-downstream] config: {config_path}")
        if args.prepare_only:
            print("[raw-downstream] prepare-only done")
            return
        pid, stdout_path, stderr_path = _launch_background(
            script_path=Path(__file__),
            model_id=args.model_id,
            root_name=root_name,
            source_add_dir=args.source_add_dir,
            secrets_path=args.secrets,
            search_llm_concurrency=max(1, int(args.search_llm_concurrency)),
            answer_concurrency=max(1, int(args.answer_concurrency)),
            eval_concurrency=max(1, int(args.eval_concurrency)),
            start_from_step=args.start_from_step,
            end_at_step=args.end_at_step,
        )
        print(f"[raw-downstream] launched pid={pid}")
        print(f"[raw-downstream] stdout={stdout_path}")
        print(f"[raw-downstream] stderr={stderr_path}")
        return

    _copy_required_workspace_files(source_dir=args.source_add_dir, target_dir=target_add_dir)
    _write_config(config_path, payload)
    link_add_database_to_search_run(
        target_add_dir,
        target_search_dir,
        extra={
            "workspace_name": "llm",
            "add_model_id": "raw",
            "add_backend": "raw",
            "add_model": model_spec["model"],
            "add_repeat_index": 1,
            "search_backend": "llm",
            "downstream_model_id": model_spec["id"],
            "downstream_model": model_spec["model"],
        },
    )
    if args.print_config:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    import asyncio

    load_runtime_env()
    asyncio.run(
        run_pipeline_from_config(
            payload,
            start_from_step=args.start_from_step,
            end_at_step=args.end_at_step,
            config_path=config_path,
            load_env=False,
            pipeline_dir=PIPELINE_DIR,
            version_dir=VERSION_DIR,
            tee_log=True,
        )
    )


if __name__ == "__main__":
    main()
