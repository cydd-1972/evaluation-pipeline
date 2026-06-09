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
WORKSPACES_DIR = VERSION_DIR / "workspaces"
LOGS_DIR = WORKSPACES_DIR / "logs"
DEFAULT_CONFIG = VERSION_DIR / "config.minimaxadd_qwen3_searchanswer.yaml"
DEFAULT_SOURCE_WORKSPACE = WORKSPACES_DIR / "allconv_v5_minimax"
DEFAULT_TARGET_WORKSPACE = WORKSPACES_DIR / "allconv_v5_minimaxadd_qwen3_searchanswer"

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from core.infra.env import evaluator_dashscope_keys, load_runtime_env
from core.infra.flat_export import flattened_eval_output_path
from core.pipeline.runner import run_pipeline_from_config


def _resolve_path(raw: Path, *, base_dir: Path) -> Path:
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    candidate = base_dir / raw
    if candidate.exists():
        return candidate.resolve()
    return candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def _copy_required_workspace_files(*, source_dir: Path, target_dir: Path) -> None:
    workspace_json = source_dir / "workspace.json"
    add_snapshot = source_dir / "add_snapshot.json"
    if not workspace_json.exists():
        raise FileNotFoundError(f"missing source workspace metadata: {workspace_json}")
    if not add_snapshot.exists():
        raise FileNotFoundError(f"missing source add snapshot: {add_snapshot}")

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(workspace_json, target_dir / "workspace.json")
    shutil.copy2(add_snapshot, target_dir / "add_snapshot.json")
    manifest = {
        "source_workspace": str(source_dir),
        "reused_files": {
            "workspace_json": str(workspace_json),
            "add_snapshot": str(add_snapshot),
        },
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target_dir / "reused_add_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _answer_mode(config: dict[str, Any]) -> str:
    return str(config.get("answer_prompt_mode") or "history")


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


def _qwen_client_config() -> dict[str, str]:
    load_runtime_env()
    keys = evaluator_dashscope_keys()
    api_key = keys[0] if keys else ""
    api_base = os.getenv("EVALUATOR_DASHSCOPE_API_BASE", "").strip() or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = os.getenv("EVALUATOR_DASHSCOPE_MODEL", "").strip() or "qwen3-14b"
    if not (api_key and api_base and model):
        raise ValueError("missing DashScope qwen3-14b config in .env")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
    }


def _materialize_config(config: dict[str, Any]) -> dict[str, Any]:
    qwen = _qwen_client_config()
    resolved = dict(config)
    resolved["search_llm_client"] = dict(qwen)
    resolved["answer_llm_client"] = dict(qwen)
    resolved["selected_model_id"] = "minimax"
    resolved["selected_model_name"] = "MiniMax-M2.7"
    resolved["fixed_downstream_model_id"] = "qwen3-14b"
    resolved["fixed_downstream_model_name"] = qwen["model"]
    return resolved


def _launch_background(
    *,
    config_path: Path,
    start_from_step: str,
    reset_outputs_from_step: str | None,
    no_copy_workspace: bool,
) -> tuple[int, Path | None, Path, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOGS_DIR / f"launch_minimaxadd_qwen3_searchanswer_{stamp}.out.txt"
    stderr_path = LOGS_DIR / f"launch_minimaxadd_qwen3_searchanswer_{stamp}.err.txt"
    command = [
        sys.executable,
        "v5/run_minimaxadd_qwen3_searchanswer.py",
        "--config",
        str(config_path),
        "--start-from-step",
        start_from_step,
        "--foreground",
    ]
    if reset_outputs_from_step:
        command.extend(["--reset-outputs-from-step", reset_outputs_from_step])
    if no_copy_workspace:
        command.append("--no-copy-workspace")
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(PIPELINE_DIR),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    time.sleep(3)
    run_log_pointer = LOGS_DIR / "run_log_current.txt"
    run_log_path = None
    if run_log_pointer.exists():
        raw = run_log_pointer.read_text(encoding="utf-8", errors="ignore").strip()
        run_log_path = Path(raw) if raw else None
    return proc.pid, run_log_path, stdout_path, stderr_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Reuse minimax add outputs, but run search+answer with qwen3-14b.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--target-workspace", type=Path, default=DEFAULT_TARGET_WORKSPACE)
    parser.add_argument("--start-from-step", default="search", choices=("search", "answer", "eval", "score"))
    parser.add_argument("--reset-outputs-from-step", default=None, choices=("search", "answer", "eval", "score"))
    parser.add_argument("--no-copy-workspace", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()

    source_workspace = _resolve_path(args.source_workspace, base_dir=PIPELINE_DIR)
    target_workspace = _resolve_path(args.target_workspace, base_dir=PIPELINE_DIR)
    config_path = _resolve_path(args.config, base_dir=PIPELINE_DIR)

    if not args.foreground:
        if not args.no_copy_workspace:
            _copy_required_workspace_files(source_dir=source_workspace, target_dir=target_workspace)
        print(f"[v5-qwen] prepared target workspace: {target_workspace}")
        print(f"[v5-qwen] source add workspace: {source_workspace}")
        if args.prepare_only:
            print("[v5-qwen] prepare-only done")
            return
        pid, run_log_path, stdout_path, stderr_path = _launch_background(
            config_path=config_path,
            start_from_step=args.start_from_step,
            reset_outputs_from_step=args.reset_outputs_from_step,
            no_copy_workspace=args.no_copy_workspace,
        )
        print(f"[v5-qwen] launched pid={pid}")
        if run_log_path is not None:
            print(f"[v5-qwen] run_log={run_log_path}")
        print(f"[v5-qwen] stdout={stdout_path}")
        print(f"[v5-qwen] stderr={stderr_path}")
        return

    load_runtime_env()
    config = _materialize_config(_load_yaml(config_path))
    if not args.no_copy_workspace:
        _copy_required_workspace_files(source_dir=source_workspace, target_dir=target_workspace)
    archive_dir_name = str(config.get("reset_archive_dir") or "archived_outputs").strip() or "archived_outputs"
    if args.reset_outputs_from_step:
        archived = _archive_step_outputs(
            workspace_dir=target_workspace,
            step=args.reset_outputs_from_step,
            answer_mode=_answer_mode(config),
            archive_dir_name=archive_dir_name,
        )
        if archived:
            print(
                f"[v5-qwen] archived {len(archived)} file(s) before rerun: "
                + ", ".join(str(path) for path in archived),
                flush=True,
            )
        else:
            print(f"[v5-qwen] no existing outputs to archive for step={args.reset_outputs_from_step}", flush=True)
    if args.print_config:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    import asyncio

    asyncio.run(
        run_pipeline_from_config(
            config,
            start_from_step=args.start_from_step,
            config_path=config_path,
            load_env=False,
            pipeline_dir=PIPELINE_DIR,
            version_dir=VERSION_DIR,
            tee_log=True,
        )
    )


if __name__ == "__main__":
    main()
