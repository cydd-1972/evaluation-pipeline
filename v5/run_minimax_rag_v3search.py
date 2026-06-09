from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


VERSION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = VERSION_DIR.parent
WORKSPACES_DIR = VERSION_DIR / "workspaces"
LOGS_DIR = WORKSPACES_DIR / "logs"
DEFAULT_CONFIG = VERSION_DIR / "config.minimax_rag_v3search.yaml"
DEFAULT_SOURCE_WORKSPACE = WORKSPACES_DIR / "allconv_v5_minimax"
DEFAULT_TARGET_WORKSPACE = WORKSPACES_DIR / "allconv_v5_minimax_ragv3search"


def _resolve_path(raw: Path, *, base_dir: Path) -> Path:
    if raw.is_absolute():
        return raw
    if raw.exists():
        return raw.resolve()
    candidate = base_dir / raw
    if candidate.exists():
        return candidate.resolve()
    return candidate


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

    reuse_manifest = {
        "source_workspace": str(source_dir),
        "reused_files": {
            "workspace_json": str(workspace_json),
            "add_snapshot": str(add_snapshot),
        },
        "prepared_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target_dir / "reused_add_manifest.json").write_text(
        json.dumps(reuse_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _launch_pipeline(*, config_path: Path, model_id: str, background: bool) -> tuple[int | None, Path | None, Path, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = LOGS_DIR / f"launch_minimax_rag_v3search_{stamp}.out.txt"
    stderr_path = LOGS_DIR / f"launch_minimax_rag_v3search_{stamp}.err.txt"
    command = [
        sys.executable,
        "v5/run.py",
        "--config",
        str(config_path),
        "--model-id",
        model_id,
        "--from",
        "search",
        "--reset-outputs-from-step",
        "search",
    ]
    if background:
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

    completed = subprocess.run(command, cwd=str(PIPELINE_DIR), check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return None, None, stdout_path, stderr_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reuse completed v5 minimax add workspace, but rerun downstream with v3-style global RAG search.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-id", default="minimax", choices=("minimax",))
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--target-workspace", type=Path, default=DEFAULT_TARGET_WORKSPACE)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    args = parser.parse_args()

    source_workspace = _resolve_path(args.source_workspace, base_dir=PIPELINE_DIR)
    target_workspace = _resolve_path(args.target_workspace, base_dir=PIPELINE_DIR)
    config_path = _resolve_path(args.config, base_dir=PIPELINE_DIR)

    _copy_required_workspace_files(source_dir=source_workspace, target_dir=target_workspace)
    print(f"[v5-rag] prepared target workspace: {target_workspace}")
    print(f"[v5-rag] source add workspace: {source_workspace}")

    if args.prepare_only:
        print("[v5-rag] prepare-only done")
        return

    pid, run_log_path, stdout_path, stderr_path = _launch_pipeline(
        config_path=config_path,
        model_id=args.model_id,
        background=not args.foreground,
    )
    if pid is None:
        print("[v5-rag] foreground run completed")
        return

    print(f"[v5-rag] launched pid={pid}")
    if run_log_path is not None:
        print(f"[v5-rag] run_log={run_log_path}")
    print(f"[v5-rag] stdout={stdout_path}")
    print(f"[v5-rag] stderr={stderr_path}")


if __name__ == "__main__":
    main()
