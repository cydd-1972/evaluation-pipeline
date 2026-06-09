from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


BALANCE_PATTERNS = ("402", "insufficient", "balance", "quota", "payment required", "credit")


def _write_log(path: Path, message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["cmd", "/c", "tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    return str(pid) in (result.stdout or "")


def _test_completed(log_path: Path, score_path: Path) -> bool:
    if score_path.exists():
        return True
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        return "[pipeline] all steps completed." in text
    return False


def _test_balance_failure(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    return any(pattern in text for pattern in BALANCE_PATTERNS)


def _set_minimax_key(secrets_path: Path, api_key: str) -> None:
    payload = yaml.safe_load(secrets_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("minimax"), dict):
        raise ValueError(f"invalid secrets yaml: {secrets_path}")
    payload["minimax"]["api_key"] = api_key
    secrets_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _resolve_current_run_log(log_pointer_path: Path) -> Path | None:
    if not log_pointer_path.exists():
        return None
    raw = log_pointer_path.read_text(encoding="utf-8", errors="ignore").strip()
    return Path(raw) if raw else None


def _launch_minimax(eval_dir: Path) -> tuple[subprocess.Popen[bytes], Path, Path]:
    logs_dir = eval_dir / "v5" / "workspaces" / "logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = logs_dir / f"relaunch_minimax_{stamp}.out.txt"
    stderr_path = logs_dir / f"relaunch_minimax_{stamp}.err.txt"
    stdout_handle = stdout_path.open("wb")
    stderr_handle = stderr_path.open("wb")
    proc = subprocess.Popen(
        [sys.executable, "v5/run.py", "--model-id", "minimax", "--from", "search"],
        cwd=str(eval_dir),
        stdout=stdout_handle,
        stderr=stderr_handle,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return proc, stdout_path, stderr_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch minimax process; switch to fallback key on balance failure.")
    parser.add_argument("--wait-process-id", type=int, required=True)
    parser.add_argument("--minimax-log-path", type=Path, required=True)
    parser.add_argument("--fallback-minimax-api-key", required=True)
    parser.add_argument("--secrets-path", type=Path, default=Path("evaluation_pipeline/configs/matrix_secrets.yaml"))
    parser.add_argument(
        "--minimax-score-path",
        type=Path,
        default=Path("evaluation_pipeline/v5/workspaces/allconv_v5_minimax/score_summary_answerhistory.json"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    eval_dir = repo_root
    logs_dir = Path(__file__).resolve().parent / "workspaces" / "logs"
    chain_log = logs_dir / f"watch_minimax_fallback_{datetime.now():%Y%m%d_%H%M%S}.log"
    run_log_pointer = logs_dir / "run_log_current.txt"
    secrets_path = args.secrets_path if args.secrets_path.is_absolute() else repo_root.parent / args.secrets_path
    score_path = args.minimax_score_path if args.minimax_score_path.is_absolute() else repo_root.parent / args.minimax_score_path

    current_pid = args.wait_process_id
    current_log = args.minimax_log_path
    fallback_used = False

    _write_log(chain_log, f"Watcher started. wait_pid={current_pid} minimax_log={current_log}")

    while True:
        while _process_exists(current_pid):
            time.sleep(max(3, int(args.poll_seconds)))

        _write_log(chain_log, f"Process exited. pid={current_pid}")
        if _test_completed(current_log, score_path):
            _write_log(chain_log, "Minimax completed successfully. Watcher exits.")
            return 0

        if (not fallback_used) and _test_balance_failure(current_log):
            _write_log(chain_log, "Detected balance/quota-style failure. Switching to fallback key and relaunching from search.")
            _set_minimax_key(secrets_path, args.fallback_minimax_api_key)
            relaunched_proc, stdout_path, stderr_path = _launch_minimax(eval_dir)
            _write_log(
                chain_log,
                f"Relaunched minimax with fallback key. pid={relaunched_proc.pid} stdout={stdout_path} stderr={stderr_path}",
            )
            time.sleep(3)
            current_pid = relaunched_proc.pid
            current_log = _resolve_current_run_log(run_log_pointer) or current_log
            _write_log(chain_log, f"Updated current minimax log to {current_log}")
            fallback_used = True
            continue

        _write_log(chain_log, "Minimax exited without completion and without eligible fallback. Watcher exits with failure.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
