from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = VERSION_DIR.parent
LOGS_DIR = VERSION_DIR / "workspaces" / "logs"
HELPER = VERSION_DIR / "run_raw_downstream_llm.py"


def _spawn(
    *,
    model_id: str,
    root_name: str,
    source_add_dir: Path,
    secrets: Path,
    search_llm_concurrency: int,
    answer_concurrency: int,
    eval_concurrency: int,
    start_from_step: str,
    end_at_step: str | None,
    stamp: str,
) -> tuple[subprocess.Popen[bytes], Path, Path]:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOGS_DIR / f"launch_raw_parallel_{model_id}_{start_from_step}_{stamp}.out.txt"
    stderr_path = LOGS_DIR / f"launch_raw_parallel_{model_id}_{start_from_step}_{stamp}.err.txt"
    command = [
        sys.executable,
        str(HELPER),
        "--model-id",
        model_id,
        "--root-name",
        root_name,
        "--source-add-dir",
        str(source_add_dir),
        "--secrets",
        str(secrets),
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
    return proc, stdout_path, stderr_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw-add downstream experiments with parallel search+answer and serial eval.")
    parser.add_argument("--source-add-dir", type=Path, default=PIPELINE_DIR / "workspaces" / "matrix" / "raw" / "add")
    parser.add_argument("--secrets", type=Path, default=PIPELINE_DIR / "configs" / "matrix_secrets.yaml")
    parser.add_argument("--minimax-root", default="matrix_raw_downstream_minimax_parallel")
    parser.add_argument("--qwen-root", default="matrix_raw_downstream_qwen3_parallel")
    parser.add_argument("--search-llm-concurrency", type=int, default=4)
    parser.add_argument("--answer-concurrency", type=int, default=1)
    parser.add_argument("--eval-concurrency", type=int, default=4)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    first_wave = []
    for model_id, root_name in (("minimax", args.minimax_root), ("qwen3", args.qwen_root)):
        proc, stdout_path, stderr_path = _spawn(
            model_id=model_id,
            root_name=root_name,
            source_add_dir=args.source_add_dir,
            secrets=args.secrets,
            search_llm_concurrency=max(1, int(args.search_llm_concurrency)),
            answer_concurrency=max(1, int(args.answer_concurrency)),
            eval_concurrency=max(1, int(args.eval_concurrency)),
            start_from_step="search",
            end_at_step="answer",
            stamp=stamp,
        )
        first_wave.append((model_id, root_name, proc, stdout_path, stderr_path))
        print(f"[raw-parallel] launched {model_id} search+answer pid={proc.pid}")
        print(f"[raw-parallel] stdout={stdout_path}")
        print(f"[raw-parallel] stderr={stderr_path}")

    failed = False
    for model_id, root_name, proc, stdout_path, stderr_path in first_wave:
        return_code = proc.wait()
        print(f"[raw-parallel] {model_id} search+answer exit={return_code}")
        if return_code != 0:
            failed = True

    if failed:
        print("[raw-parallel] stop before eval because at least one search+answer run failed")
        return

    for model_id, root_name in (("minimax", args.minimax_root), ("qwen3", args.qwen_root)):
        proc, stdout_path, stderr_path = _spawn(
            model_id=model_id,
            root_name=root_name,
            source_add_dir=args.source_add_dir,
            secrets=args.secrets,
            search_llm_concurrency=max(1, int(args.search_llm_concurrency)),
            answer_concurrency=max(1, int(args.answer_concurrency)),
            eval_concurrency=max(1, int(args.eval_concurrency)),
            start_from_step="eval",
            end_at_step=None,
            stamp=stamp,
        )
        print(f"[raw-parallel] launched {model_id} eval+score pid={proc.pid}")
        print(f"[raw-parallel] stdout={stdout_path}")
        print(f"[raw-parallel] stderr={stderr_path}")
        return_code = proc.wait()
        print(f"[raw-parallel] {model_id} eval+score exit={return_code}")
        if return_code != 0:
            print("[raw-parallel] stop chain because eval+score failed")
            return
        time.sleep(1)

    print("[raw-parallel] all done")


if __name__ == "__main__":
    main()
