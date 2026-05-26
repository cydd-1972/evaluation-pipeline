"""矩阵实验：3 模型 × add 3 次 × search(llm/rag) 各 1 次 = 27 runs。

并行策略（默认）：
  - 三模型 add 并行
  - 同 add_run 下 llm/rag search+answer 并行
  - eval+score 全局串行

日志与统计：
  - 详细日志 → workspaces/matrix/matrix_run_YYYYMMDD_HHMMSS.log（每次 session 新文件）
  - 当前日志路径 → workspaces/matrix/matrix_log_current.txt
  - 分环节耗时 → workspaces/matrix/matrix_timings.json
  - 汇总分数 → workspaces/matrix/matrix_final_scores.json

用法：
  python run_matrix.py --dry-run
  python run_matrix.py
  python run_matrix.py --only-add --model gemini --repeat 1
  python run_matrix.py --serial   # 旧版逐 run 串行
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from lib.env import load_runtime_env
from lib.matrix import (
    AddModelSpec,
    MatrixRunSpec,
    RAW_ADD_MODEL_ID,
    add_workspace_dir,
    add_db_workspace_name,
    apply_add_model_env,
    build_base_pipeline_config,
    config_for_add_run,
    config_for_raw_add_run,
    config_for_raw_search_run,
    config_for_search_run,
    link_add_database_to_search_run,
    load_matrix_bundle,
    matrix_root,
    parse_add_models,
    plan_all_matrix_runs,
    raw_add_dir,
    resolve_pipeline_llm_model,
    write_manifest,
)
from lib.matrix_lock import MatrixProcessLock
from lib.matrix_log import LogOpenMode, matrix_file_logging, resolve_session_log_path
from lib.matrix_runner import run_matrix_parallel
from lib.matrix_status import load_status, save_status
from lib.matrix_telemetry import PhaseTimer, TimingStore, rebuild_final_scores
from run_pipeline import run_pipeline_from_config

DEFAULT_MATRIX_CONFIG = PIPELINE_DIR / "configs" / "matrix.yaml"


def _model_by_id(models: list[AddModelSpec], model_id: str) -> AddModelSpec:
    for model in models:
        if model.id == model_id:
            return model
    raise KeyError(f"unknown add model id: {model_id}")


def _filter_runs(
    runs: list[MatrixRunSpec],
    *,
    only_add: bool,
    only_search: bool,
    model_id: str | None,
    search_backend: str | None,
    repeat: int | None,
) -> list[MatrixRunSpec]:
    filtered: list[MatrixRunSpec] = []
    for run in runs:
        if only_add and not run.is_add:
            continue
        if only_search and run.is_add:
            continue
        if model_id and run.add_model_id != model_id:
            continue
        if search_backend and run.search_backend != search_backend:
            continue
        if repeat is not None and run.add_repeat_index != repeat:
            continue
        filtered.append(run)
    return filtered


def _status_path(root: Path) -> Path:
    return root / "matrix_status.json"


def _log_path(root: Path, matrix_cfg: dict[str, Any]) -> tuple[Path, LogOpenMode]:
    mode = str(matrix_cfg.get("matrix_log_mode") or "session").strip().lower()
    if mode == "append":
        name = str(matrix_cfg.get("matrix_log_file") or "matrix_run.log")
        return root / name, "append"
    prefix = str(matrix_cfg.get("matrix_log_prefix") or "matrix_run")
    return resolve_session_log_path(root, prefix=prefix), "write"


async def _execute_run(
    run: MatrixRunSpec,
    *,
    base_config: dict[str, Any],
    models: list[AddModelSpec],
    root: Path,
    matrix_cfg: dict[str, Any],
    store: TimingStore | None = None,
    start_from_step: str | None = None,
    end_at_step: str | None = None,
) -> None:
    if run.is_add:
        if run.add_model_id != RAW_ADD_MODEL_ID:
            apply_add_model_env(_model_by_id(models, run.add_model_id))
    else:
        if run.add_model_id == RAW_ADD_MODEL_ID:
            apply_add_model_env(resolve_pipeline_llm_model(matrix_cfg, models))
        else:
            apply_add_model_env(_model_by_id(models, run.add_model_id))

    async def _run_pipeline(cfg: dict[str, Any], *, phase: str | None = None) -> None:
        step_start = start_from_step or phase or ("add" if run.is_add else "search")
        step_end = end_at_step or phase
        if store and phase:
            async with PhaseTimer(store, run, phase):
                await run_pipeline_from_config(
                    cfg, start_from_step=step_start, end_at_step=step_end, load_env=False
                )
        else:
            await run_pipeline_from_config(
                cfg, start_from_step=step_start, end_at_step=step_end, load_env=False
            )

    if run.is_add:
        if run.add_model_id == RAW_ADD_MODEL_ID:
            cfg = config_for_raw_add_run(base_config, add_dir=run.workspace_dir)
        else:
            cfg = config_for_add_run(
                base_config,
                model=_model_by_id(models, run.add_model_id),
                db_workspace_name=run.workspace_name,
                model_dir=run.workspace_dir,
                add_repeat_index=int(run.add_repeat_index or 1),
            )
        await _run_pipeline(cfg, phase="add")
        return

    if run.add_model_id == RAW_ADD_MODEL_ID:
        add_dir = raw_add_dir(root)
        pipeline_model = resolve_pipeline_llm_model(matrix_cfg, models)
        if not (add_dir / "workspace.json").exists():
            raise FileNotFoundError(
                f"raw add workspace missing: {add_dir / 'workspace.json'}. "
                "Run add first: python run_matrix.py --config configs/matrix_raw.yaml --only-add"
            )
        cfg = config_for_raw_search_run(
            base_config,
            pipeline_model=pipeline_model,
            search_backend=str(run.search_backend),
            search_dir=run.workspace_dir,
            parent_add_workspace=str(add_dir),
        )
        link_add_database_to_search_run(
            add_dir,
            run.workspace_dir,
            extra={
                "workspace_name": run.workspace_name,
                "add_model_id": RAW_ADD_MODEL_ID,
                "add_backend": "raw",
                "add_model": pipeline_model.model,
                "add_repeat_index": 1,
                "search_backend": run.search_backend,
            },
        )
    else:
        add_repeat = int(run.add_repeat_index or 1)
        add_dir = add_workspace_dir(root, run.add_model_id, add_repeat)
        model = _model_by_id(models, run.add_model_id)
        if not (add_dir / "workspace.json").exists():
            raise FileNotFoundError(
                f"add workspace missing for model '{run.add_model_id}' add_run{add_repeat:02d}: "
                f"{add_dir / 'workspace.json'}. "
                f"Run add first: python run_matrix.py --only-add --model {run.add_model_id} --repeat {add_repeat}"
            )

        db_workspace_name = add_db_workspace_name(matrix_cfg, run.add_model_id, add_repeat)
        cfg = config_for_search_run(
            base_config,
            model_id=run.add_model_id,
            add_model=model.model,
            search_backend=str(run.search_backend),
            add_repeat_index=add_repeat,
            db_workspace_name=db_workspace_name,
            search_dir=run.workspace_dir,
            parent_add_workspace=str(add_dir),
        )
        link_add_database_to_search_run(
            add_dir,
            run.workspace_dir,
            extra={
                "workspace_name": run.workspace_name,
                "add_model_id": run.add_model_id,
                "add_model": model.model,
                "add_repeat_index": add_repeat,
                "search_backend": run.search_backend,
            },
        )

    if store and not start_from_step and not end_at_step:
        for phase in ("search", "answer", "eval", "score"):
            await _run_pipeline(cfg, phase=phase)
    else:
        await _run_pipeline(cfg)


async def _run_serial(
    *,
    runs: list[MatrixRunSpec],
    all_runs: list[MatrixRunSpec],
    base_config: dict[str, Any],
    models: list[AddModelSpec],
    root: Path,
    matrix_cfg: dict[str, Any],
    status_path: Path,
    skip_completed: bool,
    continue_on_error: bool,
) -> None:
    status = load_status(status_path)
    store = TimingStore(root / "matrix_timings.json")
    answer_mode = str(base_config.get("answer_prompt_mode") or "history")

    ordered = sorted(
        runs,
        key=lambda r: (
            r.add_repeat_index or 0,
            r.add_model_id,
            0 if r.is_add else 1,
            r.search_backend or "",
        ),
    )

    for index, run in enumerate(ordered, start=1):
        if skip_completed and run.run_id in status.get("completed", {}):
            print(f"\n[matrix] ({index}/{len(ordered)}) SKIP completed: {run.run_id}", flush=True)
            continue

        print(f"\n[matrix] ({index}/{len(ordered)}) START {run.run_id}", flush=True)
        print(f"[matrix] workspace={run.workspace_dir}", flush=True)
        started = time.perf_counter()
        try:
            await _execute_run(
                run,
                base_config=base_config,
                models=models,
                root=root,
                matrix_cfg=matrix_cfg,
                store=store,
            )
            elapsed = time.perf_counter() - started
            status["completed"][run.run_id] = {
                "workspace_dir": str(run.workspace_dir),
                "elapsed_s": round(elapsed, 1),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            status["failed"].pop(run.run_id, None)
            save_status(status_path, status)
            print(f"[matrix] OK {run.run_id} ({elapsed:.1f}s)", flush=True)
            if not run.is_add:
                rebuild_final_scores(root=root, runs=all_runs, answer_mode=answer_mode)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            status["failed"][run.run_id] = {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "elapsed_s": round(elapsed, 1),
            }
            save_status(status_path, status)
            print(f"[matrix] FAIL {run.run_id}: {exc}", flush=True)
            if not continue_on_error:
                raise

    rebuild_final_scores(root=root, runs=all_runs, answer_mode=answer_mode)
    failed_count = len(status.get("failed", {}))
    if failed_count:
        print(f"\n[matrix] finished with {failed_count} failed run(s). Re-run to retry failed only.", flush=True)
    print(f"\n[matrix] timings={root / 'matrix_timings.json'}", flush=True)
    print(f"[matrix] final_scores={root / 'matrix_final_scores.json'}", flush=True)
    print(f"[matrix] done. status={status_path}", flush=True)


async def _run_matrix_body(
    *,
    runs: list[MatrixRunSpec],
    all_runs: list[MatrixRunSpec],
    base_config: dict[str, Any],
    models: list[AddModelSpec],
    root: Path,
    matrix_cfg: dict[str, Any],
    status_path: Path,
    skip_completed: bool,
    continue_on_error: bool,
    serial: bool,
    parallel_models: int,
    parallel_search: int,
) -> None:
    answer_mode = str(base_config.get("answer_prompt_mode") or "history")
    rebuild_final_scores(root=root, runs=all_runs, answer_mode=answer_mode)

    if serial:
        await _run_serial(
            runs=runs,
            all_runs=all_runs,
            base_config=base_config,
            models=models,
            root=root,
            matrix_cfg=matrix_cfg,
            status_path=status_path,
            skip_completed=skip_completed,
            continue_on_error=continue_on_error,
        )
    else:
        await run_matrix_parallel(
            runs=runs,
            all_runs=all_runs,
            base_config=base_config,
            models=models,
            root=root,
            matrix_cfg=matrix_cfg,
            status_path=status_path,
            skip_completed=skip_completed,
            continue_on_error=continue_on_error,
            parallel_models=parallel_models,
            parallel_search=parallel_search,
        )


def run_matrix(
    *,
    matrix_config_path: Path,
    dry_run: bool = False,
    only_add: bool = False,
    only_search: bool = False,
    model_id: str | None = None,
    search_backend: str | None = None,
    repeat: int | None = None,
    skip_completed: bool = True,
    continue_on_error: bool = True,
    serial: bool = False,
) -> None:
    load_runtime_env()
    matrix_cfg, secrets = load_matrix_bundle(matrix_config_path=matrix_config_path)
    models = parse_add_models(matrix_cfg, secrets)
    base_config = build_base_pipeline_config(matrix_cfg)
    root = matrix_root(matrix_cfg)
    all_runs = plan_all_matrix_runs(matrix_cfg, models)
    runs = _filter_runs(
        all_runs,
        only_add=only_add,
        only_search=only_search,
        model_id=model_id,
        search_backend=search_backend,
        repeat=repeat,
    )

    write_manifest(root / "manifest.json", matrix_cfg=matrix_cfg, models=models, runs=all_runs)
    status_path = _status_path(root)
    log_file, log_open_mode = _log_path(root, matrix_cfg)
    log_pointer = root / "matrix_log_current.txt"
    parallel_models = int(matrix_cfg.get("parallel_models") or len(models))
    parallel_search = int(matrix_cfg.get("parallel_search") or len(models) * len(matrix_cfg.get("search_backends") or []))

    print(f"[matrix] root={root}")
    print(f"[matrix] planned runs this invocation: {len(runs)} / total {len(all_runs)}")
    print(f"[matrix] mode={'serial' if serial else 'parallel'} parallel_models={parallel_models} parallel_search={parallel_search}")
    print(f"[matrix] log_file={log_file}")
    print(f"[matrix] log_pointer={log_pointer}")
    print(f"[matrix] timings={root / 'matrix_timings.json'} final_scores={root / 'matrix_final_scores.json'}")

    if dry_run:
        for index, run in enumerate(runs, start=1):
            print(f"  {index:02d}. {run.run_id} -> {run.workspace_dir} start={run.start_from_step}")
        return

    lock_path = root / "matrix_run.lock"
    with MatrixProcessLock(lock_path):
        with matrix_file_logging(log_file, open_mode=log_open_mode):
            asyncio.run(
                _run_matrix_body(
                    runs=runs,
                    all_runs=all_runs,
                    base_config=base_config,
                    models=models,
                    root=root,
                    matrix_cfg=matrix_cfg,
                    status_path=status_path,
                    skip_completed=skip_completed,
                    continue_on_error=continue_on_error,
                    serial=serial,
                    parallel_models=parallel_models,
                    parallel_search=parallel_search,
                )
            )

    print(f"[matrix] session log: {log_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoCoMo matrix (3 models × 3 add × 2 search)")
    parser.add_argument("--config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不执行")
    parser.add_argument("--only-add", action="store_true", help="只跑 add")
    parser.add_argument("--only-search", action="store_true", help="只跑 search 子流水线")
    parser.add_argument("--model", choices=["gemini", "minimax", "deepseek", "raw"], default=None)
    parser.add_argument("--search-backend", choices=["llm", "rag"], default=None)
    parser.add_argument("--repeat", type=int, choices=[1, 2, 3], default=None, help="add_run 编号 1..3")
    parser.add_argument("--no-skip-completed", action="store_true", help="不跳过已完成的 run")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="单次 run 失败时立即退出（默认记录失败并继续后续 run）",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="逐 run 串行（默认并行编排：add 并行、search+answer 并行、eval 串行）",
    )
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else PIPELINE_DIR / args.config
    run_matrix(
            matrix_config_path=config_path,
            dry_run=args.dry_run,
            only_add=args.only_add,
            only_search=args.only_search,
            model_id=args.model,
            search_backend=args.search_backend,
            repeat=args.repeat,
            skip_completed=not args.no_skip_completed,
            continue_on_error=not args.stop_on_error,
            serial=args.serial,
        )


if __name__ == "__main__":
    main()
