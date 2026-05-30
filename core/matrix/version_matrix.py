"""各版本 run.py --matrix 入口。"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from core.infra.env import load_runtime_env
from core.matrix.matrix import (
    AddModelSpec,
    MatrixRunSpec,
    RAW_ADD_MODEL_ID,
    add_workspace_dir,
    add_db_workspace_name,
    apply_add_model_env,
    apply_run_model_env,
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
from core.matrix.matrix_lock import MatrixProcessLock
from core.matrix.matrix_log import LogOpenMode, matrix_file_logging, resolve_session_log_path
from core.matrix.matrix_runner import run_matrix_parallel
from core.matrix.matrix_status import load_status, save_status
from core.matrix.matrix_telemetry import PhaseTimer, TimingStore, rebuild_final_scores
from core.pipeline.runner import run_pipeline_from_config
from core.paths import EVAL_PIPELINE_ROOT


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
    store: TimingStore | None,
    pipeline_dir: Path,
    version_dir: Path,
    start_from_step: str | None = None,
    end_at_step: str | None = None,
) -> None:
    if run.is_add:
        if run.add_model_id != RAW_ADD_MODEL_ID:
            apply_add_model_env(_model_by_id(models, run.add_model_id))
    else:
        apply_run_model_env(matrix_cfg, models, run)

    async def _run_pipeline(cfg: dict[str, Any], *, phase: str | None = None) -> None:
        step_start = start_from_step or phase or ("add" if run.is_add else "search")
        step_end = end_at_step or phase
        kwargs = dict(
            cfg,
            start_from_step=step_start,
            end_at_step=step_end,
            load_env=False,
            pipeline_dir=pipeline_dir,
            version_dir=version_dir,
            tee_log=False,
        )
        if store and phase:
            async with PhaseTimer(store, run, phase):
                await run_pipeline_from_config(**kwargs)
        else:
            await run_pipeline_from_config(**kwargs)

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
            raise FileNotFoundError(f"raw add workspace missing: {add_dir / 'workspace.json'}")
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
                f"{add_dir / 'workspace.json'}"
            )
        db_workspace_name = add_db_workspace_name(matrix_cfg, run.add_model_id, add_repeat)
        pipeline_llm = (
            resolve_pipeline_llm_model(matrix_cfg, models)
            if str(run.search_backend or "").strip().lower() == "llm"
            else None
        )
        cfg = config_for_search_run(
            base_config,
            model_id=run.add_model_id,
            add_model=model.model,
            search_backend=str(run.search_backend),
            add_repeat_index=add_repeat,
            db_workspace_name=db_workspace_name,
            search_dir=run.workspace_dir,
            parent_add_workspace=str(add_dir),
            pipeline_llm=pipeline_llm,
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
    pipeline_dir: Path,
    version_dir: Path,
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
        started = time.perf_counter()
        try:
            await _execute_run(
                run,
                base_config=base_config,
                models=models,
                root=root,
                matrix_cfg=matrix_cfg,
                store=store,
                pipeline_dir=pipeline_dir,
                version_dir=version_dir,
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
    pipeline_dir: Path,
    version_dir: Path,
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
            pipeline_dir=pipeline_dir,
            version_dir=version_dir,
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
            pipeline_dir=pipeline_dir,
            version_dir=version_dir,
        )


async def run_version_matrix(
    *,
    version_dir: Path,
    pipeline_dir: Path,
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
    no_lock: bool = False,
) -> None:
    load_runtime_env()
    secrets_path = pipeline_dir / "configs" / "matrix_secrets.yaml"
    matrix_cfg, secrets = load_matrix_bundle(
        matrix_config_path=matrix_config_path,
        secrets_path=secrets_path,
    )
    models = parse_add_models(matrix_cfg, secrets)
    base_config = build_base_pipeline_config(matrix_cfg)
    root = matrix_root(matrix_cfg, base_dir=version_dir)
    all_runs = plan_all_matrix_runs(matrix_cfg, models, base_dir=version_dir)
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
    parallel_models = int(matrix_cfg.get("parallel_models") or len(models))
    parallel_search = int(
        matrix_cfg.get("parallel_search") or len(models) * len(matrix_cfg.get("search_backends") or [])
    )

    print(f"[matrix] version={version_dir.name} root={root}")
    print(f"[matrix] planned runs: {len(runs)} / {len(all_runs)}")

    if dry_run:
        for index, run in enumerate(runs, start=1):
            print(f"  {index:02d}. {run.run_id} -> {run.workspace_dir}")
        return

    lock_path = root / "matrix_run.lock"

    def _run() -> None:
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
                    pipeline_dir=pipeline_dir,
                    version_dir=version_dir,
                )
            )

    if no_lock:
        _run()
    else:
        with MatrixProcessLock(lock_path):
            _run()
