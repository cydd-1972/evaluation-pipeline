"""矩阵并行编排：三模型 add 并行；llm/rag search+answer 并行；eval+score 全局串行。"""
from __future__ import annotations

import asyncio
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Awaitable

from lib.matrix import (
    AddModelSpec,
    MatrixRunSpec,
    RAW_ADD_MODEL_ID,
    add_workspace_dir,
    add_db_workspace_name,
    apply_add_model_env,
    config_for_add_run,
    config_for_raw_add_run,
    config_for_raw_search_run,
    config_for_search_run,
    link_add_database_to_search_run,
    raw_add_dir,
    resolve_pipeline_llm_model,
)
from lib.matrix_status import load_status, mark_run_completed, mark_run_failed, save_status
from lib.matrix_telemetry import PhaseTimer, TimingStore, rebuild_final_scores
from run_pipeline import run_pipeline_from_config


def model_by_id(models: list[AddModelSpec], model_id: str) -> AddModelSpec:
    for model in models:
        if model.id == model_id:
            return model
    raise KeyError(f"unknown add model id: {model_id}")


async def execute_phase(
    run: MatrixRunSpec,
    *,
    phase: str,
    base_config: dict[str, Any],
    models: list[AddModelSpec],
    root: Path,
    matrix_cfg: dict[str, Any],
) -> None:
    if run.is_add:
        if run.add_model_id != RAW_ADD_MODEL_ID:
            apply_add_model_env(model_by_id(models, run.add_model_id))
    else:
        if run.add_model_id == RAW_ADD_MODEL_ID:
            apply_add_model_env(resolve_pipeline_llm_model(matrix_cfg, models))
        else:
            apply_add_model_env(model_by_id(models, run.add_model_id))

    if run.is_add:
        if phase != "add":
            raise ValueError(f"add run cannot execute phase={phase}")
        if run.add_model_id == RAW_ADD_MODEL_ID:
            cfg = config_for_raw_add_run(base_config, add_dir=run.workspace_dir)
        else:
            model = model_by_id(models, run.add_model_id)
            cfg = config_for_add_run(
                base_config,
                model=model,
                db_workspace_name=run.workspace_name,
                model_dir=run.workspace_dir,
                add_repeat_index=int(run.add_repeat_index or 1),
            )
        await run_pipeline_from_config(cfg, start_from_step="add", end_at_step="add", load_env=False)
        return

    if run.add_model_id == RAW_ADD_MODEL_ID:
        add_dir = raw_add_dir(root)
        if not (add_dir / "workspace.json").exists():
            raise FileNotFoundError(f"raw add workspace missing: {add_dir / 'workspace.json'}")
        pipeline_model = resolve_pipeline_llm_model(matrix_cfg, models)
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
        await run_pipeline_from_config(
            cfg,
            start_from_step=phase,
            end_at_step=phase,
            load_env=False,
        )
        return

    add_repeat = int(run.add_repeat_index or 1)
    add_dir = add_workspace_dir(root, run.add_model_id, add_repeat)
    if not (add_dir / "workspace.json").exists():
        raise FileNotFoundError(
            f"add workspace missing: {add_dir / 'workspace.json'} "
            f"(run add first: {run.add_model_id} add_run{add_repeat:02d})"
        )

    model = model_by_id(models, run.add_model_id)
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
    await run_pipeline_from_config(
        cfg,
        start_from_step=phase,
        end_at_step=phase,
        load_env=False,
    )


async def _run_phases_serial(
    run: MatrixRunSpec,
    phases: tuple[str, ...],
    *,
    base_config: dict[str, Any],
    models: list[AddModelSpec],
    root: Path,
    matrix_cfg: dict[str, Any],
    store: TimingStore,
    skip_if_ok: bool,
) -> None:
    for phase in phases:
        if skip_if_ok and store.phases_done(run.run_id, (phase,)):
            print(f"[matrix] SKIP {run.run_id} phase={phase} (already ok)", flush=True)
            continue
        async with PhaseTimer(store, run, phase):
            await execute_phase(
                run,
                phase=phase,
                base_config=base_config,
                models=models,
                root=root,
                matrix_cfg=matrix_cfg,
            )


async def _parallel_map(
    items: list[MatrixRunSpec],
    worker: Callable[[MatrixRunSpec], Awaitable[None]],
    *,
    concurrency: int,
    label: str,
) -> list[BaseException | None]:
    if not items:
        return []
    sem = asyncio.Semaphore(max(1, concurrency))
    print(f"[matrix] parallel {label}: {len(items)} task(s), concurrency={concurrency}", flush=True)

    async def _wrap(run: MatrixRunSpec) -> None:
        async with sem:
            await worker(run)

    results = await asyncio.gather(*[_wrap(r) for r in items], return_exceptions=True)
    return list(results)


async def run_matrix_parallel(
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
    parallel_models: int,
    parallel_search: int,
) -> None:
    status = load_status(status_path)
    store = TimingStore(root / "matrix_timings.json")

    answer_mode = str(base_config.get("answer_prompt_mode") or "history")
    add_repeats = sorted({int(r.add_repeat_index or 1) for r in runs})

    async def _mark_ok(run: MatrixRunSpec, elapsed: float) -> None:
        await mark_run_completed(
            status_path,
            status,
            run_id=run.run_id,
            workspace_dir=run.workspace_dir,
            elapsed_s=elapsed,
        )

    async def _mark_fail(run: MatrixRunSpec, exc: BaseException, elapsed: float) -> None:
        await mark_run_failed(
            status_path,
            status,
            run_id=run.run_id,
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
            elapsed_s=elapsed,
        )

    def _skip_run(run: MatrixRunSpec) -> bool:
        return skip_completed and run.run_id in status.get("completed", {})

    def _clear_failed(run_id: str) -> None:
        if run_id in status.get("failed", {}):
            status["failed"].pop(run_id, None)
            save_status(status_path, status)
            print(f"[matrix] retry {run_id} (cleared previous failure)", flush=True)

    for add_repeat in add_repeats:
        wave_add = [r for r in runs if r.is_add and int(r.add_repeat_index or 0) == add_repeat]
        wave_add = [r for r in wave_add if not _skip_run(r)]

        async def _add_worker(run: MatrixRunSpec) -> None:
            if store.has_ok(run.run_id, "add"):
                print(f"[matrix] SKIP {run.run_id} add (timing ok)", flush=True)
                if run.run_id not in status.get("completed", {}):
                    await _mark_ok(run, 0.0)
                return
            _clear_failed(run.run_id)
            started = time.perf_counter()
            try:
                async with PhaseTimer(store, run, "add"):
                    await execute_phase(
                        run,
                        phase="add",
                        base_config=base_config,
                        models=models,
                        root=root,
                        matrix_cfg=matrix_cfg,
                    )
                await _mark_ok(run, time.perf_counter() - started)
            except Exception as exc:
                await _mark_fail(run, exc, time.perf_counter() - started)
                if not continue_on_error:
                    raise

        failures = await _parallel_map(wave_add, _add_worker, concurrency=parallel_models, label=f"add_run{add_repeat:02d}")
        if any(isinstance(x, Exception) for x in failures) and not continue_on_error:
            return

        wave_search = [
            r
            for r in runs
            if (not r.is_add) and int(r.add_repeat_index or 0) == add_repeat and not _skip_run(r)
        ]

        async def _search_answer_worker(run: MatrixRunSpec) -> None:
            if store.phases_done(run.run_id, ("search", "answer")):
                print(f"[matrix] SKIP {run.run_id} search+answer (timing ok)", flush=True)
                return
            _clear_failed(run.run_id)
            try:
                await _run_phases_serial(
                    run,
                    ("search", "answer"),
                    base_config=base_config,
                    models=models,
                    root=root,
                    matrix_cfg=matrix_cfg,
                    store=store,
                    skip_if_ok=True,
                )
            except Exception as exc:
                await _mark_fail(run, exc, 0.0)
                if not continue_on_error:
                    raise

        failures = await _parallel_map(
            wave_search,
            _search_answer_worker,
            concurrency=parallel_search,
            label=f"search+answer add_run{add_repeat:02d}",
        )
        if any(isinstance(x, Exception) for x in failures) and not continue_on_error:
            return

        for run in wave_search:
            if _skip_run(run):
                continue
            if run.run_id in status.get("failed", {}):
                continue
            if store.phases_done(run.run_id, ("eval", "score")):
                if run.run_id not in status.get("completed", {}):
                    await _mark_ok(run, 0.0)
                continue
            started = time.perf_counter()
            try:
                await _run_phases_serial(
                    run,
                    ("eval", "score"),
                    base_config=base_config,
                    models=models,
                    root=root,
                    matrix_cfg=matrix_cfg,
                    store=store,
                    skip_if_ok=True,
                )
                await _mark_ok(run, time.perf_counter() - started)
                rebuild_final_scores(root=root, runs=all_runs, answer_mode=answer_mode)
            except Exception as exc:
                await _mark_fail(run, exc, time.perf_counter() - started)
                if not continue_on_error:
                    return

    rebuild_final_scores(root=root, runs=all_runs, answer_mode=answer_mode)
    failed = len(status.get("failed", {}))
    print(f"[matrix] timings={root / 'matrix_timings.json'}", flush=True)
    print(f"[matrix] final_scores={root / 'matrix_final_scores.json'}", flush=True)
    if failed:
        print(f"[matrix] finished with {failed} failed run(s)", flush=True)
    print(f"[matrix] done. status={status_path}", flush=True)
