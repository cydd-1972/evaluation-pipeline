"""LoCoMo 评测流水线：五步 add → search → answer → eval → score。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

from core.infra.env import evaluator_settings, load_runtime_env
from core.infra.flat_export import (
    flattened_eval_output_path,
    write_flattened_eval_records,
    write_flattened_eval_records_from_file,
)
from core.infra.llm_client import PipelineLLM
from core.infra.scoring import load_and_summarize
from core.paths import EVAL_PIPELINE_ROOT
from core.pipeline.steps.answer import reanswer_dataset
from core.pipeline.steps.eval import evaluate_records
from core.run_log import LogOpenMode, matrix_file_logging, resolve_run_log_path
from core.search.search_llm import run_search_llm
from core.search.search_llm_global import run_search_llm_global
from core.search.search_hybrid_llm_global import run_search_hybrid_llm_global
from core.search.search_rag import run_search_rag
from core.search.search_rag_global import run_search_rag_global
from core.telemetry import PipelinePhaseTimer, RunTimingStore
from v1_mem0.add import run_add_mem0
from v2_raw.add import run_add_raw
from v3_global.add import run_add_global
from v4_global.add import run_add_global_v4
from v4_plus.add import run_add_global_v4_plus

PIPELINE_STEPS = ("add", "search", "answer", "eval", "score")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a YAML mapping")
    return payload


def _resolve_dataset_path(raw: str | Path, *, pipeline_dir: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else pipeline_dir / path


def _workspace_dir(
    config: dict[str, Any],
    *,
    pipeline_dir: Path,
    version_dir: Path | None,
) -> Path:
    base_raw = str(config.get("workspace_base_dir") or "workspaces")
    base_path = Path(base_raw)
    if base_path.is_absolute():
        base = base_path
    else:
        root = version_dir or pipeline_dir
        base = root / base_path
    name = str(config.get("workspace_name") or "smoke")
    return base / name


def _resolve_steps(start_from: str, *, end_at: str | None = None) -> tuple[str, ...]:
    normalized = str(start_from or "add").strip().lower()
    if normalized not in PIPELINE_STEPS:
        raise ValueError(f"unsupported start step: {start_from}")
    start_idx = PIPELINE_STEPS.index(normalized)
    if end_at is not None:
        end_normalized = str(end_at).strip().lower()
        if end_normalized not in PIPELINE_STEPS:
            raise ValueError(f"unsupported end step: {end_at}")
        end_idx = PIPELINE_STEPS.index(end_normalized)
        if end_idx < start_idx:
            raise ValueError(f"end_at {end_at} must not be before start_from {start_from}")
        return PIPELINE_STEPS[start_idx : end_idx + 1]
    return PIPELINE_STEPS[start_idx:]


def _answer_paths(workspace_dir: Path, answer_prompt_mode: str) -> tuple[Path, Path]:
    suffix = f"_answer{answer_prompt_mode}"
    return (
        workspace_dir / f"search_results{suffix}.json",
        workspace_dir / f"evaluation_metrics{suffix}.json",
    )


def _search_output_path(workspace_dir: Path) -> Path:
    return workspace_dir / "search_results.json"


def _read_database_url(workspace_dir: Path) -> str:
    workspace_json = workspace_dir / "workspace.json"
    if not workspace_json.exists():
        raise FileNotFoundError(f"missing workspace metadata: {workspace_json}")
    payload = json.loads(workspace_json.read_text(encoding="utf-8"))
    database_url = str(payload.get("database_url") or "").strip()
    if not database_url:
        raise ValueError(f"workspace.json missing database_url: {workspace_json}")
    return database_url


def _is_global_search(config: dict[str, Any]) -> bool:
    search_mode = str(config.get("search_mode") or "").strip().lower()
    add_backend = str(config.get("add_backend") or "mem0").strip().lower()
    return search_mode == "global" or add_backend in {"global", "global_v4", "global_v4_plus"}


def _search_llm_from_config(config: dict[str, Any]) -> PipelineLLM | None:
    spec = config.get("search_llm_client")
    if not isinstance(spec, dict):
        return None
    api_key = str(spec.get("api_key") or "").strip()
    api_base = str(spec.get("api_base") or "").strip()
    model = str(spec.get("model") or "").strip()
    if not (api_key and api_base and model):
        return None
    thinking_mode = str(spec.get("llm_thinking_mode") or "").strip().lower()
    if thinking_mode:
        os.environ["PIPELINE_LLM_THINKING_MODE"] = thinking_mode
    return PipelineLLM(api_key=api_key, api_base=api_base, model=model)


def _add_llm_from_config(config: dict[str, Any]) -> PipelineLLM | None:
    spec = config.get("add_llm_client")
    if not isinstance(spec, dict):
        return None
    api_key = str(spec.get("api_key") or "").strip()
    api_base = str(spec.get("api_base") or "").strip()
    model = str(spec.get("model") or "").strip()
    if not (api_key and api_base and model):
        return None
    thinking_mode = str(spec.get("llm_thinking_mode") or "").strip().lower()
    if thinking_mode:
        os.environ["PIPELINE_LLM_THINKING_MODE"] = thinking_mode
    else:
        os.environ.pop("PIPELINE_LLM_THINKING_MODE", None)
    return PipelineLLM(api_key=api_key, api_base=api_base, model=model)


def _answer_llm_from_config(config: dict[str, Any]) -> PipelineLLM | None:
    spec = config.get("answer_llm_client")
    if not isinstance(spec, dict):
        return None
    api_key = str(spec.get("api_key") or "").strip()
    api_base = str(spec.get("api_base") or "").strip()
    model = str(spec.get("model") or "").strip()
    if not (api_key and api_base and model):
        return None
    thinking_mode = str(spec.get("llm_thinking_mode") or "").strip().lower()
    if thinking_mode:
        os.environ["PIPELINE_LLM_THINKING_MODE"] = thinking_mode
    return PipelineLLM(api_key=api_key, api_base=api_base, model=model)


def _resolve_search_prompt_path(config: dict[str, Any], *, pipeline_dir: Path) -> Path | None:
    raw = config.get("search_llm_prompt")
    if raw is None or not str(raw).strip():
        return None
    path = Path(str(raw).strip())
    if path.is_absolute():
        return path
    return pipeline_dir / path


def _search_llm_require_non_empty(config: dict[str, Any]) -> bool:
    return bool(config.get("search_llm_require_non_empty"))


def _search_hybrid_settings(config: dict[str, Any]) -> tuple[int, int]:
    hybrid_cfg = config.get("search_hybrid")
    recall_k = config.get("search_hybrid_recall_k")
    rrf_k = config.get("search_hybrid_rrf_k")
    if isinstance(hybrid_cfg, dict):
        if recall_k is None:
            recall_k = hybrid_cfg.get("recall_k")
        if rrf_k is None:
            rrf_k = hybrid_cfg.get("rrf_k")
    return (
        int(recall_k or 80),
        int(rrf_k or 60),
    )


async def _run_add(
    config: dict[str, Any],
    workspace_dir: Path,
    *,
    pipeline_dir: Path,
) -> dict[str, Any]:
    db_workspace_name = str(config.get("workspace_db_name") or config["workspace_name"])
    add_kwargs = {
        "dataset_path": _resolve_dataset_path(str(config["dataset_path"]), pipeline_dir=pipeline_dir),
        "workspace_dir": workspace_dir,
        "database_url": os.getenv("EVAL_DATABASE_URL") or os.getenv("DATABASE_URL"),
        "workspace_name": db_workspace_name,
        "database_prefix": str(config.get("database_prefix") or "eval_pipeline"),
        "reset_database": bool(config.get("reset_database_on_add", True)),
        "max_conversations": config.get("max_conversations"),
        "max_sessions_per_conversation": config.get("max_sessions_per_conversation"),
        "progress_label": config.get("progress_label"),
    }
    backend = str(config.get("add_backend") or "mem0").strip().lower()
    add_llm = _add_llm_from_config(config)
    if add_llm is None and backend in {"mem0", "global", "global_v4", "global_v4_plus"}:
        add_llm = PipelineLLM()
    if backend == "raw":
        print("[pipeline] step=add (raw: session transcript → postgres + embedding, no LLM)")
        return await run_add_raw(**add_kwargs)
    if backend == "global_v4":
        batch = int(config.get("add_llm_concurrency") or 1)
        history_window = int(config.get("add_history_window") or 2)
        flush_per_session = bool(config.get("add_flush_per_session", True))
        backfill_embeddings = bool(config.get("backfill_embeddings_on_add", True))
        print(
            f"[pipeline] step=add (global v4 incremental, batch={batch}, "
            f"history_window={history_window}, model={add_llm.model})",
        )
        return await run_add_global_v4(
            **add_kwargs,
            llm=add_llm,
            add_llm_concurrency=batch,
            add_history_window=history_window,
            add_flush_per_session=flush_per_session,
            memory_prompt_path=config.get("memory_decision_prompt"),
            memory_prompt_max_items=config.get("memory_prompt_max_items"),
            backfill_embeddings=backfill_embeddings,
            enable_slot_aggregates=bool(config.get("enable_slot_aggregates", True)),
        )
    if backend == "global_v4_plus":
        batch = int(config.get("add_llm_concurrency") or 1)
        history_window = int(config.get("add_history_window") or 2)
        flush_per_session = bool(config.get("add_flush_per_session", True))
        backfill_embeddings = bool(config.get("backfill_embeddings_on_add", True))
        print(
            f"[pipeline] step=add (global v4 plus incremental, batch={batch}, "
            f"history_window={history_window}, model={add_llm.model})",
        )
        return await run_add_global_v4_plus(
            **add_kwargs,
            llm=add_llm,
            add_llm_concurrency=batch,
            add_history_window=history_window,
            add_flush_per_session=flush_per_session,
            memory_prompt_path=config.get("memory_decision_prompt"),
            memory_prompt_max_items=config.get("memory_prompt_max_items"),
            backfill_embeddings=backfill_embeddings,
            update_judge_prompt_path=config.get("memory_update_judge_prompt"),
            none_similarity_threshold=float(config.get("update_none_similarity_threshold", 0.92)),
            add_similarity_threshold=float(config.get("update_add_similarity_threshold", 0.55)),
            update_candidate_top_k=int(config.get("update_candidate_top_k", 3)),
        )
    if backend == "global":
        batch = int(config.get("add_llm_concurrency") or 1)
        history_window = int(config.get("add_history_window") or 2)
        flush_per_session = bool(config.get("add_flush_per_session", True))
        print(
            f"[pipeline] step=add (global session state machine, batch={batch}, "
            f"history_window={history_window}, model={add_llm.model})",
        )
        return await run_add_global(
            **add_kwargs,
            llm=add_llm,
            add_llm_concurrency=batch,
            add_history_window=history_window,
            add_flush_per_session=flush_per_session,
            memory_prompt_path=config.get("memory_decision_prompt"),
            memory_prompt_max_items=config.get("memory_prompt_max_items"),
        )
    batch = int(config.get("add_llm_concurrency") or 1)
    prompt_max = config.get("memory_prompt_max_items")
    print(f"[pipeline] step=add (mem0-style, batch={batch}, model={add_llm.model})")
    return await run_add_mem0(
        **add_kwargs,
        llm=add_llm,
        add_llm_concurrency=batch,
        memory_prompt_max_items=prompt_max,
    )


async def _run_search(
    config: dict[str, Any],
    workspace_dir: Path,
    *,
    pipeline_dir: Path,
) -> list[dict[str, Any]]:
    backend = str(config.get("search_backend") or "llm").strip().lower()
    is_global = _is_global_search(config)
    database_url = _read_database_url(workspace_dir)
    os.environ["EVAL_DATABASE_URL"] = database_url
    search_kwargs = {
        "dataset_path": _resolve_dataset_path(str(config["dataset_path"]), pipeline_dir=pipeline_dir),
        "workspace_dir": workspace_dir,
        "database_url": database_url,
        "max_conversations": config.get("max_conversations"),
        "max_questions_per_conversation": config.get("max_questions_per_conversation"),
        "top_k": int(config.get("search_top_k") or 30),
        "progress_label": config.get("progress_label"),
    }
    batch = int(config.get("search_llm_concurrency") or 1)
    search_llm = _search_llm_from_config(config)
    search_prompt_path = _resolve_search_prompt_path(config, pipeline_dir=pipeline_dir)
    require_non_empty = _search_llm_require_non_empty(config)
    hybrid_recall_k, hybrid_rrf_k = _search_hybrid_settings(config)

    if backend in {"llm", "hybrid_llm"}:
        if backend == "hybrid_llm":
            if not is_global:
                raise ValueError("search_backend hybrid_llm requires search_mode global or add_backend global/global_v4/global_v4_plus")
            frozen = f" frozen={search_llm.model}" if search_llm else ""
            print(
                f"[pipeline] step=search (global hybrid_llm: BM25+vector RRF→LLM, "
                f"recall_k={hybrid_recall_k}, batch={batch}{frozen})",
            )
            return await run_search_hybrid_llm_global(
                **search_kwargs,
                llm=search_llm,
                search_llm_concurrency=batch,
                search_prompt_path=search_prompt_path,
                search_llm_require_non_empty=require_non_empty,
                search_hybrid_recall_k=hybrid_recall_k,
                search_hybrid_rrf_k=hybrid_rrf_k,
                search_multihop_max_hops=int(config.get("search_multihop_max_hops") or 1),
                search_multihop_max_queries=int(config.get("search_multihop_max_queries") or 0),
            )
        if is_global:
            frozen = f" frozen={search_llm.model}" if search_llm else ""
            print(f"[pipeline] step=search (global llm, batch={batch}{frozen})")
            return await run_search_llm_global(
                **search_kwargs,
                llm=search_llm,
                search_llm_concurrency=batch,
                search_prompt_path=search_prompt_path,
                search_llm_require_non_empty=require_non_empty,
            )
        frozen = f" frozen={search_llm.model}" if search_llm else ""
        print(f"[pipeline] step=search (llm, batch={batch}{frozen})")
        return await run_search_llm(
            **search_kwargs,
            llm=search_llm,
            search_llm_concurrency=batch,
            search_prompt_path=search_prompt_path,
            search_llm_require_non_empty=require_non_empty,
        )
    if backend == "rag":
        if is_global:
            print("[pipeline] step=search (global rag, text-embedding-v4 + pgvector)")
            return await run_search_rag_global(**search_kwargs)
        print("[pipeline] step=search (rag, text-embedding-v4 + pgvector)")
        return await run_search_rag(**search_kwargs)
    raise ValueError(f"unsupported search_backend: {backend} (use llm, hybrid_llm, or rag)")


async def _run_answer(config: dict[str, Any], workspace_dir: Path) -> list[dict[str, Any]]:
    answer_mode = str(config.get("answer_prompt_mode") or "history")
    search_output = _search_output_path(workspace_dir)
    answer_output, _ = _answer_paths(workspace_dir, answer_mode)
    concurrency = int(config.get("concurrency") or config.get("answer_concurrency") or 2)
    answer_llm = _answer_llm_from_config(config)
    frozen = f" frozen={answer_llm.model}" if answer_llm else ""
    print(f"[pipeline] step=answer (prompt_mode={answer_mode}{frozen})")
    return await reanswer_dataset(
        input_path=search_output,
        output_path=answer_output,
        concurrency=concurrency,
        answer_prompt_mode=answer_mode,
        llm=answer_llm,
        progress_label=config.get("progress_label"),
    )


async def _run_eval(config: dict[str, Any], workspace_dir: Path) -> list[dict[str, Any]]:
    answer_mode = str(config.get("answer_prompt_mode") or "history")
    answer_output, eval_output = _answer_paths(workspace_dir, answer_mode)
    eval_cfg = config.get("eval") if isinstance(config.get("eval"), dict) else {}
    metrics = list(eval_cfg.get("metrics") or ["llm", "f1", "bleu"])
    evaluator_model, _, _ = evaluator_settings()
    eval_concurrency = int(
        config.get("eval_concurrency") or config.get("concurrency") or 6
    )
    print(f"[pipeline] step=eval metrics={metrics}")
    evaluated = await evaluate_records(
        input_path=answer_output,
        output_path=eval_output,
        concurrency=eval_concurrency,
        metrics=metrics,
        evaluator_model=evaluator_model,
        prefer_evaluator_slots=True,
        progress_label=config.get("progress_label"),
    )
    write_flattened_eval_records(records=evaluated, output_path=flattened_eval_output_path(eval_output))
    return evaluated


async def _run_score(config: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    answer_mode = str(config.get("answer_prompt_mode") or "history")
    _, eval_output = _answer_paths(workspace_dir, answer_mode)
    score_output = workspace_dir / f"score_summary_answer{answer_mode}.json"
    print("[pipeline] step=score")
    summary = load_and_summarize(eval_output)
    score_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_flattened_eval_records_from_file(input_path=eval_output)
    return summary


async def run_pipeline_from_config(
    config: dict[str, Any],
    *,
    start_from_step: str = "add",
    end_at_step: str | None = None,
    config_path: Path | None = None,
    load_env: bool = True,
    pipeline_dir: Path | None = None,
    version_dir: Path | None = None,
    tee_log: bool = True,
) -> None:
    root = pipeline_dir or EVAL_PIPELINE_ROOT
    if load_env:
        load_runtime_env()
    workspace_dir = _workspace_dir(config, pipeline_dir=root, version_dir=version_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    resolved_end = end_at_step if end_at_step is not None else config.get("end_at_step")
    snapshot: dict[str, Any] = {
        "start_from_step": start_from_step,
        "end_at_step": resolved_end,
        **config,
    }
    if config_path is not None:
        snapshot["config_path"] = str(config_path)
    (workspace_dir / "pipeline_config.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    steps = _resolve_steps(start_from_step, end_at=resolved_end)
    print(f"[pipeline] workspace={workspace_dir}")
    print(f"[pipeline] dataset={config.get('dataset_path')}")
    print(f"[pipeline] steps={' → '.join(steps)}")

    timing_store = RunTimingStore(workspace_dir)
    logs_dir = (version_dir or root) / "workspaces" / "logs"
    if version_dir is not None:
        logs_dir = version_dir / "workspaces" / "logs"
    log_ctx = None
    if tee_log:
        log_path = resolve_run_log_path(logs_dir)
        log_ctx = matrix_file_logging(log_path, open_mode="write")
        log_ctx.__enter__()
        print(f"[pipeline] log={log_path}", flush=True)

    try:
        for step_index, step in enumerate(steps, start=1):
            print(
                f"\n[pipeline] >>> step {step_index}/{len(steps)}: {step}",
                flush=True,
            )
            async with PipelinePhaseTimer(timing_store, step):
                if step == "add":
                    summary = await _run_add(config, workspace_dir, pipeline_dir=root)
                    print(f"[pipeline] add done: {summary.get('add_snapshot_path')}", flush=True)
                elif step == "search":
                    records = await _run_search(config, workspace_dir, pipeline_dir=root)
                    print(f"[pipeline] search done: {len(records)} records", flush=True)
                elif step == "answer":
                    records = await _run_answer(config, workspace_dir)
                    print(f"[pipeline] answer done: {len(records)} records", flush=True)
                elif step == "eval":
                    records = await _run_eval(config, workspace_dir)
                    _, eval_output = _answer_paths(
                        workspace_dir, str(config.get("answer_prompt_mode") or "history")
                    )
                    print(f"[pipeline] eval done: {len(records)} records → {eval_output}", flush=True)
                elif step == "score":
                    summary = await _run_score(config, workspace_dir)
                    print(
                        json.dumps(summary.get("overall", {}), ensure_ascii=False, indent=2),
                        flush=True,
                    )
        print("\n[pipeline] all steps completed.", flush=True)
    finally:
        if log_ctx is not None:
            log_ctx.__exit__(None, None, None)


def run_pipeline_cli(
    *,
    version_dir: Path,
    pipeline_dir: Path,
    default_config_name: str = "config.yaml",
    default_add_backend: str | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=f"LoCoMo pipeline ({version_dir.name})")
    parser.add_argument("--config", type=Path, default=version_dir / default_config_name)
    parser.add_argument("--start-from-step", "--from", dest="start_from_step", default="add", choices=PIPELINE_STEPS)
    parser.add_argument("--end-at-step", "--only", dest="end_at_step", default=None, choices=PIPELINE_STEPS)
    parser.add_argument("--matrix", action="store_true", help="run matrix experiment from config.matrix.yaml")
    parser.add_argument("--dry-run", action="store_true", help="matrix: print planned runs only")
    parser.add_argument("--no-tee-log", action="store_true")
    args = parser.parse_args()

    if str(pipeline_dir) not in sys.path:
        sys.path.insert(0, str(pipeline_dir))

    if args.matrix:
        from core.matrix.version_matrix import run_version_matrix

        matrix_config = version_dir / "config.matrix.yaml"
        asyncio.run(
            run_version_matrix(
                version_dir=version_dir,
                pipeline_dir=pipeline_dir,
                matrix_config_path=matrix_config,
                dry_run=args.dry_run,
            )
        )
        return

    config_path = args.config if args.config.is_absolute() else version_dir / args.config
    load_runtime_env()
    config = _load_config(config_path)
    if default_add_backend and not config.get("add_backend"):
        config["add_backend"] = default_add_backend
    asyncio.run(
        run_pipeline_from_config(
            config,
            start_from_step=args.start_from_step,
            end_at_step=args.end_at_step,
            config_path=config_path,
            pipeline_dir=pipeline_dir,
            version_dir=version_dir,
            tee_log=not args.no_tee_log,
        )
    )


async def run_pipeline(config_path: Path, *, start_from_step: str = "add") -> None:
    load_runtime_env()
    config = _load_config(config_path)
    await run_pipeline_from_config(
        config,
        start_from_step=start_from_step,
        config_path=config_path,
        pipeline_dir=EVAL_PIPELINE_ROOT,
    )
