"""LoCoMo 评测流水线入口。

五步顺序（可用 --start-from-step 从中间续跑）：
  add    → 对话写入 Postgres memories（mem0 风格，见 backends/add.py）
  search → LLM 从库中选记忆 id（见 backends/search_llm.py）
  answer → 基于检索结果生成答案（见 steps/answer.py）
  eval   → llm/f1/bleu 打分（见 steps/eval.py）
  score  → 按 category 汇总均值（见 lib/scoring.py）

产物目录：workspaces/<workspace_name>/
  workspace.json              # add 写入的 database_url，后续步骤依赖
  search_results.json         # search 输出
  search_results_answer22.json  # answer 输出（后缀随 answer_prompt_mode）
  evaluation_metrics_answer22.json
  score_summary_answer22.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# 本文件所在目录即 evaluation_pipeline 根；所有相对路径以此为基准
PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from backends.add import run_add_mem0
from backends.add_global import run_add_global
from backends.add_raw import run_add_raw
from backends.search_llm import run_search_llm
from backends.search_llm_global import run_search_llm_global
from backends.search_rag import run_search_rag
from backends.search_rag_global import run_search_rag_global
from lib.env import evaluator_settings, load_runtime_env
from lib.flat_export import (
    flattened_eval_output_path,
    write_flattened_eval_records,
    write_flattened_eval_records_from_file,
)
from lib.scoring import load_and_summarize
from steps.answer import reanswer_dataset
from steps.eval import evaluate_records

DEFAULT_CONFIG = PIPELINE_DIR / "config.yaml"
PIPELINE_STEPS = ("add", "search", "answer", "eval", "score")


def _load_config(path: Path) -> dict[str, Any]:
    """读取并解析 YAML 配置文件为 dict。"""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a YAML mapping")
    return payload


def _resolve_path(raw: str | Path) -> Path:
    """config 里的相对路径（如 datasets/...）相对 PIPELINE_DIR 解析。"""
    path = Path(raw)
    return path if path.is_absolute() else PIPELINE_DIR / path


def _resolve_steps(start_from: str, *, end_at: str | None = None) -> tuple[str, ...]:
    """例如 start_from=search → (search, answer, eval, score)；end_at=add 则只跑 add。"""
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


def _workspace_dir(config: dict[str, Any]) -> Path:
    """根据 workspace_base_dir + workspace_name 得到本次实验产物目录。"""
    base = _resolve_path(str(config.get("workspace_base_dir") or "workspaces"))
    name = str(config.get("workspace_name") or "locomo_refined_smoke")
    return base / name


def _answer_paths(workspace_dir: Path, answer_prompt_mode: str) -> tuple[Path, Path]:
    """返回 (answer 输出, eval 输出)；文件名带 _answer{mode} 后缀便于多 prompt 对比。"""
    suffix = f"_answer{answer_prompt_mode}"
    return (
        workspace_dir / f"search_results{suffix}.json",
        workspace_dir / f"evaluation_metrics{suffix}.json",
    )


def _search_output_path(workspace_dir: Path) -> Path:
    """search 步骤写出的 JSON 路径（answer 的输入）。"""
    return workspace_dir / "search_results.json"


def _read_database_url(workspace_dir: Path) -> str:
    """search 及之后步骤从 add 落盘的 workspace.json 读取独立库 URL。"""
    workspace_json = workspace_dir / "workspace.json"
    if not workspace_json.exists():
        raise FileNotFoundError(f"missing workspace metadata: {workspace_json}")
    payload = json.loads(workspace_json.read_text(encoding="utf-8"))
    database_url = str(payload.get("database_url") or "").strip()
    if not database_url:
        raise ValueError(f"workspace.json missing database_url: {workspace_json}")
    return database_url


async def _run_add(config: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    """调度 add backend：mem0（LLM 抽取）或 raw（session 原文入库）。"""
    db_workspace_name = str(config.get("workspace_db_name") or config["workspace_name"])
    add_kwargs = {
        "dataset_path": _resolve_path(str(config["dataset_path"])),
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
    if backend == "raw":
        print("[pipeline] step=add (raw: session transcript → postgres + embedding, no LLM)")
        return await run_add_raw(**add_kwargs)
    if backend == "global":
        batch = int(config.get("add_llm_concurrency") or 1)
        history_window = int(config.get("add_history_window") or 2)
        flush_per_session = bool(config.get("add_flush_per_session", True))
        print(
            f"[pipeline] step=add (global session state machine, batch={batch}, "
            f"history_window={history_window})",
        )
        return await run_add_global(
            **add_kwargs,
            add_llm_concurrency=batch,
            add_history_window=history_window,
            add_flush_per_session=flush_per_session,
        )
    batch = int(config.get("add_llm_concurrency") or 1)
    print(f"[pipeline] step=add (mem0-style, batch={batch})")
    return await run_add_mem0(**add_kwargs, add_llm_concurrency=batch)


def _is_global_search(config: dict[str, Any]) -> bool:
    search_mode = str(config.get("search_mode") or "").strip().lower()
    add_backend = str(config.get("add_backend") or "mem0").strip().lower()
    return search_mode == "global" or add_backend == "global"


async def _run_search(config: dict[str, Any], workspace_dir: Path) -> list[dict[str, Any]]:
    """调度 search_llm 或 search_rag；需 workspace.json 中的 database_url。"""
    backend = str(config.get("search_backend") or "llm").strip().lower()
    is_global = _is_global_search(config)
    database_url = _read_database_url(workspace_dir)
    os.environ["EVAL_DATABASE_URL"] = database_url
    search_kwargs = {
        "dataset_path": _resolve_path(str(config["dataset_path"])),
        "workspace_dir": workspace_dir,
        "database_url": database_url,
        "max_conversations": config.get("max_conversations"),
        "max_questions_per_conversation": config.get("max_questions_per_conversation"),
        "top_k": int(config.get("search_top_k") or 30),
        "progress_label": config.get("progress_label"),
    }
    if backend == "llm":
        batch = int(config.get("search_llm_concurrency") or 1)
        if is_global:
            print(f"[pipeline] step=search (global llm, batch={batch})")
            return await run_search_llm_global(**search_kwargs, search_llm_concurrency=batch)
        print(f"[pipeline] step=search (llm, batch={batch})")
        return await run_search_llm(**search_kwargs, search_llm_concurrency=batch)
    if backend == "rag":
        if is_global:
            print("[pipeline] step=search (global rag, text-embedding-v4 + pgvector)")
            return await run_search_rag_global(**search_kwargs)
        print("[pipeline] step=search (rag, text-embedding-v4 + pgvector)")
        return await run_search_rag(**search_kwargs)
    raise ValueError(f"unsupported search_backend: {backend} (use llm or rag)")


async def _run_answer(config: dict[str, Any], workspace_dir: Path) -> list[dict[str, Any]]:
    """调度 steps.answer.reanswer_dataset。"""
    answer_mode = str(config.get("answer_prompt_mode") or "history")
    search_output = _search_output_path(workspace_dir)
    answer_output, _ = _answer_paths(workspace_dir, answer_mode)
    print(f"[pipeline] step=answer (prompt_mode={answer_mode})")
    return await reanswer_dataset(
        input_path=search_output,
        output_path=answer_output,
        concurrency=int(config.get("concurrency") or 2),
        answer_prompt_mode=answer_mode,
        progress_label=config.get("progress_label"),
    )


async def _run_eval(config: dict[str, Any], workspace_dir: Path) -> list[dict[str, Any]]:
    """调度 steps.eval.evaluate_records 并写 flattened 导出。"""
    answer_mode = str(config.get("answer_prompt_mode") or "history")
    answer_output, eval_output = _answer_paths(workspace_dir, answer_mode)
    eval_cfg = config.get("eval") if isinstance(config.get("eval"), dict) else {}
    metrics = list(eval_cfg.get("metrics") or ["llm", "f1", "bleu"])
    evaluator_model, evaluator_base_url, evaluator_api_key = evaluator_settings()
    print(f"[pipeline] step=eval metrics={metrics}")
    evaluated = await evaluate_records(
        input_path=answer_output,
        output_path=eval_output,
        concurrency=int(config.get("concurrency") or 2),
        metrics=metrics,
        evaluator_model=evaluator_model,
        evaluator_base_url=evaluator_base_url,
        evaluator_api_key=evaluator_api_key,
        progress_label=config.get("progress_label"),
    )
    write_flattened_eval_records(records=evaluated, output_path=flattened_eval_output_path(eval_output))
    return evaluated


async def _run_score(config: dict[str, Any], workspace_dir: Path) -> dict[str, Any]:
    """读取 eval 结果，汇总 overall/by_category 并写入 score_summary_*.json。"""
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
) -> None:
    """使用内存中的 config dict 跑流水线（供 run_matrix 等批量脚本调用）。"""
    if load_env:
        load_runtime_env()
    workspace_dir = _workspace_dir(config)
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

    for step_index, step in enumerate(steps, start=1):
        started = time.perf_counter()
        print(
            f"\n[pipeline] >>> step {step_index}/{len(steps)}: {step}",
            flush=True,
        )
        if step == "add":
            summary = await _run_add(config, workspace_dir)
            print(f"[pipeline] add done: {summary.get('add_snapshot_path')}", flush=True)
        elif step == "search":
            records = await _run_search(config, workspace_dir)
            print(f"[pipeline] search done: {len(records)} records", flush=True)
        elif step == "answer":
            records = await _run_answer(config, workspace_dir)
            print(f"[pipeline] answer done: {len(records)} records", flush=True)
        elif step == "eval":
            records = await _run_eval(config, workspace_dir)
            _, eval_output = _answer_paths(workspace_dir, str(config.get("answer_prompt_mode") or "history"))
            print(f"[pipeline] eval done: {len(records)} records → {eval_output}", flush=True)
        elif step == "score":
            summary = await _run_score(config, workspace_dir)
            print(json.dumps(summary.get("overall", {}), ensure_ascii=False, indent=2), flush=True)
        elapsed = time.perf_counter() - started
        print(f"[pipeline] <<< {step} finished in {elapsed:.1f}s", flush=True)

    print("\n[pipeline] all steps completed.", flush=True)


async def run_pipeline(
    *,
    config_path: Path,
    start_from_step: str = "add",
) -> None:
    """加载 YAML 配置并执行流水线。"""
    load_runtime_env()
    config = _load_config(config_path)
    await run_pipeline_from_config(
        config,
        start_from_step=start_from_step,
        config_path=config_path,
    )


def main() -> None:
    """CLI 入口：解析 --config 与 --start-from-step 后启动 asyncio 流水线。"""
    parser = argparse.ArgumentParser(description="LoCoMo evaluation_pipeline (standalone)")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--start-from-step",
        default="add",
        choices=PIPELINE_STEPS,
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PIPELINE_DIR / args.config
    asyncio.run(run_pipeline(config_path=config_path, start_from_step=args.start_from_step))


if __name__ == "__main__":
    main()
