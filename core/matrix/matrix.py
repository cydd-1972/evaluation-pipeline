"""矩阵实验：目录规划、配置合并、LLM 环境切换。"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.infra.env import load_runtime_env

PIPELINE_DIR = Path(__file__).resolve().parents[2]
RAW_ADD_MODEL_ID = "raw"


@dataclass(frozen=True)
class AddModelSpec:
    id: str
    model: str
    api_key: str = ""
    api_base: str = ""
    llm_thinking_mode: str = ""  # split | disabled | 空=默认


@dataclass(frozen=True)
class MatrixRunSpec:
    """单次可执行实验（add 或 search 子运行）。"""

    run_id: str
    add_model_id: str
    add_repeat_index: int | None  # add 第几次（1..add_repeats）；search 时指向对应 add_run
    search_backend: str | None  # add-only 时为 None
    repeat_index: int | None  # 与 add_repeat_index 相同，保留兼容
    workspace_name: str
    workspace_dir: Path
    start_from_step: str
    is_add: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "add_model_id": self.add_model_id,
            "add_repeat_index": self.add_repeat_index,
            "search_backend": self.search_backend,
            "repeat_index": self.repeat_index,
            "workspace_name": self.workspace_name,
            "workspace_dir": str(self.workspace_dir),
            "start_from_step": self.start_from_step,
            "is_add": self.is_add,
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return payload


def load_matrix_bundle(
    *,
    matrix_config_path: Path | None = None,
    secrets_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix_path = matrix_config_path or (PIPELINE_DIR / "configs" / "matrix.yaml")
    secrets_file = secrets_path or (PIPELINE_DIR / "configs" / "matrix_secrets.yaml")
    matrix_cfg = _load_yaml(matrix_path)
    secrets: dict[str, Any] = {}
    if secrets_file.exists():
        secrets = _load_yaml(secrets_file)
    return matrix_cfg, secrets


def parse_add_models(matrix_cfg: dict[str, Any], secrets: dict[str, Any]) -> list[AddModelSpec]:
    load_runtime_env()
    default_key = os.getenv("OPENAI_API_KEY", "").strip()
    default_base = os.getenv("OPENAI_API_BASE", "").strip()
    models: list[AddModelSpec] = []
    for raw in matrix_cfg.get("add_models") or []:
        if not isinstance(raw, dict):
            continue
        model_id = str(raw.get("id") or "").strip()
        if not model_id:
            continue
        secret_block = secrets.get(model_id) if isinstance(secrets.get(model_id), dict) else {}
        api_key = str(raw.get("api_key") or secret_block.get("api_key") or "").strip() or default_key
        api_base = str(raw.get("api_base") or secret_block.get("api_base") or "").strip() or default_base
        models.append(
            AddModelSpec(
                id=model_id,
                model=str(raw.get("model") or "").strip(),
                api_key=api_key,
                api_base=api_base,
                llm_thinking_mode=str(raw.get("llm_thinking_mode") or "").strip().lower(),
            )
        )
    if not models:
        raise ValueError("matrix.yaml add_models is empty")
    return models


def matrix_root(matrix_cfg: dict[str, Any], *, base_dir: Path | None = None) -> Path:
    base = Path(str(matrix_cfg.get("matrix_base_dir") or "workspaces/matrix"))
    if base.is_absolute():
        return base
    root = base_dir or PIPELINE_DIR
    return root / base


def legacy_add_dir(root: Path, model_id: str) -> Path:
    """旧实验单次 add 目录 workspaces/matrix/<model>/_add。"""
    return root / model_id / "_add"


def add_db_workspace_name(matrix_cfg: dict[str, Any], model_id: str, add_repeat: int) -> str:
    """add_run01 复用旧库 eval_pipeline_matrix_<model>_add；其余用 add_run0N。"""
    if add_repeat == 1 and bool(matrix_cfg.get("reuse_legacy_add_run01", True)):
        return f"matrix_{model_id}_add"
    return f"matrix_{model_id}_add_run{add_repeat:02d}"


def add_run_is_reused(matrix_cfg: dict[str, Any], add_dir: Path, add_repeat: int) -> bool:
    """add_run01 且已有 add_snapshot（从旧 _add 迁移或已跑完）则不再 reset 库。"""
    if add_repeat != 1 or not bool(matrix_cfg.get("reuse_legacy_add_run01", True)):
        return False
    return (add_dir / "add_snapshot.json").exists() and (add_dir / "workspace.json").exists()


def add_workspace_dir(root: Path, model_id: str, add_repeat: int = 1) -> Path:
    return root / model_id / f"add_run{add_repeat:02d}"


def search_workspace_dir(root: Path, model_id: str, add_repeat: int, backend: str) -> Path:
    return add_workspace_dir(root, model_id, add_repeat) / backend


def raw_add_dir(root: Path) -> Path:
    return root / RAW_ADD_MODEL_ID / "add"


def raw_search_dir(root: Path, backend: str) -> Path:
    return root / RAW_ADD_MODEL_ID / backend


def raw_db_workspace_name() -> str:
    return "matrix_raw_add"


def resolve_pipeline_llm_model(matrix_cfg: dict[str, Any], models: list[AddModelSpec]) -> AddModelSpec:
    """raw add 无 LLM；search/answer 仍用此模型凭据。"""
    preferred = str(matrix_cfg.get("pipeline_llm_model_id") or "gemini").strip()
    for model in models:
        if model.id == preferred:
            return model
    if models:
        return models[0]
    raise ValueError("no add_models configured for pipeline_llm_model_id")


def apply_run_model_env(
    matrix_cfg: dict[str, Any],
    models: list[AddModelSpec],
    run: MatrixRunSpec,
) -> None:
    """add 用各自 add 模型；search_llm 用 pipeline_llm_model_id（如 deepseek）；rag 的 answer 仍用 add 模型。"""
    if run.is_add:
        if run.add_model_id == RAW_ADD_MODEL_ID:
            raise ValueError("raw add should not call apply_run_model_env")
        for model in models:
            if model.id == run.add_model_id:
                apply_add_model_env(model)
                return
        raise KeyError(f"unknown add model id: {run.add_model_id}")
    if run.add_model_id == RAW_ADD_MODEL_ID:
        apply_add_model_env(resolve_pipeline_llm_model(matrix_cfg, models))
        return
    if search_backend_uses_llm(run.search_backend):
        pipeline = resolve_pipeline_llm_model(matrix_cfg, models)
        apply_add_model_env(pipeline)
        return
    for model in models:
        if model.id == run.add_model_id:
            apply_add_model_env(model)
            return
    raise KeyError(f"unknown add model id: {run.add_model_id}")


def search_backend_uses_llm(backend: str | None) -> bool:
    return str(backend or "").strip().lower() in {"llm", "hybrid_llm"}


def build_base_pipeline_config(matrix_cfg: dict[str, Any]) -> dict[str, Any]:
    """从 matrix.yaml 生成与 run_pipeline 兼容的配置骨架。"""
    return {
        "dataset_path": str(matrix_cfg.get("dataset_path") or "datasets/locomo_refined.json"),
        "max_conversations": matrix_cfg.get("max_conversations"),
        "max_questions_per_conversation": matrix_cfg.get("max_questions_per_conversation"),
        "max_sessions_per_conversation": matrix_cfg.get("max_sessions_per_conversation"),
        "workspace_base_dir": str(matrix_cfg.get("matrix_base_dir") or "workspaces/matrix"),
        "database_prefix": str(matrix_cfg.get("database_prefix") or "eval_pipeline"),
        "reset_database_on_add": bool(matrix_cfg.get("reset_database_on_add", True)),
        "answer_prompt_mode": str(matrix_cfg.get("answer_prompt_mode") or "history"),
        "search_top_k": int(matrix_cfg.get("search_top_k") or 30),
        "search_llm_concurrency": int(matrix_cfg.get("search_llm_concurrency") or 1),
        "add_llm_concurrency": int(matrix_cfg.get("add_llm_concurrency") or 1),
        "concurrency": int(matrix_cfg.get("concurrency") or 2),
        "eval": copy.deepcopy(matrix_cfg.get("eval") or {"metrics": ["llm", "f1", "bleu"]}),
        "matrix_experiment": True,
        "reuse_legacy_add_run01": bool(matrix_cfg.get("reuse_legacy_add_run01", True)),
        "add_backend": str(matrix_cfg.get("add_backend") or "mem0").strip().lower(),
        "search_mode": str(matrix_cfg.get("search_mode") or "").strip().lower() or None,
        "add_history_window": int(matrix_cfg.get("add_history_window") or 2),
        "add_flush_per_session": bool(matrix_cfg.get("add_flush_per_session", True)),
        "pipeline_llm_model_id": str(matrix_cfg.get("pipeline_llm_model_id") or "gemini").strip(),
        "memory_decision_prompt": str(matrix_cfg.get("memory_decision_prompt") or "").strip() or None,
        "memory_prompt_max_items": matrix_cfg.get("memory_prompt_max_items"),
        "search_llm_prompt": matrix_cfg.get("search_llm_prompt"),
        "search_llm_require_non_empty": matrix_cfg.get("search_llm_require_non_empty"),
        "search_hybrid_recall_k": matrix_cfg.get("search_hybrid_recall_k"),
        "search_hybrid_rrf_k": matrix_cfg.get("search_hybrid_rrf_k"),
    }


def search_llm_client_spec(model: AddModelSpec) -> dict[str, str]:
    """写入 pipeline config，供 search 使用固定凭据（避免并行 run 踩 os.environ）。"""
    if not (model.api_key and model.api_base and model.model):
        raise RuntimeError(
            f"pipeline llm '{model.id}' missing api_key/api_base/model "
            f"(check .env or configs/matrix_secrets.yaml)"
        )
    spec: dict[str, str] = {
        "api_key": model.api_key,
        "api_base": model.api_base,
        "model": model.model,
    }
    if model.llm_thinking_mode:
        spec["llm_thinking_mode"] = model.llm_thinking_mode
    return spec


def apply_add_model_env(model: AddModelSpec) -> dict[str, str]:
    """设置当前进程的 OPENAI_*（add / search_llm 使用）。"""
    if not (model.api_key and model.api_base and model.model):
        raise RuntimeError(
            f"add model '{model.id}' missing api_key/api_base/model "
            f"(check .env or configs/matrix_secrets.yaml)"
        )
    os.environ["OPENAI_API_KEY"] = model.api_key
    os.environ["OPENAI_API_BASE"] = model.api_base
    os.environ["OPENAI_MODEL"] = model.model
    os.environ["key"] = model.api_key
    os.environ["api_base"] = model.api_base
    os.environ["model_name"] = model.model
    if model.llm_thinking_mode:
        os.environ["PIPELINE_LLM_THINKING_MODE"] = model.llm_thinking_mode
    else:
        os.environ.pop("PIPELINE_LLM_THINKING_MODE", None)
    return {
        "OPENAI_API_KEY": model.api_key,
        "OPENAI_API_BASE": model.api_base,
        "OPENAI_MODEL": model.model,
    }


def config_for_add_run(
    base: dict[str, Any],
    *,
    model: AddModelSpec,
    db_workspace_name: str,
    model_dir: Path,
    add_repeat_index: int = 1,
) -> dict[str, Any]:
    """产物目录 workspaces/matrix/<model>/add_run0N；DB 名 eval_pipeline_<db_workspace_name>。"""
    cfg = copy.deepcopy(base)
    cfg["workspace_base_dir"] = str(model_dir.parent.relative_to(PIPELINE_DIR)).replace("\\", "/")
    cfg["workspace_name"] = model_dir.name
    cfg["workspace_db_name"] = db_workspace_name
    cfg["add_model_id"] = model.id
    cfg["add_model"] = model.model
    cfg["add_backend"] = str(base.get("add_backend") or "mem0").strip().lower()
    cfg["search_mode"] = base.get("search_mode")
    cfg["add_history_window"] = base.get("add_history_window", 2)
    cfg["add_flush_per_session"] = base.get("add_flush_per_session", True)
    cfg["add_repeat_index"] = add_repeat_index
    cfg["search_backend"] = None
    cfg["progress_label"] = f"{model.id}/add_run{add_repeat_index:02d}"
    if add_run_is_reused(base, model_dir, add_repeat_index):
        cfg["reset_database_on_add"] = False
    return cfg


def config_for_raw_add_run(base: dict[str, Any], *, add_dir: Path) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["workspace_base_dir"] = str(add_dir.parent.relative_to(PIPELINE_DIR)).replace("\\", "/")
    cfg["workspace_name"] = add_dir.name
    cfg["workspace_db_name"] = raw_db_workspace_name()
    cfg["add_model_id"] = RAW_ADD_MODEL_ID
    cfg["add_model"] = ""
    cfg["add_backend"] = "raw"
    cfg["add_repeat_index"] = 1
    cfg["search_backend"] = None
    cfg["progress_label"] = f"{RAW_ADD_MODEL_ID}/add"
    return cfg


def config_for_raw_search_run(
    base: dict[str, Any],
    *,
    pipeline_model: AddModelSpec,
    search_backend: str,
    search_dir: Path,
    parent_add_workspace: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["workspace_base_dir"] = str(search_dir.parent.relative_to(PIPELINE_DIR)).replace("\\", "/")
    cfg["workspace_name"] = search_dir.name
    cfg["workspace_db_name"] = raw_db_workspace_name()
    cfg["add_model_id"] = RAW_ADD_MODEL_ID
    cfg["add_model"] = pipeline_model.model
    cfg["add_backend"] = "raw"
    cfg["add_repeat_index"] = 1
    cfg["search_backend"] = search_backend
    cfg["search_repeat"] = 1
    cfg["reset_database_on_add"] = False
    cfg["parent_add_workspace"] = parent_add_workspace
    cfg["progress_label"] = f"{RAW_ADD_MODEL_ID}/{search_backend}"
    if search_backend_uses_llm(search_backend):
        cfg["search_llm_client"] = search_llm_client_spec(pipeline_model)
    return cfg


def config_for_search_run(
    base: dict[str, Any],
    *,
    model_id: str,
    add_model: str,
    search_backend: str,
    add_repeat_index: int,
    db_workspace_name: str,
    search_dir: Path,
    parent_add_workspace: str,
    pipeline_llm: AddModelSpec | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["workspace_base_dir"] = str(search_dir.parent.relative_to(PIPELINE_DIR)).replace("\\", "/")
    cfg["workspace_name"] = search_dir.name
    cfg["workspace_db_name"] = db_workspace_name
    cfg["add_model_id"] = model_id
    cfg["add_model"] = add_model
    cfg["add_backend"] = str(base.get("add_backend") or "mem0").strip().lower()
    cfg["search_mode"] = base.get("search_mode")
    cfg["search_backend"] = search_backend
    cfg["add_repeat_index"] = add_repeat_index
    cfg["search_repeat"] = 1
    cfg["reset_database_on_add"] = False
    cfg["parent_add_workspace"] = parent_add_workspace
    cfg["progress_label"] = f"{model_id}/{search_backend}"
    if search_backend_uses_llm(search_backend) and pipeline_llm is not None:
        cfg["search_llm_client"] = search_llm_client_spec(pipeline_llm)
    return cfg


def link_add_database_to_search_run(add_dir: Path, search_dir: Path, *, extra: dict[str, Any]) -> None:
    """把 add 的 database_url 写入 search 子目录的 workspace.json。"""
    add_ws = add_dir / "workspace.json"
    if not add_ws.exists():
        raise FileNotFoundError(f"missing add workspace.json: {add_ws}")
    payload = json.loads(add_ws.read_text(encoding="utf-8"))
    database_url = str(payload.get("database_url") or "").strip()
    if not database_url:
        raise ValueError(f"add workspace.json missing database_url: {add_ws}")
    search_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "database_url": database_url,
        "parent_add_workspace": str(add_dir),
        **extra,
    }
    (search_dir / "workspace.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plan_matrix_runs(
    matrix_cfg: dict[str, Any],
    models: list[AddModelSpec],
    *,
    base_dir: Path | None = None,
) -> list[MatrixRunSpec]:
    """3 模型 × add_repeats 次 add × 每种 search 各 search_repeats 次。"""
    root = matrix_root(matrix_cfg, base_dir=base_dir)
    backends = [str(b).strip().lower() for b in (matrix_cfg.get("search_backends") or []) if str(b).strip()]
    add_repeats = max(1, int(matrix_cfg.get("add_repeats") or 3))
    search_repeats = max(1, int(matrix_cfg.get("search_repeats") or 1))
    runs: list[MatrixRunSpec] = []

    for model in models:
        for add_repeat in range(1, add_repeats + 1):
            add_name = add_db_workspace_name(matrix_cfg, model.id, add_repeat)
            add_dir = add_workspace_dir(root, model.id, add_repeat)
            runs.append(
                MatrixRunSpec(
                    run_id=f"{model.id}__add__run{add_repeat:02d}",
                    add_model_id=model.id,
                    add_repeat_index=add_repeat,
                    search_backend=None,
                    repeat_index=add_repeat,
                    workspace_name=add_name,
                    workspace_dir=add_dir,
                    start_from_step="add",
                    is_add=True,
                )
            )
            for backend in backends:
                for search_repeat in range(1, search_repeats + 1):
                    suffix = f"_{search_repeat:02d}" if search_repeats > 1 else ""
                    ws_name = f"matrix_{model.id}_add_run{add_repeat:02d}_{backend}{suffix}"
                    run_id = f"{model.id}__add_run{add_repeat:02d}__{backend}{suffix}"
                    search_dir = search_workspace_dir(root, model.id, add_repeat, backend)
                    if search_repeats > 1:
                        search_dir = search_dir.parent / f"{backend}_run{search_repeat:02d}"
                    runs.append(
                        MatrixRunSpec(
                            run_id=run_id,
                            add_model_id=model.id,
                            add_repeat_index=add_repeat,
                            search_backend=backend,
                            repeat_index=add_repeat,
                            workspace_name=ws_name,
                            workspace_dir=search_dir,
                            start_from_step="search",
                            is_add=False,
                        )
                    )
    return runs


def plan_raw_matrix_runs(matrix_cfg: dict[str, Any], *, base_dir: Path | None = None) -> list[MatrixRunSpec]:
    """raw add（无 LLM）× search(llm/rag) 各 1 次 = 3 runs。"""
    root = matrix_root(matrix_cfg, base_dir=base_dir)
    backends = [str(b).strip().lower() for b in (matrix_cfg.get("search_backends") or []) if str(b).strip()]
    add_dir = raw_add_dir(root)
    runs: list[MatrixRunSpec] = [
        MatrixRunSpec(
            run_id=f"{RAW_ADD_MODEL_ID}__add",
            add_model_id=RAW_ADD_MODEL_ID,
            add_repeat_index=1,
            search_backend=None,
            repeat_index=1,
            workspace_name=raw_db_workspace_name(),
            workspace_dir=add_dir,
            start_from_step="add",
            is_add=True,
        )
    ]
    for backend in backends:
        runs.append(
            MatrixRunSpec(
                run_id=f"{RAW_ADD_MODEL_ID}__{backend}",
                add_model_id=RAW_ADD_MODEL_ID,
                add_repeat_index=1,
                search_backend=backend,
                repeat_index=1,
                workspace_name=f"matrix_{RAW_ADD_MODEL_ID}_{backend}",
                workspace_dir=raw_search_dir(root, backend),
                start_from_step="search",
                is_add=False,
            )
        )
    return runs


def plan_all_matrix_runs(
    matrix_cfg: dict[str, Any],
    models: list[AddModelSpec],
    *,
    base_dir: Path | None = None,
) -> list[MatrixRunSpec]:
    add_mode = str(matrix_cfg.get("add_mode") or "mem0").strip().lower()
    if add_mode == "raw":
        return plan_raw_matrix_runs(matrix_cfg, base_dir=base_dir)
    return plan_matrix_runs(matrix_cfg, models, base_dir=base_dir)


def write_manifest(path: Path, *, matrix_cfg: dict[str, Any], models: list[AddModelSpec], runs: list[MatrixRunSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix_config": {
            "add_mode": matrix_cfg.get("add_mode"),
            "add_backend": matrix_cfg.get("add_backend"),
            "dataset_path": matrix_cfg.get("dataset_path"),
            "add_repeats": matrix_cfg.get("add_repeats"),
            "reuse_legacy_add_run01": matrix_cfg.get("reuse_legacy_add_run01"),
            "search_backends": matrix_cfg.get("search_backends"),
            "search_repeats": matrix_cfg.get("search_repeats"),
            "add_models": [{"id": m.id, "model": m.model} for m in models],
        },
        "total_runs": len(runs),
        "add_runs": sum(1 for r in runs if r.is_add),
        "search_runs": sum(1 for r in runs if not r.is_add),
        "runs": [r.to_dict() for r in runs],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
