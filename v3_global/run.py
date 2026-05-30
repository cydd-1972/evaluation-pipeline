"""v3 global：conversation 级记忆 + global search。"""
from __future__ import annotations

import sys
from pathlib import Path

VERSION_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = VERSION_DIR.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from core.pipeline.runner import run_pipeline_cli

if __name__ == "__main__":
    run_pipeline_cli(
        version_dir=VERSION_DIR,
        pipeline_dir=PIPELINE_DIR,
        default_add_backend="global",
    )
