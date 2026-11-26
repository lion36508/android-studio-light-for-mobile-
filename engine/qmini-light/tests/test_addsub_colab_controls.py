from __future__ import annotations

import ast
import json
from pathlib import Path


def test_addsub_pretraining_colab_exposes_safe_cpu_thread_control():
    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_AddSub_V1_From_Scratch.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    pretraining_cell = "".join(notebook["cells"][6]["source"])

    ast.parse(pretraining_cell)
    assert "PRETRAIN_NUM_THREADS = 0" in pretraining_cell
    assert "PRETRAIN_NUM_THREADS must be a non-negative integer." in pretraining_cell
    assert "command += ['--threads', str(PRETRAIN_NUM_THREADS)]" in pretraining_cell
    assert "if PRETRAIN_NUM_THREADS > 0:" in pretraining_cell
