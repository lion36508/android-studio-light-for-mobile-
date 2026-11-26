import ast
import json
from pathlib import Path



def test_colab_setup_preserves_preinstalled_cuda_pytorch():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])

    assert "'--no-deps','-e','.'" in setup
    assert "'pip','install','--upgrade'" not in setup
    assert "from packaging.version import Version" in setup
    assert "Version('2.10.0')" in setup
    assert "pyarrow>=15.0" in setup
    assert "pip','install','-e','.[dev,colab]'" not in setup


def test_colab_expensive_stages_fail_closed_without_gpu_and_preflight_memory():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    light_stage = "".join(notebook["cells"][8]["source"])
    long_stage = "".join(notebook["cells"][10]["source"])
    sft_stage = "".join(notebook["cells"][12]["source"])

    assert "if not torch.cuda.is_available():" in light_stage
    assert "if not torch.cuda.is_available():" in long_stage
    assert "torch.cuda.mem_get_info(0)" in long_stage
    assert "LONG_MIN_FREE_GPU_GIB = 8.0" in long_stage
    assert "if not torch.cuda.is_available():" in sft_stage
    assert "torch.cuda.mem_get_info(0)" in sft_stage
    assert "LONG_TARGET_TOKENS_OVERRIDE must be a positive integer or None." in long_stage

def test_colab_long_training_has_no_fixed_optimizer_step_cap():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    long_stage = "".join(notebook["cells"][10]["source"])

    assert "LONG_MAX_OPTIMIZER_STEPS" not in long_stage
    assert "LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "BASE_STEPS = LONG_OPTIMIZER_STEPS_OVERRIDE" in long_stage
    assert "if LONG_OPTIMIZER_STEPS_OVERRIDE is None:" in long_stage
    assert "if TARGET_TOKENS < TOKENS_PER_OPTIMIZER_STEP:" in long_stage
    assert "No arbitrary maximum optimizer-step cap is applied." in long_stage


def test_colab_training_targets_do_not_change_checkpoint_run_identity():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    long_stage = "".join(notebook["cells"][10]["source"])
    long_identity = long_stage.split("base_identity = json.dumps({", 1)[1].split("BASE_KEY =", 1)[0]
    assert "'steps': BASE_STEPS" not in long_identity
    assert "'target_tokens': ACTUAL_TOKENS" not in long_identity
    assert "args.append('--extend-schedule')" in long_stage

    light_stage = "".join(notebook["cells"][8]["source"])
    five_m_identity = light_stage.split("identity_light = {", 1)[1].split("KEY_5M =", 1)[0]
    assert "'steps': INTERMEDIATE_LIGHT_STEPS" not in five_m_identity
    assert "ARGS_LIGHT.append('--extend-schedule')" in light_stage

    sft_stage = "".join(notebook["cells"][12]["source"])
    sft_identity = sft_stage.split("json.dumps({", 1)[1].split("SFT_KEY =", 1)[0]
    assert "'steps': SFT_STEPS" not in sft_identity
    assert "sft_args.append('--extend-schedule')" in sft_stage

def test_sft_never_silently_selects_the_tiny_smoke_checkpoint():
    notebook_path = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'legacy_colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][12]['source'])
    assert 'SFT_SOURCE_OVERRIDE = None' in sft_stage
    assert 'SFT_MINIMUM_SOURCE_PARAMETERS = 500_000' in sft_stage
    assert 'base_*/final_model/model_manifest.json' in sft_stage
    assert 'qmini-final-model-v2' in sft_stage
    assert "SFT_SOURCE = SMOKE/'final_model'" not in sft_stage
    assert 'The tiny smoke model is never selected silently.' in sft_stage
    assert sft_stage.index('SFT_SOURCE = select_sft_source()') > sft_stage.index('if not sft_files:')


def test_sft_requires_independent_task_and_conversation_holdouts():
    notebook_path = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'legacy_colab' / 'QMini_Training.ipynb'
    notebook = json.loads(notebook_path.read_text(encoding='utf-8'))
    sft_stage = ''.join(notebook['cells'][12]['source'])
    gate = sft_stage.index('if not SFT_HELDOUT_TASKS.is_file() or not SFT_HELDOUT_CONVERSATIONS.is_file():')
    training_run = sft_stage.index("SFT_RUN = ART/'runs'/f'sft_{SFT_KEY}'")
    identity = sft_stage.index("sft_identity.update(SFT_HELDOUT_TASKS_SHA256.encode('utf-8'))")
    assert gate < training_run
    assert identity < sft_stage.index('SFT_KEY =')
    assert 'training_file_hashes.intersection(heldout_hashes)' in sft_stage



def test_colab_runs_random_brain_architecture_preflight_before_dataset_loading():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    setup = "".join(notebook["cells"][2]["source"])
    preflight = "".join(notebook["cells"][3]["source"])
    tokenizer_data_stage = "".join(notebook["cells"][4]["source"])

    assert "qmini.cli','preflight'" in preflight
    assert "qmini_light_0.7m.json" in preflight
    assert "'--sequence-length','8'" in preflight
    assert "if torch.cuda.is_available():" in preflight
    assert "qmini_light_0.7m.json" in preflight
    assert "qmini_smoke.json" in preflight
    assert "PREFLIGHT_DEVICE = 'cpu'" in preflight
    assert "'--device',PREFLIGHT_DEVICE" in preflight
    assert "optimizer step" in preflight
    assert "weights updated" in preflight
    import ast
    ast.parse(preflight)
    assert "subprocess.run(ARCHITECTURE_PREFLIGHT,check=True)" in preflight
    assert "qmini.cli','tokenizer'" in tokenizer_data_stage
    assert setup.index("pip','install','--no-deps','-e','.'") < len(setup)

def test_colab_smoke_learning_proof_loads_training_checkpoint_not_dense_snapshot():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    smoke_stage = "".join(notebook["cells"][6]["source"])

    assert "checkpoint_payload = torch.load(checkpoint_path" in smoke_stage
    assert "checkpoint_payload.get('format') != 'qmini-checkpoint-v2'" in smoke_stage
    assert "trained_state = checkpoint_payload.get('model')" in smoke_stage
    assert "SMOKE/'final_model'/'model.pt'" not in smoke_stage
    assert "minimum_nll_improvement': 0.05" in smoke_stage
    assert "before_report['cross_entropy'] - after_report['cross_entropy'] >= 0.05" in smoke_stage
    assert "if not checkpoint_path.is_file():" in smoke_stage
    assert smoke_stage.index("if not checkpoint_path.is_file():") < smoke_stage.index(
        "checkpoint_payload = torch.load(checkpoint_path"
    )

def test_training_ready_builder_bundles_dormant_training_runtime_and_never_starts_it():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "legacy_colab"
        / "QMini_Training_Ready_Build.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    import ast
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{notebook_path}:code-cell-{index}")

    settings = code_cells[0]
    build = code_cells[4]
    final_guard = code_cells[5]
    assert 'MODEL_CONFIG_REL = "configs/qmini_light_0.7m.json"' in settings
    assert "RUN_TRAINING_NOW" not in settings
    assert "build-initial" in build
    assert "verify_packed_export" in build
    assert "_verify_initial_brain_artifact" in build
    assert "_verify_prepared_dataset_manifest" in build
    assert "training_runtime" in build
    assert '"enabled": False' in build
    assert "training_progress.json" in build
    assert "training_runtime_enabled" in build
    assert "training_runtime_manifest_sha256" in build
    assert 'model_manifest.get("training_runtime_manifest_sha256") != sha256_file(runtime_manifest_path)' in build
    assert "QMini_Training_Runner.ipynb" in final_guard
    assert "train_model(" not in build
    markdown = "".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    assert "QMini_Training_Runner.ipynb" in markdown


def test_training_ready_package_identity_includes_preparation_contract():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "legacy_colab"
        / "QMini_Training_Ready_Build.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    identity_cell = next(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        and "package_identity =" in "".join(cell.get("source", []))
    )
    assert '"loss_mask_mode": LOSS_MASK_MODE' in identity_cell
    assert '"minimum_sequence_length": TRAIN_SEQUENCE_LENGTH' in identity_cell
def test_training_runner_is_separate_and_has_verified_resume_controls():
    notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / "QMini_Training_Runner.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    import ast
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{notebook_path}:code-cell-{index}")
    settings = code_cells[0]
    run = "".join(code_cells)
    assert "START_TRAINING = False" in settings
    assert "restore_latest_checkpoint" in run
    assert "TrainingSessionLease" in run
    assert "checkpoint_mirror_dir" in run
    assert "checkpoint_session_id" in run
    assert "mark_training_status" in run
    assert "model.qmini.npz" in run
    assert "training_runtime_manifest_sha256" in run
    assert "No checkpoint restore or optimizer update has run in this cell." in code_cells[3]
    assert "restore_latest_checkpoint" not in code_cells[3]
    activation = next(
        source for source in code_cells
        if "with TrainingSessionLease(" in source and "restore_latest_checkpoint" in source
    )
    assert activation.index("with TrainingSessionLease(") < activation.index("restore_latest_checkpoint")
    assert activation.index("restore_latest_checkpoint") < activation.index("train_model(")




def test_light_profile_reduces_size_but_keeps_core_model_mechanisms():
    from qmini.config import QMiniConfig

    config_path = Path(__file__).resolve().parents[1] / "configs" / "qmini_light_0.7m.json"
    config = QMiniConfig.from_json(config_path)
    assert config.estimated_parameter_count() == 705_742
    assert config.d_model == 64
    assert config.d_hidden == 128
    assert config.n_layers == 6
    assert config.n_experts == 2
    assert config.top_k == 1
    assert config.use_state_space is True
    assert config.gradient_checkpointing is True
    assert config.max_thinking_steps == 4
    assert config.max_context == 512


def test_training_ready_and_runner_notebooks_contain_executable_code():
    for name in (
        "QMini_Training_Ready_Build.ipynb",
        "QMini_Training_Runner.ipynb",
    ):
        notebook_path = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "legacy_colab" / name
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            (index, cell)
            for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"
        ]
        assert code_cells, name
        for index, cell in code_cells:
            source_parts = cell.get("source", [])
            source = source_parts if isinstance(source_parts, str) else "".join(source_parts)
            assert source.strip(), f"{name}:cell-{index} is empty"
            parsed = ast.parse(source, filename=f"{name}:cell-{index}")
            assert parsed.body, f"{name}:cell-{index} is comment-only or a no-op"


def test_training_runner_defaults_match_the_bundled_package_contract():
    root = Path(__file__).resolve().parents[1]
    runner = json.loads(
        (root / "tests" / "fixtures" / "legacy_colab" / "QMini_Training_Runner.ipynb").read_text(encoding="utf-8")
    )
    builder = json.loads(
        (root / "tests" / "fixtures" / "legacy_colab" / "QMini_Training_Ready_Build.ipynb").read_text(encoding="utf-8")
    )
    settings = "".join(runner["cells"][1]["source"])
    setup = "".join(runner["cells"][2]["source"])
    validation = "".join(runner["cells"][3]["source"])
    package_builder = "".join(builder["cells"][5]["source"])

    assert 'TRAINING_DEFAULTS_PATH = RUNTIME_DIR / "TRAINING_DEFAULTS.json"' in setup
    assert "TRAINING_DEFAULTS_PATH.is_file()" in setup
    assert "training_defaults = json.loads(TRAINING_DEFAULTS_PATH.read_text" in validation
    assert 'runtime_dir / "TRAINING_DEFAULTS.json"' in package_builder
    assert '"checkpoint_every_optimizer_steps": 10' in package_builder
    assert "CHECKPOINT_EVERY = 10" in settings
    assert "SEQUENCE_LENGTH = 512" in settings
    builder_settings = "".join(builder["cells"][1]["source"])
    assert "TRAIN_SEQUENCE_LENGTH = 512" in builder_settings
    assert '"sequence_length": TRAIN_SEQUENCE_LENGTH' in package_builder
    assert "config.max_context != 512" in validation
    assert "packaged_sequence_length != config.max_context" in validation


def test_one_click_colab_notebook_builds_only_and_bundles_training_runtime():
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "legacy_colab"
        / "QMini_One_Click_Training.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert code_cells
    for index, source in enumerate(code_cells, start=1):
        ast.parse(source, filename=f"{notebook_path}:code-cell-{index}")

    all_code = "\n".join(code_cells)
    settings = code_cells[0]
    build = next(source for source in code_cells if "build-initial" in source)
    final_guard = code_cells[-1]
    assert 'MODEL_CONFIG_REL = "configs/qmini_light_0.7m.json"' in settings
    assert "TRAIN_SEQUENCE_LENGTH = 512" in settings
    assert "START_TRAINING" not in all_code
    assert "train_model(" not in all_code
    assert "TrainingSessionLease" not in all_code
    assert "restore_latest_checkpoint" not in all_code
    assert 'drive.mount("/content/drive")' in all_code
    assert "https://github.com/lion36508/Q-mini-light-0.7M.git" in all_code
    assert "QMini-Light-0.7M/datasets" in all_code
    assert "build-initial" in build
    assert "training_runtime" in build
    assert '"enabled": False' in build
    assert "training_progress.json" in build
    assert "training_runtime_enabled" in build
    assert "config.max_context != 512" in all_code
    assert "No neural-model training was run in this build stage." in build
    assert "MODEL BUILD COMPLETE — TRAINING REMAINS DISABLED" in final_guard
    assert "QMini_Training_Runner.ipynb" in final_guard

def test_cpu_resume_notebook_trains_on_conversation_data():
    notebook_path = (
        Path(__file__).resolve().parents[1] / "colab" / "QMini_English_CPU_One_Click_Resume.ipynb"
    )
    raw = notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(raw)
    assert len(notebook["cells"]) == 2
    markdown = "".join(notebook["cells"][0]["source"])
    code = "".join(notebook["cells"][1]["source"])
    ast.parse(code)
    assert "tinystories" not in raw.lower()
    assert "roneneldan" not in raw.lower()
    assert "V12" in markdown
    assert 'TRAINING_SOURCE_REVISION = "da4df686b8d1073c43b26ad9e6e02c0278ac622c"' in code
    assert "prepare_conversation_datasets" in code
    assert "allow_dataset_change" in code
    assert "QMini-English-512-Q2v2-e1a187f0ca1567cd" in code
    assert "dailydialog" in code
    assert "oasst1_en" in code
    assert "checkpoint-step-000000012540" in code
    assert "--loss-mask" in code
    assert '"assistant"' in code or "'assistant'" in code

def test_english_cpu_only_resume_v9_is_syntax_valid_and_uses_safe_resume_flow():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "tests" / "fixtures" / "legacy_colab" / "QMini_English_One_Click_CPU_Only_Resume_V9.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook.get("nbformat") == 4
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]
    assert len(code_cells) == 1
    source = code_cells[0]
    parsed = ast.parse(source, filename=str(notebook_path))
    assert parsed.body

    # Catch the historical Path NameError and ensure the CPU lock is set
    # before importing PyTorch in a fresh Colab runtime.
    assert source.index("from pathlib import Path") < source.index('Path("/content/drive/MyDrive")')
    assert source.index('os.environ["CUDA_VISIBLE_DEVICES"] = ""') < source.index("import torch")
    assert "CPU-only lock did not take effect." in source
    assert "torch.cuda.is_available()" in source

    # The active notebook must keep the intended model/data contract and
    # use the checked Drive checkpoint-resume path rather than reinitializing.
    assert "TRAIN_SEQUENCE_LENGTH = 512" in source
    assert "VOCAB_SIZE = 4096" in source
    assert "TARGET_OPTIMIZER_STEPS = 20000" in source
    assert "BATCH_SIZE = 4" in source
    assert "GRAD_ACCUM = 2" in source
    assert "TrainingSessionLease(" in source
    assert "restore_latest_checkpoint(" in source
    assert "resume=resume_run" in source
    assert "init_from_initial=None if resume_run else INITIAL_MODEL_PATH" in source

