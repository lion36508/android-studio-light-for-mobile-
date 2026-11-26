import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini_addsub.config import AddSubConfig
from qmini_addsub.data import (
    accept_oasst1_message,
    escape_special_token_literals,
    normalize_training_text,
    stable_split,
)
from qmini_addsub.model import AddSubLanguageModel, AddSubLinear, SparseTop1Experts
from qmini_addsub.optim import assert_finite_training_state, build_adamw, cosine_warmup_lr, reconcile_metrics_file
from qmini_addsub.quantization import (
    codes_to_values,
    pack_3bit_codes,
    quantize_to_codes,
    unpack_3bit_codes,
    quantize_weight_ste,
    weight_scale,
)
from qmini_addsub.runtime import additive_projection, saturating_add
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from qmini_addsub.sft import ConversationDataset, _load_conversations


def test_colab_addsub_notebook_is_separate_and_training_is_off_by_default():
    import json

    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "colab"
        / "QMini_AddSub_V1_From_Scratch.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    for index, source in enumerate(code):
        ast.parse(source, filename=f"{notebook_path}:cell-{index}")
    all_code = "\n".join(code)
    assert "Q mini 2 light" in all_code
    assert "REPO_BRANCH = 'main'" in all_code
    assert "VERSION_ROOT = DRIVE_ROOT / 'trainability_addsub_main_v1'" in all_code
    assert "REPO_COMMIT = subprocess.check_output" in all_code
    assert "algorithm/addsub-stability-routing-v2" not in all_code
    assert "trainability_v2" not in all_code
    assert "PREPARE_PINNED_CORPUS = False" in all_code
    assert "IMPORT_USER_DATA = False" in all_code
    assert "BUILD_TOKENIZER_AND_STREAMS = False" in all_code
    assert "START_PRETRAIN = False" in all_code
    assert "PRETRAIN_EXTEND_SCHEDULE = False" in all_code
    assert "SFT_EXTEND_SCHEDULE = False" in all_code
    assert all_code.count("['--extend-schedule']") == 2
    assert "START_SFT = False" in all_code
    assert "EXPORT_PACKED_WEIGHTS = False" in all_code
    assert "RUN_HELDOUT_EVAL = False" in all_code
    assert "qmini_addsub.evaluate" in all_code
    assert "ACCEPT_SOURCE_LICENSES = False" in all_code
    assert "DEVICE = 'cpu'" in all_code
    assert "qmini_addsub.export" in all_code
    assert "--sft-test" in all_code
    assert "--test-jsonl" in all_code
    assert "QMini-Light-0.7M/training_ready" not in all_code
    assert "qmini_addsub.train" in all_code
    assert "qmini_addsub.sft" in all_code


def test_addsub_config_is_independent_and_validates():
    config = AddSubConfig()
    config.validate()
    assert config.vocab_size == 1024
    assert config.max_context == 512
    with pytest.raises(ValueError, match="positive"):
        AddSubConfig(n_layers=0).validate()


def test_five_symbol_quantizer_preserves_exact_zero_and_three_bit_round_trip():
    values = np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5, 0.0], dtype=np.float32)
    codes = quantize_to_codes(values)
    assert codes.tolist() == [0, 1, 2, 3, 4, 2]
    assert codes_to_values(codes).tolist() == values.tolist()
    packed = pack_3bit_codes(codes)
    restored = unpack_3bit_codes(packed, codes.shape)
    assert np.array_equal(restored, codes)

    tampered_padding = bytearray(packed)
    tampered_padding[-1] |= 0b10000000
    with pytest.raises(ValueError, match="padding bits"):
        unpack_3bit_codes(tampered_padding, codes.shape)
    with pytest.raises(ValueError, match="dimensions"):
        unpack_3bit_codes(packed, (True, 6))


def test_scaled_five_symbol_quantizer_and_zero_master_gradient():
    values = torch.tensor(
        [[-2.0, -1.5, -1.25, -1.0, -0.75],
         [-0.5, -0.25, 0.0, 0.25, 0.5],
         [0.75, 1.0, 1.25, 1.5, 2.0]],
        dtype=torch.float32, requires_grad=True,
    )
    scale = weight_scale(values)
    actual = quantize_weight_ste(values)
    codes = quantize_to_codes(values.detach().numpy(), scale=float(scale.item()))
    expected = torch.from_numpy(codes_to_values(codes)) * scale
    torch.testing.assert_close(actual.detach(), expected, rtol=0.0, atol=0.0)
    codebook = torch.tensor([-1.5,-0.5,0.0,0.5,1.5])
    assert torch.isclose(actual.detach()/scale,codebook.view(1,1,5),atol=1e-5,rtol=0.0).any(dim=-1).all()
    actual.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))

    layer = torch.nn.Linear(4,4,bias=False)
    with torch.no_grad():
        layer.weight.zero_()
    output = torch.nn.functional.linear(torch.ones(2,4),quantize_weight_ste(layer.weight))
    output.sum().backward()
    assert torch.count_nonzero(layer.weight.grad)==layer.weight.numel()
    optimizer = build_adamw(layer,learning_rate=1e-2,weight_decay=0.0)
    optimizer.step()
    assert torch.count_nonzero(layer.weight)==layer.weight.numel()
    assert torch.count_nonzero(quantize_weight_ste(layer.weight).detach())>0


def test_five_symbol_packer_rejects_reserved_codes():
    with pytest.raises(ValueError, match="0..4"):
        pack_3bit_codes(np.asarray([7], dtype=np.uint8))
    with pytest.raises(ValueError, match="Reserved"):
        unpack_3bit_codes(bytes([0b00000111]), (1,))


def test_additive_projection_matches_five_symbol_reference():
    x = np.asarray([[2, -3, 4]], dtype=np.int8)
    codes = np.asarray([[4, 0, 2], [1, 3, 4]], dtype=np.uint8)
    actual = additive_projection(x, codes)
    levels = np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32)
    decoded = levels[codes]
    expected = (x.astype(np.int64) @ decoded.T * 2).astype(np.int64)
    assert np.array_equal(actual, expected)


def test_zero_symbol_adds_no_projection_contribution():
    activations = np.asarray([[12, -8, 5]], dtype=np.int8)
    zero_weights = np.full((4, 3), 2, dtype=np.uint8)
    assert np.array_equal(
        additive_projection(activations, zero_weights),
        np.zeros((1, 4), dtype=np.int64),
    )


def test_sft_dataset_masks_user_context_and_supervises_assistant_end_token():
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    dataset = ConversationDataset(
        [{"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]}],
        tokenizer,
        max_context=64,
    )
    token_ids, labels = dataset[0]
    user_position = token_ids.index(vocab["<|user|>"])
    assistant_position = token_ids.index(vocab["<|assistant|>"])
    assert labels[user_position + 1] == -100
    assert labels[assistant_position] == -100
    assert labels[assistant_position + 1] == vocab["<unk>"]
    assert labels[-1] == vocab["<|end|>"]


def test_sft_loader_accepts_only_curated_human_non_synthetic_sources(tmp_path):
    import json

    path = tmp_path / "sft_train.jsonl"
    record = {
        "source": "OpenAssistant/oasst1",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "What is a compiler?"},
            {"role": "assistant", "content": "A compiler translates source code."},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert len(_load_conversations(path)) == 1

    record["synthetic"] = True
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic/unverified"):
        _load_conversations(path)

    record["synthetic"] = False
    record["source"] = "unknown/copied-corpus"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        _load_conversations(path)


def test_additive_runtime_has_no_multiply_divide_or_matmul_ast():
    source = Path(__file__).resolve().parents[1] / "qmini_addsub" / "runtime.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden = (ast.Mult, ast.Div, ast.FloorDiv, ast.MatMult)
    violations = [
        node.lineno for node in ast.walk(tree) if isinstance(node, forbidden)
    ]
    assert violations == []


def test_oasst_filter_fails_closed_on_synthetic_unreviewed_and_missing_rank():
    base = {
        "message_id": "m1",
        "parent_id": None,
        "text": "A genuine English response.",
        "role": "assistant",
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "rank": 0,
        "synthetic": False,
        "message_tree_id": "tree-1",
        "tree_state": "ready_for_export",
    }
    assert accept_oasst1_message(base) == (True, "accepted")
    for field, value, reason in (
        ("synthetic", True, "synthetic"),
        ("review_result", False, "unreviewed"),
        ("rank", None, "missing_rank"),
        ("rank", 1, "non_top_rank"),
        ("tree_state", "halted_by_moderator", "not_ready"),
    ):
        invalid = dict(base)
        invalid[field] = value
        assert accept_oasst1_message(invalid) == (False, reason)
    missing = dict(base)
    missing.pop("synthetic")
    assert accept_oasst1_message(missing) == (False, "missing_fields")


def test_special_token_literals_are_escaped_before_training():
    escaped = escape_special_token_literals(
        "literal <|assistant|> text and <bos> markers"
    )
    assert "<|assistant|>" not in escaped
    assert "<bos>" not in escaped
    assert "literal" in escaped


def test_clean_text_rejects_secret_like_tokens():
    from qmini_addsub.data import is_clean_english_text

    assert not is_clean_english_text("Please use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456")
    assert not is_clean_english_text("My email is person@example.com")
    assert is_clean_english_text("A normal English sentence.")


def test_group_split_is_deterministic_and_group_based():
    assert stable_split("same-conversation") == stable_split("same-conversation")
    assert stable_split("same-conversation") in {"train", "validation", "test"}
    with pytest.raises(ValueError):
        stable_split("")


def test_parallel_state_space_scan_matches_sequential_reference_and_gradients():
    from qmini_addsub.model import GatedDiagonalStateSpace

    torch.manual_seed(11)
    decay = torch.rand(2, 17, 5, dtype=torch.float32) * 0.8 + 0.1
    injection = torch.randn(2, 17, 5, dtype=torch.float32, requires_grad=True)
    actual = GatedDiagonalStateSpace.parallel_scan(decay, injection)
    state = torch.zeros(2, 5)
    expected = []
    for position in range(decay.shape[1]):
        state = decay[:, position] * state + injection[:, position]
        expected.append(state)
    expected = torch.stack(expected, dim=1)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.square().mean().backward()
    assert injection.grad is not None
    assert torch.isfinite(injection.grad).all()


def test_parallel_state_space_scan_is_causal():
    from qmini_addsub.model import GatedDiagonalStateSpace

    torch.manual_seed(12)
    decay = torch.full((1, 12, 3), 0.7)
    injection = torch.randn(1, 12, 3)
    changed_future = injection.clone()
    changed_future[:, 7:] = changed_future[:, 7:] + 50.0
    first = GatedDiagonalStateSpace.parallel_scan(decay, injection)
    second = GatedDiagonalStateSpace.parallel_scan(decay, changed_future)
    assert torch.allclose(first[:, :7], second[:, :7], atol=0.0, rtol=0.0)


def test_new_model_forward_from_random_initialization():
    torch.manual_seed(7)
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=2,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    model = AddSubLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 12))
    result = model(inputs)
    assert result["logits"].shape == (2, 12, config.vocab_size)
    assert torch.isfinite(result["logits"]).all()
    assert result["aux_loss"].ndim == 0
    assert model.parameter_count() > 0


def test_addsub_pretraining_runs_from_zero_saves_and_resumes_verified_checkpoint(tmp_path):
    import hashlib
    import json

    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from qmini_addsub.train import train_from_scratch
    from qmini_addsub.sft import fine_tune_from_addsub_checkpoint

    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=64,
        dropout=0.0,
    )
    special_tokens = [
        "<unk>", "<eos>", "<pad>", "<bos>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    ]
    tokenizer_vocab = {token: index for index, token in enumerate(special_tokens)}
    tokenizer_vocab.update({f"unused_{index}": index for index in range(8, 300)})
    tokenizer = Tokenizer(WordLevel(tokenizer_vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    streams = {}
    pretraining_splits = {}
    for split, count in (("train", 64), ("validation", 32), ("test", 32)):
        text_path = data_dir / f"pretrain_{split}.txt"
        text_path.write_text(f"English {split} sample text.\n", encoding="utf-8")
        raw_sha = hashlib.sha256(text_path.read_bytes()).hexdigest()
        pretraining_splits[split] = {"sha256": raw_sha}
        token_path = data_dir / f"pretrain_{split}.u16"
        values = (np.arange(count, dtype=np.uint16) % 299) + 1
        values.tofile(token_path)
        streams[split] = {
            "path": token_path.name,
            "tokens": count,
            "sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
            "source_sha256": raw_sha,
            "source_bytes": text_path.stat().st_size,
        }

    manifest = {
        "format": "qmini-addsub-data-v1",
        "language": "English",
        "synthetic_training_records_included": 0,
        "pretraining": {"splits": pretraining_splits},
        "token_streams": streams,
        "tokenizer": {
            "path": tokenizer_path.name,
            "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        },
    }
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    output_dir = tmp_path / "run"
    common = {
        "train_bin": data_dir / "pretrain_train.u16",
        "valid_bin": data_dir / "pretrain_validation.u16",
        "tokenizer_path": tokenizer_path,
        "output_dir": output_dir,
        "config": config,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "sequence_length": 8,
        "save_every": 1,
        "validate_every": 1,
        "device_name": "cpu",
        "seed": 42,
        "num_threads": 1,
    }
    first = train_from_scratch(steps=2, **common)
    checkpoint = output_dir / "checkpoint-step-000000001.pt"
    assert checkpoint.is_file()
    assert first["steps"] == 2
    assert first["heldout_test_token_loss"] > 0
    assert first["torch_num_threads"] == 1
    assert first["requested_num_threads"] == 1
    assert first["session_train_steps"] == 2
    assert first["session_target_tokens"] == 16
    assert first["session_train_step_seconds"] > 0
    assert first["session_train_tokens_per_second_excluding_validation"] > 0
    metric_rows = [
        json.loads(line)
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert metric_rows
    assert metric_rows[0]["train_steps_since_last_validation"] == 1
    assert metric_rows[0]["train_target_tokens_since_last_validation"] == 8
    assert metric_rows[0]["train_tokens_per_second_excluding_validation"] > 0
    assert metric_rows[0]["torch_num_threads"] == 1

    with pytest.raises(ValueError, match="num_threads must be a non-negative integer"):
        train_from_scratch(**{**common, "output_dir": tmp_path / "bad-threads", "num_threads": -1}, steps=1)

    import shutil

    # Simulate a crash after the step-2 metrics row was appended but before
    # checkpoint step 2 was committed. The recovery folder contains only the
    # step-1 checkpoint and the too-new metrics file, just like a real run.
    resumed_run_dir = tmp_path / "resumed-run"
    resumed_run_dir.mkdir()
    recovery_checkpoint = resumed_run_dir / checkpoint.name
    shutil.copy2(checkpoint, recovery_checkpoint)
    shutil.copy2(output_dir / "metrics.jsonl", resumed_run_dir / "metrics.jsonl")
    resumed_common = {**common, "output_dir": resumed_run_dir}
    second = train_from_scratch(
        steps=2, resume=recovery_checkpoint, **resumed_common
    )
    assert second["steps"] == 2
    final_pretrain_checkpoint = output_dir / "checkpoint-step-000000002.pt"
    resumed_pretrain_checkpoint = resumed_run_dir / "checkpoint-step-000000002.pt"
    assert final_pretrain_checkpoint.is_file()
    assert resumed_pretrain_checkpoint.is_file()
    recovered_metric_steps = [
        json.loads(line)["step"]
        for line in (resumed_run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert recovered_metric_steps == [1, 2]
    assert second["heldout_test_sha256"] == streams["test"]["sha256"]
    from qmini_addsub.generate import load_addsub_model

    loaded_model = load_addsub_model(
        final_pretrain_checkpoint,
        device_name="cpu",
        tokenizer_path=tokenizer_path,
    )
    assert loaded_model.config == config
    mismatched_tokenizer = tmp_path / "mismatched-tokenizer.json"
    mismatched_tokenizer.write_text(
        tokenizer_path.read_text(encoding="utf-8") + " ",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Tokenizer fingerprint"):
        load_addsub_model(
            final_pretrain_checkpoint,
            device_name="cpu",
            tokenizer_path=mismatched_tokenizer,
        )
    full_state = torch.load(final_pretrain_checkpoint, map_location="cpu", weights_only=True)
    resumed_state = torch.load(resumed_pretrain_checkpoint, map_location="cpu", weights_only=True)
    for name, tensor in full_state["model"].items():
        assert torch.equal(tensor, resumed_state["model"][name]), name

    # Extending the horizon is opt-in and keeps the existing run lineage intact.
    extended_pretrain_dir = tmp_path / "extended-pretrain"
    extended_pretrain_dir.mkdir()
    step_two = resumed_run_dir / "checkpoint-step-000000002.pt"
    shutil.copy2(step_two, extended_pretrain_dir / step_two.name)
    shutil.copy2(resumed_run_dir / "metrics.jsonl", extended_pretrain_dir / "metrics.jsonl")
    extended_common = {**common, "output_dir": extended_pretrain_dir}
    with pytest.raises(ValueError, match="extend_schedule=True"):
        train_from_scratch(
            steps=3, resume=extended_pretrain_dir / step_two.name, **extended_common
        )
    extended_result = train_from_scratch(
        steps=3, resume=extended_pretrain_dir / step_two.name,
        extend_schedule=True, **extended_common,
    )
    assert extended_result["schedule_total_steps"] == 3
    extended_state = torch.load(
        extended_pretrain_dir / "checkpoint-step-000000003.pt",
        map_location="cpu", weights_only=True,
    )
    assert extended_state["step"] == 3
    assert extended_state["schedule_total_steps"] == 3
    pretrain_metrics = [
        json.loads(line) for line in
        (extended_pretrain_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in pretrain_metrics] == [1, 2, 3]
    assert [row["schedule_total_steps"] for row in pretrain_metrics] == [2, 2, 3]

    def write_conversation(path):
        record = {
            "source": "databricks/databricks-dolly-15k",
            "human_authored": True,
            "synthetic": False,
            "messages": [
                {"role": "user", "content": f"What is a compiler in {path.stem}?"},
                {"role": "assistant", "content": f"A compiler translates program source for {path.stem}."},
            ],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sft_train = data_dir / "sft_train.jsonl"
    sft_valid = data_dir / "sft_validation.jsonl"
    sft_test = data_dir / "sft_test.jsonl"
    train_sft_sha = write_conversation(sft_train)
    valid_sft_sha = write_conversation(sft_valid)
    test_sft_sha = write_conversation(sft_test)
    manifest["instruction_splits"] = {
        "train": {"path": sft_train.name, "sha256": train_sft_sha},
        "validation": {"path": sft_valid.name, "sha256": valid_sft_sha},
        "test": {"path": sft_test.name, "sha256": test_sft_sha},
    }
    manifest["approved_instruction_sources"] = ["databricks/databricks-dolly-15k"]
    (data_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )

    sft_result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=tmp_path / "sft",
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
    )
    assert sft_result["steps"] == 2
    assert sft_result["heldout_test_response_loss"] > 0
    assert (tmp_path / "sft" / "sft-checkpoint-step-000000001.pt").is_file()

    sft_checkpoint = tmp_path / "sft" / "sft-checkpoint-step-000000001.pt"
    sft_state = torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
    assert sft_state["format"] == "qmini-addsub-sft-v7"
    assert "curriculum" in sft_state
    assert sft_state["curriculum"]["format"] == "qmini-self-paced-curriculum-v2"
    assert sft_state["curriculum"]["uniform_fraction"] == 1.0
    assert "torch_rng" in sft_state and "python_rng" in sft_state

    resumed_sft_dir = tmp_path / "sft-resumed"
    resumed_sft_dir.mkdir()
    recovery_sft_checkpoint = resumed_sft_dir / sft_checkpoint.name
    shutil.copy2(sft_checkpoint, recovery_sft_checkpoint)
    # SFT validates at the end of this two-step smoke run. Reusing its step-2
    # metrics with a step-1 checkpoint exercises the write-before-checkpoint crash.
    shutil.copy2(tmp_path / "sft" / "sft_metrics.jsonl", resumed_sft_dir / "sft_metrics.jsonl")
    resumed_sft = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=resumed_sft_dir,
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
        resume=recovery_sft_checkpoint,
    )
    assert resumed_sft["steps"] == 2
    assert resumed_sft["final_validation_response_loss"] > 0
    resumed_sft_checkpoint = resumed_sft_dir / "sft-checkpoint-step-000000002.pt"
    full_sft_checkpoint = tmp_path / "sft" / "sft-checkpoint-step-000000002.pt"
    assert resumed_sft_checkpoint.is_file()
    recovered_sft_metric_steps = [
        json.loads(line)["step"]
        for line in (resumed_sft_dir / "sft_metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert recovered_sft_metric_steps == [2]
    full_sft_state = torch.load(full_sft_checkpoint, map_location="cpu", weights_only=True)
    resumed_sft_state = torch.load(resumed_sft_checkpoint, map_location="cpu", weights_only=True)
    for name, tensor in full_sft_state["model"].items():
        assert torch.equal(tensor, resumed_sft_state["model"][name]), name

    # SFT v6 had the same uniform sampler but no curriculum state. It remains
    # exactly resumable in baseline mode; opting into curriculum requires new
    # per-record loss history and is rejected by the trainer.
    legacy_sft_dir = tmp_path / "sft-legacy-v6"
    legacy_sft_dir.mkdir()
    legacy_checkpoint = torch.load(sft_checkpoint, map_location="cpu", weights_only=True)
    legacy_checkpoint["format"] = "qmini-addsub-sft-v6"
    legacy_checkpoint.pop("curriculum")
    legacy_sft_path = legacy_sft_dir / sft_checkpoint.name
    torch.save(legacy_checkpoint, legacy_sft_path)
    shutil.copy2(tmp_path / "sft" / "sft_metrics.jsonl", legacy_sft_dir / "sft_metrics.jsonl")
    legacy_resumed = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=legacy_sft_dir,
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
        resume=legacy_sft_path,
    )
    assert legacy_resumed["steps"] == 2
    legacy_final = torch.load(
        legacy_sft_dir / "sft-checkpoint-step-000000002.pt",
        map_location="cpu",
        weights_only=True,
    )
    for name, tensor in full_sft_state["model"].items():
        assert torch.equal(tensor, legacy_final["model"][name]), name

    legacy_curriculum_dir = tmp_path / "sft-legacy-curriculum"
    legacy_curriculum_dir.mkdir()
    legacy_curriculum_checkpoint = legacy_curriculum_dir / sft_checkpoint.name
    torch.save(legacy_checkpoint, legacy_curriculum_checkpoint)
    shutil.copy2(tmp_path / "sft" / "sft_metrics.jsonl", legacy_curriculum_dir / "sft_metrics.jsonl")
    with pytest.raises(ValueError, match="Legacy v6 SFT checkpoints"):
        fine_tune_from_addsub_checkpoint(
            base_checkpoint=final_pretrain_checkpoint,
            tokenizer_path=tokenizer_path,
            train_jsonl=sft_train,
            valid_jsonl=sft_valid,
            test_jsonl=sft_test,
            output_dir=legacy_curriculum_dir,
            steps=2,
            batch_size=1,
            save_every=1,
            device_name="cpu",
            seed=43,
            resume=legacy_curriculum_checkpoint,
            curriculum_uniform_fraction=0.35,
            curriculum_warmup_steps=1,
        )

    extended_sft_dir = tmp_path / "sft-extended"
    extended_sft_dir.mkdir()
    step_two_sft = resumed_sft_dir / "sft-checkpoint-step-000000002.pt"
    shutil.copy2(step_two_sft, extended_sft_dir / step_two_sft.name)
    shutil.copy2(
        resumed_sft_dir / "sft_metrics.jsonl",
        extended_sft_dir / "sft_metrics.jsonl",
    )
    extended_sft_args = {
        "base_checkpoint": final_pretrain_checkpoint,
        "tokenizer_path": tokenizer_path,
        "train_jsonl": sft_train,
        "valid_jsonl": sft_valid,
        "test_jsonl": sft_test,
        "output_dir": extended_sft_dir,
        "steps": 3,
        "batch_size": 1,
        "save_every": 1,
        "device_name": "cpu",
        "seed": 43,
        "resume": extended_sft_dir / step_two_sft.name,
    }
    with pytest.raises(ValueError, match="extend_schedule=True"):
        fine_tune_from_addsub_checkpoint(**extended_sft_args)
    extended_sft_result = fine_tune_from_addsub_checkpoint(
        **extended_sft_args, extend_schedule=True
    )
    assert extended_sft_result["schedule_total_steps"] == 3
    extended_sft_state = torch.load(
        extended_sft_dir / "sft-checkpoint-step-000000003.pt",
        map_location="cpu", weights_only=True,
    )
    assert extended_sft_state["step"] == 3
    assert extended_sft_state["schedule_total_steps"] == 3
    extended_sft_metrics = [
        json.loads(line) for line in
        (extended_sft_dir / "sft_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["step"] for row in extended_sft_metrics] == [2, 3]
    assert [row["schedule_total_steps"] for row in extended_sft_metrics] == [2, 3]

    # Curriculum is opt-in. Its record-level response-loss state survives resume
    # and the explicit cosine schedule extension path.
    curriculum_dir = tmp_path / "sft-curriculum"
    curriculum_result = fine_tune_from_addsub_checkpoint(
        base_checkpoint=final_pretrain_checkpoint,
        tokenizer_path=tokenizer_path,
        train_jsonl=sft_train,
        valid_jsonl=sft_valid,
        test_jsonl=sft_test,
        output_dir=curriculum_dir,
        steps=2,
        batch_size=1,
        save_every=1,
        device_name="cpu",
        seed=43,
        curriculum_uniform_fraction=0.35,
        curriculum_warmup_steps=1,
    )
    assert curriculum_result["curriculum"]["enabled"] is True
    assert curriculum_result["curriculum"]["seen_records"] > 0
    curriculum_step2 = curriculum_dir / "sft-checkpoint-step-000000002.pt"
    curriculum_state = torch.load(curriculum_step2, map_location="cpu", weights_only=True)
    assert curriculum_state["curriculum"]["step"] == 2
    assert any(value is not None for value in curriculum_state["curriculum"]["ema_loss"])

    curriculum_resume_args = {
        "base_checkpoint": final_pretrain_checkpoint,
        "tokenizer_path": tokenizer_path,
        "train_jsonl": sft_train,
        "valid_jsonl": sft_valid,
        "test_jsonl": sft_test,
        "output_dir": curriculum_dir,
        "steps": 3,
        "batch_size": 1,
        "save_every": 1,
        "device_name": "cpu",
        "seed": 43,
        "resume": curriculum_step2,
        "curriculum_uniform_fraction": 0.35,
        "curriculum_warmup_steps": 1,
    }
    with pytest.raises(ValueError, match="extend_schedule=True"):
        fine_tune_from_addsub_checkpoint(**curriculum_resume_args)
    curriculum_extended = fine_tune_from_addsub_checkpoint(
        **curriculum_resume_args, extend_schedule=True
    )
    assert curriculum_extended["schedule_total_steps"] == 3
    curriculum_step3 = torch.load(
        curriculum_dir / "sft-checkpoint-step-000000003.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert curriculum_step3["curriculum"]["step"] == 3
    assert len(curriculum_step3["curriculum"]["ema_loss"]) == curriculum_result["train_records"]


def test_addsub_repetition_penalty_applies_before_greedy_choice():
    from qmini_addsub.generate import _apply_repetition_penalty, _choose_next_token
    logits=torch.tensor([4.0,-4.0,3.5])
    adjusted=_apply_repetition_penalty(logits,[0,1],penalty=1.2,window=8)
    assert adjusted[0]<logits[0] and adjusted[1]<logits[1]
    assert adjusted[2]==logits[2]
    assert _choose_next_token(logits,0.7,0,1.0,True,generated=[0],repetition_penalty=1.2)==2


@pytest.mark.parametrize(
    "legacy_format",
    [
        "qmini-addsub-training-v4",
        "qmini-addsub-sft-v5",
        "qmini-addsub-training-v3",
        "qmini-addsub-sft-v4",
        "qmini-addsub-sft-v3",
        "qmini-addsub-sft-v2",
    ],
)
def test_legacy_unscaled_checkpoints_keep_unit_quantization_scale(tmp_path, legacy_format):
    from qmini_addsub.generate import load_addsub_model
    from qmini_addsub.quantization import quantize_weight_ste, weight_scale

    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=813,
    )
    torch.manual_seed(813)
    original = AddSubLanguageModel(config)
    checkpoint_path = tmp_path / f"{legacy_format}.pt"
    torch.save(
        {
            "format": legacy_format,
            "step": 10,
            "config": config.to_dict(),
            "model": original.state_dict(),
        },
        checkpoint_path,
    )

    loaded = load_addsub_model(checkpoint_path, device_name="cpu")
    matrix_parameters = [
        parameter for parameter in loaded.parameters() if parameter.ndim >= 2
    ]
    assert matrix_parameters
    for parameter in matrix_parameters:
        assert float(weight_scale(parameter).item()) == pytest.approx(1.0)

        # This is the legacy unscaled five-symbol threshold rule.
        values = parameter.detach()
        expected = torch.where(
            values <= -1.0,
            -1.5,
            torch.where(
                values <= -0.25,
                -0.5,
                torch.where(
                    values <= 0.25,
                    0.0,
                    torch.where(values <= 1.0, 0.5, 1.5),
                ),
            ),
        )
        torch.testing.assert_close(
            quantize_weight_ste(parameter).detach(),
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_default_addsub_model_parameter_count_is_stable():
    model = AddSubLanguageModel(AddSubConfig())
    assert model.parameter_count() == 760340


def test_default_matrix_weight_decay_is_small():
    model = torch.nn.Linear(8, 8)
    optimizer = build_adamw(model, learning_rate=1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.01)


def test_swglu_experts_and_bounded_residual_gates_are_active():
    config = AddSubConfig()
    model = AddSubLanguageModel(config)
    block = model.blocks[0]
    for gate in (
        block.conv_residual_logit,
        block.state_residual_logit,
        block.expert_residual_logit,
    ):
        assert torch.sigmoid(gate).item() == pytest.approx(0.1, abs=1e-6)
    for expert in block.experts.experts:
        assert expert.up_gate.out_features == config.d_hidden * 2
        assert expert.down.in_features == config.d_hidden
    inputs = torch.randint(0, config.vocab_size, (2, 32))
    output = model(inputs)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["aux_loss"])


def test_optimizer_uses_no_decay_for_vectors_and_cosine_warmup_is_bounded():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.LayerNorm(8),
    )
    optimizer = build_adamw(model, learning_rate=1e-3, weight_decay=0.1)
    decay_ids = {id(parameter) for parameter in optimizer.param_groups[0]["params"]}
    no_decay_ids = {id(parameter) for parameter in optimizer.param_groups[1]["params"]}
    assert id(model[0].weight) in decay_ids
    assert id(model[0].bias) in no_decay_ids
    assert id(model[1].weight) in no_decay_ids
    assert id(model[1].bias) in no_decay_ids
    assert optimizer.param_groups[1]["weight_decay"] == 0.0

    assert cosine_warmup_lr(1, 100, 1e-3) < 1e-3
    assert cosine_warmup_lr(3, 100, 1e-3) == 1e-3
    assert cosine_warmup_lr(100, 100, 1e-3) == 1e-4


def test_default_model_initialization_is_dense_and_five_symbol_aligned():
    torch.manual_seed(29)
    model = AddSubLanguageModel(AddSubConfig())
    zero_decay = {id(block.state_space.decay_projection) for block in model.blocks}
    codebook = torch.tensor([-1.5,-0.5,0.0,0.5,1.5])
    linear_count = 0
    for module in model.modules():
        if not isinstance(module,AddSubLinear): continue
        linear_count += 1
        quantized = quantize_weight_ste(module.weight).detach()
        if id(module) in zero_decay:
            assert torch.count_nonzero(quantized)==0
            continue
        normalized=quantized/weight_scale(module.weight)
        assert torch.isclose(normalized.unsqueeze(-1),codebook,atol=1e-5,rtol=0.0).any(dim=-1).all()
        assert torch.count_nonzero(normalized).item()/normalized.numel()>0.5
    embedding=quantize_weight_ste(model.token_embedding.weight).detach()
    embedding=embedding/weight_scale(model.token_embedding.weight)
    assert torch.isclose(embedding.unsqueeze(-1),codebook,atol=1e-5,rtol=0.0).any(dim=-1).all()
    assert torch.count_nonzero(embedding).item()/embedding.numel()>0.5
    for block in model.blocks:
        depthwise=quantize_weight_ste(block.conv.depthwise_weight).detach()
        depthwise=depthwise/weight_scale(block.conv.depthwise_weight)
        assert torch.isclose(depthwise.unsqueeze(-1),codebook,atol=1e-5,rtol=0.0).any(dim=-1).all()
        assert torch.count_nonzero(depthwise).item()/depthwise.numel()>0.5
    assert linear_count>0


def test_new_model_rejects_legacy_context_overflow():
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        kernel_size=3,
        max_context=8,
    )
    model = AddSubLanguageModel(config)
    with pytest.raises(ValueError, match="maximum context"):
        model(torch.ones((1, 9), dtype=torch.long))


def test_training_text_cleaner_normalizes_wikitext_artifacts():
    cleaned, reason = normalize_training_text("A test , with @-@ marks .")
    assert reason == "accepted"
    assert cleaned == "A test, with - marks."


def test_training_text_cleaner_normalizes_unicode_controls_and_whitespace():
    cleaned, reason = normalize_training_text("Cafe\u0301\r\nhello\tworld\x00\n\n\nnext")
    assert reason == "accepted"
    assert cleaned == "Café\nhello world\n\nnext"


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("Contact me at person@example.com", "email"),
        ("Use ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456", "credential"),
        ("Call me at +1 (415) 555-0134", "phone_contact"),
        ("\x00\x01", "too_short"),
        ("123456", "no_letters"),
        ("x" * 32769, "too_long"),
    ],
)
def test_training_text_cleaner_rejects_sensitive_or_low_quality_fields(value, reason):
    _, actual_reason = normalize_training_text(value)
    assert actual_reason == reason


def test_oasst_rejects_missing_or_invalid_tree_identity():
    base = {
        "message_id": "m1",
        "parent_id": None,
        "text": "A reviewed English message.",
        "role": "prompter",
        "lang": "en",
        "review_result": True,
        "deleted": False,
        "rank": None,
        "synthetic": False,
        "message_tree_id": "tree-1",
        "tree_state": "ready_for_export",
    }
    invalid = dict(base, message_tree_id="")
    assert accept_oasst1_message(invalid) == (False, "invalid_tree_id")


def test_incremental_cached_forward_matches_full_sequence_logits():
    torch.manual_seed(17)
    model = AddSubLanguageModel(AddSubConfig(dropout=0.0)).eval()
    token_ids = torch.randint(0, model.config.vocab_size, (2, 19))
    with torch.inference_mode():
        expected = model(token_ids)["logits"]
        caches = model.init_cache(batch_size=token_ids.shape[0])
        actual_steps = []
        for position in range(token_ids.shape[1]):
            logits, caches = model.forward_step(token_ids[:, position], caches)
            actual_steps.append(logits)
        actual = torch.stack(actual_steps, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_additive_runtime_rejects_negative_weight_codes():
    with pytest.raises(ValueError, match="integers in"):
        additive_projection(
            np.asarray([[1, -1]], dtype=np.int8),
            np.asarray([[-1, 2]], dtype=np.int8),
        )


def test_byte_bpe_config_requires_bytes_and_all_control_tokens():
    with pytest.raises(ValueError, match="256 byte symbols"):
        AddSubConfig(vocab_size=263).validate()



def test_user_dataset_importer_accepts_selected_local_splits_without_download(tmp_path):
    import json

    from scripts.import_qmini_addsub_dataset import import_user_dataset

    pretrain_paths = {}
    for split, content in (
        ("train", "  English train text.  \r\nContact phone: 555 12345678\r\nSecond safe line.\r\n"),
        ("validation", "English validation text.\n"),
        ("test", "English held-out test text.\n"),
    ):
        path = tmp_path / f"{split}.txt"
        path.write_text(content, encoding="utf-8", newline="")
        pretrain_paths[split] = path

    sft_train = tmp_path / "chosen_sft_train.jsonl"
    sft_validation = tmp_path / "chosen_sft_valid.jsonl"
    sft_test = tmp_path / "chosen_sft_test.jsonl"
    records = []
    for split, path in (("train", sft_train), ("validation", sft_validation), ("test", sft_test)):
        record = {
            "human_authored": True,
            "synthetic": False,
            "messages": [
                {"role": "user", "content": f"What is {split}?"},
                {"role": "assistant", "content": f"This is {split}."},
            ],
        }
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        records.append(path)

    output = tmp_path / "imported"
    manifest = import_user_dataset(
        pretrain_train=pretrain_paths["train"],
        pretrain_validation=pretrain_paths["validation"],
        pretrain_test=pretrain_paths["test"],
        output_dir=output,
        pretrain_source="user-selected-English-corpus",
        pretrain_license="user-reviewed-license",
        confirm_rights_and_human_data=True,
        confirm_english_data=True,
        sft_train=sft_train,
        sft_validation=sft_validation,
        sft_test=sft_test,
        sft_source="user-selected-assistant-corpus",
        sft_license="user-reviewed-instruction-license",
    )
    assert manifest["synthetic_training_records_included"] == 0
    assert manifest["pretraining"]["splits"]["train"]["sha256"]
    assert manifest["pretraining"]["splits"]["train"]["source_sha256"]
    assert manifest["pretraining"]["splits"]["train"]["rejected_line_counts"]["phone_contact"] == 1
    assert "555 12345678" not in (output / "pretrain_train.txt").read_text(encoding="utf-8")
    assert "Second safe line." in (output / "pretrain_train.txt").read_text(encoding="utf-8")
    assert manifest["approved_instruction_sources"] == ["user-selected-assistant-corpus"]
    imported_record = json.loads((output / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert imported_record["source"] == "user-selected-assistant-corpus"
    assert imported_record["human_authored"] is True
    assert imported_record["synthetic"] is False


def test_sft_loader_accepts_only_manifest_approved_custom_sources(tmp_path):
    import json

    path = tmp_path / "custom_sft.jsonl"
    record = {
        "source": "user-selected-corpus",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "Explain a test."},
            {"role": "assistant", "content": "A test checks expected behaviour."},
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unapproved"):
        _load_conversations(path)
    assert len(_load_conversations(path, {"user-selected-corpus"})) == 1


def test_user_dataset_importer_requires_explicit_rights_and_non_synthetic_confirmation(tmp_path):
    from scripts.import_qmini_addsub_dataset import import_user_dataset

    paths = []
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.txt"
        path.write_text("English text.\n", encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="Explicit confirmation"):
        import_user_dataset(
            pretrain_train=paths[0],
            pretrain_validation=paths[1],
            pretrain_test=paths[2],
            output_dir=tmp_path / "out",
            pretrain_source="test-corpus",
            pretrain_license="test-license",
        )



def test_user_dataset_importer_requires_explicit_english_confirmation(tmp_path):
    from scripts.import_qmini_addsub_dataset import import_user_dataset

    paths = []
    for name in ("train", "validation", "test"):
        path = tmp_path / f"{name}.txt"
        path.write_text("English text.\n", encoding="utf-8")
        paths.append(path)
    with pytest.raises(ValueError, match="English"):
        import_user_dataset(
            pretrain_train=paths[0],
            pretrain_validation=paths[1],
            pretrain_test=paths[2],
            output_dir=tmp_path / "out",
            pretrain_source="test-corpus",
            pretrain_license="test-license",
            confirm_rights_and_human_data=True,
        )



def test_byte_bpe_config_rejects_vocab_larger_than_uint16_stream_capacity():
    with pytest.raises(ValueError, match="between 264 and 65536"):
        AddSubConfig(vocab_size=65537).validate()



def test_moe_auxiliary_loss_ignores_right_padding_tokens():
    torch.manual_seed(211)
    model = AddSubLanguageModel(
        AddSubConfig(
            vocab_size=300,
            d_model=32,
            d_hidden=48,
            n_layers=2,
            n_experts=2,
            kernel_size=3,
            max_context=32,
            dropout=0.0,
        )
    ).eval()
    inputs = torch.randint(0, model.config.vocab_size, (1, 17))
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    mask[:, :9] = True
    with torch.inference_mode():
        padded = model(inputs, attention_mask=mask)
        unpadded = model(inputs[:, :9])
    torch.testing.assert_close(
        padded["logits"][:, :9],
        unpadded["logits"],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        padded["aux_loss"],
        unpadded["aux_loss"],
        rtol=1e-6,
        atol=1e-6,
    )


def test_model_rejects_empty_inputs_and_invalid_attention_masks():
    model = AddSubLanguageModel(
        AddSubConfig(
            vocab_size=300,
            d_model=32,
            d_hidden=48,
            n_layers=1,
            n_experts=2,
            kernel_size=3,
            max_context=32,
            dropout=0.0,
        )
    )
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.empty((1, 0), dtype=torch.long))
    inputs = torch.ones((1, 4), dtype=torch.long)
    with pytest.raises(ValueError, match="at least one valid token"):
        model(inputs, attention_mask=torch.zeros_like(inputs, dtype=torch.bool))
    with pytest.raises(ValueError, match="match input_ids"):
        model(inputs, attention_mask=torch.ones((1, 3), dtype=torch.bool))


def test_prompt_builder_rejects_a_user_message_that_would_be_fully_trimmed():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from qmini_addsub.generate import build_chat_prompt

    vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    with pytest.raises(ValueError, match="current user message is too long"):
        build_chat_prompt(
            tokenizer,
            [{"role": "user", "content": "question"}],
            max_context=8,
            system_text="a very long system prompt with many words",
        )



def test_saturating_integer_add_rejects_float_inputs_and_clamps():
    left = np.asarray([-120, -5, 0, 120], dtype=np.int16)
    right = np.asarray([-20, 2, 5, 20], dtype=np.int16)
    assert saturating_add(left, right).tolist() == [-127, -3, 5, 127]
    with pytest.raises(TypeError, match="integer arrays"):
        saturating_add(np.asarray([1.5]), np.asarray([2], dtype=np.int8))
    with pytest.raises(ValueError, match="identical shapes"):
        saturating_add(np.asarray([1], dtype=np.int8), np.asarray([[1]], dtype=np.int8))


def test_additive_projection_exhausts_all_single_activation_weight_pairs():
    activations = np.arange(-127, 128, dtype=np.int16).reshape(-1, 1)
    codes = np.arange(5, dtype=np.uint8).reshape(5, 1)
    result = additive_projection(activations, codes)
    expected = np.empty((255, 5), dtype=np.int64)
    for row, value in enumerate(range(-127, 128)):
        expected[row, 0] = -(value + value + value)
        expected[row, 1] = -value
        expected[row, 2] = 0
        expected[row, 3] = value
        expected[row, 4] = value + value + value
    assert np.array_equal(result, expected)



def test_model_seed_is_deterministic_and_does_not_advance_global_rng():
    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=911,
    )
    torch.manual_seed(77)
    before = torch.get_rng_state().clone()
    first = AddSubLanguageModel(config)
    after = torch.get_rng_state()
    second = AddSubLanguageModel(config)
    assert torch.equal(before, after)
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name]), name

    other = AddSubLanguageModel(AddSubConfig.from_dict({**config.to_dict(), "seed": 912}))
    assert any(
        not torch.equal(tensor, other.state_dict()[name])
        for name, tensor in first.state_dict().items()
    )



def test_finite_training_state_guard_reports_corrupted_parameters_and_moments():
    model = torch.nn.Linear(4, 3)
    optimizer = build_adamw(model, learning_rate=1e-3)
    loss = model(torch.ones(2, 4)).square().mean()
    loss.backward()
    optimizer.step()
    assert_finite_training_state(model, optimizer, step=1)

    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="model parameter"):
        assert_finite_training_state(model, optimizer, step=2)

    with torch.no_grad():
        model.weight[0, 0] = 0.0
        optimizer.state[model.weight]["exp_avg"][0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="optimizer state"):
        assert_finite_training_state(model, optimizer, step=3)



def test_sft_context_trims_old_user_prompt_but_preserves_complete_assistant_answer():
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
        "complete": 8,
        "answer": 9,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    dataset = ConversationDataset(
        [{"messages": [
            {"role": "user", "content": "question " * 80},
            {"role": "assistant", "content": "complete answer"},
        ]}],
        tokenizer,
        max_context=64,
    )
    token_ids, labels = dataset[0]
    assert len(token_ids) <= 64
    assistant_position = token_ids.index(vocab["<|assistant|>"])
    assert labels[assistant_position + 1:assistant_position + 4] == [
        vocab["complete"],
        vocab["answer"],
        vocab["<|end|>"],
    ]


def test_sft_drops_conversations_when_a_complete_answer_cannot_fit():
    vocab = {
        "<unk>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<pad>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|end|>": 7,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    with pytest.raises(ValueError, match="No conversations fit"):
        ConversationDataset(
            [{"messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer " * 100},
            ]}],
            tokenizer,
            max_context=32,
        )



def test_generation_uses_exact_same_system_prompt_as_sft_training():
    from qmini_addsub.generate import SYSTEM_TEXT as GENERATION_SYSTEM_TEXT
    from qmini_addsub.sft import SYSTEM_TEXT as TRAINING_SYSTEM_TEXT

    assert GENERATION_SYSTEM_TEXT == TRAINING_SYSTEM_TEXT



def test_custom_dataset_import_is_transactional_on_sft_leakage(tmp_path):
    import json

    from scripts.import_qmini_addsub_dataset import import_user_dataset

    pretrain = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"pretrain_{split}.txt"
        path.write_text(f"English {split} material.\n", encoding="utf-8")
        pretrain.append(path)

    sft_paths = [tmp_path / f"sft_{split}.jsonl" for split in ("train", "validation", "test")]
    duplicate = {
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "What is a model?"},
            {"role": "assistant", "content": "A model maps inputs to outputs."},
        ],
    }
    distinct = {
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "What is a tokenizer?"},
            {"role": "assistant", "content": "A tokenizer maps text to token IDs."},
        ],
    }
    sft_paths[0].write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    sft_paths[1].write_text(json.dumps(duplicate) + "\n", encoding="utf-8")
    sft_paths[2].write_text(json.dumps(distinct) + "\n", encoding="utf-8")

    output = tmp_path / "transactional-output"
    with pytest.raises(ValueError, match="leakage"):
        import_user_dataset(
            pretrain_train=pretrain[0],
            pretrain_validation=pretrain[1],
            pretrain_test=pretrain[2],
            output_dir=output,
            pretrain_source="chosen-English-corpus",
            pretrain_license="reviewed-license",
            confirm_rights_and_human_data=True,
            confirm_english_data=True,
            sft_train=sft_paths[0],
            sft_validation=sft_paths[1],
            sft_test=sft_paths[2],
            sft_source="chosen-assistant-corpus",
            sft_license="reviewed-instruction-license",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".transactional-output.staging-*"))


def test_custom_dataset_import_refuses_overwrite_without_explicit_flag(tmp_path):
    import json

    from scripts.import_qmini_addsub_dataset import import_user_dataset

    pretrain = []
    for split in ("train", "validation", "test"):
        path = tmp_path / f"{split}.txt"
        path.write_text(f"English {split} material.\n", encoding="utf-8")
        pretrain.append(path)

    output = tmp_path / "import-once"
    kwargs = {
        "pretrain_train": pretrain[0],
        "pretrain_validation": pretrain[1],
        "pretrain_test": pretrain[2],
        "output_dir": output,
        "pretrain_source": "chosen-English-corpus",
        "pretrain_license": "reviewed-license",
        "confirm_rights_and_human_data": True,
        "confirm_english_data": True,
    }
    first = import_user_dataset(**kwargs)
    before = (output / "dataset_manifest.json").read_bytes()
    assert first["format"] == "qmini-addsub-data-v1"
    with pytest.raises(FileExistsError, match="already contains prepared files"):
        import_user_dataset(**kwargs)
    assert (output / "dataset_manifest.json").read_bytes() == before



def test_packed_addsub_export_round_trip_reconstructs_complete_reference_model(tmp_path):
    from qmini_addsub.export import export_addsub_checkpoint, load_addsub_export
    from qmini_addsub.train import CHECKPOINT_FORMAT

    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=1234,
    )
    import hashlib

    original = AddSubLanguageModel(config).eval()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text('{"test": "tokenizer"}', encoding="utf-8")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    checkpoint_path = tmp_path / "trained-addsub.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "step": 1,
            "config": config.to_dict(),
            "model": original.state_dict(),
            "tokenizer_sha256": tokenizer_sha,
            "data_manifest_sha256": "b" * 64,
            "run_contract": {"planned_steps": 2, "seed": config.seed},
        },
        checkpoint_path,
    )

    export_path = tmp_path / "model.qma"
    export_result = export_addsub_checkpoint(checkpoint_path, export_path)
    restored, manifest = load_addsub_export(
        export_path, tokenizer_path=tokenizer_path
    )
    wrong_tokenizer = tmp_path / "wrong-tokenizer.json"
    wrong_tokenizer.write_text('{"test": "different"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Tokenizer fingerprint"):
        load_addsub_export(export_path, tokenizer_path=wrong_tokenizer)
    assert manifest["format"] == "qmini-addsub-packed-weights-v2"
    assert export_result["parameter_count"] == original.parameter_count()
    assert export_result["quantized_tensors"] > 0
    assert export_result["auxiliary_tensors"] > 0

    input_ids = torch.randint(0, config.vocab_size, (2, 13))
    with torch.inference_mode():
        expected = original(input_ids)["logits"]
        actual = restored(input_ids)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_packed_addsub_export_rejects_tampered_weight_payload(tmp_path):
    import zipfile

    from qmini_addsub.export import export_addsub_checkpoint, load_addsub_export
    from qmini_addsub.train import CHECKPOINT_FORMAT

    config = AddSubConfig(
        vocab_size=300,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
        seed=1235,
    )
    model = AddSubLanguageModel(config)
    checkpoint = tmp_path / "base.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "step": 1,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "tokenizer_sha256": "c" * 64,
            "data_manifest_sha256": "d" * 64,
            "run_contract": {"planned_steps": 2, "seed": config.seed},
        },
        checkpoint,
    )
    valid_export = tmp_path / "valid.qma"
    tampered_export = tmp_path / "tampered.qma"
    export_addsub_checkpoint(checkpoint, valid_export)

    with zipfile.ZipFile(valid_export, "r") as source:
        members = {name: source.read(name) for name in source.namelist()}
    damaged = bytearray(members["weights.bin"])
    damaged[0] ^= 1
    members["weights.bin"] = bytes(damaged)
    with zipfile.ZipFile(tampered_export, "w") as target:
        for name, payload in members.items():
            target.writestr(name, payload)

    with pytest.raises(ValueError, match="SHA-256 verification"):
        load_addsub_export(tampered_export)



def test_token_stream_encoder_escapes_control_literals_and_counts_encoded_bytes(tmp_path):
    import hashlib

    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from qmini_addsub.data import escape_special_token_literals
    from qmini_addsub.train import encode_text_file

    vocab = {
        "<unk>": 0,
        "<eos>": 1,
        "<|user|>": 2,
        "<": 3,
        "\u200b": 4,
        "|": 5,
        "user": 6,
        ">": 7,
        "hello": 8,
        "world": 9,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([AddedToken("<|user|>", special=True)])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    source = tmp_path / "input.txt"
    source.write_text("<|user|> hello\n\nworld\n", encoding="utf-8")
    output = tmp_path / "tokens.u16"
    count, digest, source_bytes = encode_text_file(source, tokenizer_path, output)

    safe_first_line = escape_special_token_literals("<|user|> hello")
    expected_first_ids = tokenizer.encode(safe_first_line, add_special_tokens=False).ids
    expected_ids = np.asarray(
        expected_first_ids + [tokenizer.token_to_id("<eos>")]
        + [tokenizer.token_to_id("world"), tokenizer.token_to_id("<eos>")],
        dtype=np.uint16,
    )
    actual_ids = np.fromfile(output, dtype=np.uint16)
    assert count == expected_ids.size
    assert np.array_equal(actual_ids, expected_ids)
    assert tokenizer.token_to_id("<|user|>") not in actual_ids
    expected_text = safe_first_line + "world"
    assert source_bytes == len(expected_text.encode("utf-8"))
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()


def test_tokenizer_builder_rejects_sft_manifest_with_missing_training_file(tmp_path):
    import json
    import subprocess
    import sys

    from qmini_addsub.train import sha256_file

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_text = data_dir / "pretrain_train.txt"
    train_text.write_text("English source text.\n", encoding="utf-8")
    manifest = {
        "format": "qmini-addsub-data-v1",
        "language": "English",
        "synthetic_training_records_included": 0,
        "pretraining": {"splits": {"train": {"sha256": sha256_file(train_text)}}},
        "instruction_splits": {
            "train": {"path": "sft_train.jsonl", "sha256": "0" * 64},
            "validation": {"path": "sft_validation.jsonl", "sha256": "1" * 64},
            "test": {"path": "sft_test.jsonl", "sha256": "2" * 64},
        },
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_qmini_addsub_tokenizer.py"
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--data-dir", str(data_dir),
            "--output", str(tmp_path / "artifacts" / "tokenizer.json"),
            "--vocab-size", "264",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "SFT training split" in result.stderr



def test_heldout_evaluation_lexical_metrics_are_deterministic_and_transparent():
    from qmini_addsub.evaluate import lexical_scores

    exact = lexical_scores("The quick brown fox.", "the quick brown fox")
    assert exact["exact_match"] == 1.0
    assert exact["token_precision"] == 1.0
    assert exact["token_recall"] == 1.0
    assert exact["token_f1"] == 1.0

    partial = lexical_scores("alpha beta gamma", "alpha beta delta delta")
    assert partial["exact_match"] == 0.0
    assert partial["token_precision"] == 0.5
    assert partial["token_recall"] == 2.0 / 3.0
    assert 0.0 < partial["token_f1"] < 1.0

    empty = lexical_scores("", "")
    assert empty["exact_match"] == 1.0
    assert empty["token_f1"] == 0.0


def test_heldout_evaluation_refuses_validation_data_and_wrong_tokenizer_hash(tmp_path):
    import json
    from qmini_addsub.evaluate import evaluate_heldout_conversations

    checkpoint = tmp_path / "checkpoint.pt"
    tokenizer = tmp_path / "tokenizer.json"
    validation = tmp_path / "sft_validation.jsonl"
    manifest = tmp_path / "dataset_manifest.json"
    checkpoint.write_bytes(b"placeholder")
    tokenizer.write_text("{}", encoding="utf-8")
    validation.write_text("{}", encoding="utf-8")
    import hashlib

    validation_sha = hashlib.sha256(validation.read_bytes()).hexdigest()
    tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps({
            "format": "qmini-addsub-data-v1",
            "instruction_splits": {
                "test": {"path": "sft_test.jsonl", "sha256": "0" * 64},
            },
            "tokenizer": {"sha256": tokenizer_sha},
            "approved_instruction_sources": ["approved-source"],
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out SFT test split"):
        evaluate_heldout_conversations(
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            test_jsonl=validation,
            dataset_manifest_path=manifest,
            output_dir=tmp_path / "eval",
        )



def test_model_attention_mask_requires_right_padding_without_internal_gaps():
    model = AddSubLanguageModel(
        AddSubConfig(
            vocab_size=300,
            d_model=32,
            d_hidden=48,
            n_layers=1,
            n_experts=2,
            kernel_size=3,
            max_context=32,
            dropout=0.0,
        )
    ).eval()
    inputs = torch.ones((1, 4), dtype=torch.long)
    mask = torch.tensor([[True, False, True, False]])
    with pytest.raises(ValueError, match="right-padded"):
        model(inputs, attention_mask=mask)


def test_addsub_config_requires_context_room_for_chat_controls():
    with pytest.raises(ValueError, match="at least 8"):
        AddSubConfig(max_context=7).validate()



def test_additive_projection_rejects_bias_accumulator_overflow():
    activations = np.asarray([[3, -4]], dtype=np.int8)
    weights = np.asarray([[4, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match="signed int32"):
        additive_projection(
            activations,
            weights,
            bias_accumulator=np.asarray([2**40], dtype=np.int64),
        )



def test_metrics_reconciliation_discards_only_uncommitted_future_rows(tmp_path):
    import json

    metrics = tmp_path / "metrics.jsonl"
    identity = {
        "train_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "run_contract": {"planned_steps": 200, "seed": 99},
    }
    rows = [
        {"step": 50, **identity},
        {"step": 100, **identity},
        {"step": 150, **identity},
    ]
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    discarded = reconcile_metrics_file(
        str(metrics), identity, maximum_committed_step=100, run_name="Pretraining"
    )
    remaining = [
        json.loads(line)
        for line in metrics.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert discarded == 1
    assert [row["step"] for row in remaining] == [50, 100]
    assert not metrics.with_suffix(".jsonl.tmp").exists()


def test_metrics_reconciliation_rejects_a_different_dataset_or_schedule(tmp_path):
    import json

    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({
            "step": 50,
            "train_sha256": "a" * 64,
            "run_contract": {"planned_steps": 100, "seed": 1},
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="different dataset"):
        reconcile_metrics_file(
            str(metrics),
            {
                "train_sha256": "c" * 64,
                "run_contract": {"planned_steps": 100, "seed": 1},
            },
            maximum_committed_step=40,
            run_name="SFT",
        )
    assert json.loads(metrics.read_text(encoding="utf-8").splitlines()[0])["step"] == 50



def test_custom_dataset_import_rejects_identical_normalized_pretraining_splits(tmp_path):
    from scripts.import_qmini_addsub_dataset import import_user_dataset

    train = tmp_path / "train.txt"
    validation = tmp_path / "validation.txt"
    test = tmp_path / "test.txt"
    train.write_text("Same English source document.\n", encoding="utf-8")
    validation.write_text("  Same English source document.  \n", encoding="utf-8")
    test.write_text("Different held-out English document.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identical normalized split files"):
        import_user_dataset(
            pretrain_train=train,
            pretrain_validation=validation,
            pretrain_test=test,
            output_dir=tmp_path / "dedupe-output",
            pretrain_source="selected-corpus",
            pretrain_license="reviewed-license",
            confirm_rights_and_human_data=True,
            confirm_english_data=True,
        )
    assert not (tmp_path / "dedupe-output").exists()



def test_heldout_conversation_evaluator_runs_with_manifest_verified_checkpoint(tmp_path):
    import hashlib
    import json

    from tokenizers import AddedToken, Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from qmini_addsub.evaluate import evaluate_heldout_conversations
    from qmini_addsub.train import CHECKPOINT_FORMAT

    special = [
        "<unk>", "<bos>", "<eos>", "<pad>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
    ]
    vocab = {token: index for index, token in enumerate(special)}
    vocab.update({"Hello": 8, "there": 9, "friend": 10})
    vocab.update({f"unused_{index}": index for index in range(11, 264)})
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([AddedToken(token, special=True) for token in special[1:]])
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    config = AddSubConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=64,
        dropout=0.0,
        seed=775,
    )
    model = AddSubLanguageModel(config).eval()
    checkpoint_path = tmp_path / "trained.pt"
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "step": 1,
            "config": config.to_dict(),
            "model": model.state_dict(),
            "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        },
        checkpoint_path,
    )

    test_path = tmp_path / "sft_test.jsonl"
    record = {
        "source": "approved-source",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello there friend"},
        ],
    }
    test_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps({
            "format": "qmini-addsub-data-v1",
            "instruction_splits": {
                "test": {
                    "path": test_path.name,
                    "sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
                },
            },
            "tokenizer": {
                "sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
            },
            "approved_instruction_sources": ["approved-source"],
        }),
        encoding="utf-8",
    )

    output_dir = tmp_path / "evaluation"
    summary = evaluate_heldout_conversations(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        test_jsonl=test_path,
        dataset_manifest_path=manifest_path,
        output_dir=output_dir,
        device_name="cpu",
        max_new_tokens=3,
    )
    assert summary["test_records"] == 1
    assert summary["checkpoint_sha256"] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert summary["tokenizer_sha256"] == hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    assert (output_dir / "heldout_responses.jsonl").is_file()
    assert (output_dir / "heldout_summary.json").is_file()
    diversity = summary["output_diversity"]
    assert diversity["examples"] == summary["test_records"] == 1
    assert 1 <= diversity["unique_normalized_answers"] <= 1
    assert diversity["unique_answer_fraction"] == 1.0
    assert diversity["largest_identical_answer_count"] == 1
    assert diversity["all_outputs_identical"] is True
    persisted = json.loads((output_dir / "heldout_summary.json").read_text(encoding="utf-8"))
    assert persisted["output_diversity"] == diversity



def test_metrics_reconciliation_rejects_non_monotonic_or_invalid_steps(tmp_path):
    import json

    identity = {
        "train_sha256": "a" * 64,
        "run_contract": {"planned_steps": 100, "seed": 17},
    }
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        "".join(
            json.dumps({"step": step, **identity}) + "\n"
            for step in (20, 10)
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        reconcile_metrics_file(str(metrics), identity, 20, "Pretraining")

    metrics.write_text(
        json.dumps({"step": True, **identity}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid step"):
        reconcile_metrics_file(str(metrics), identity, 20, "SFT")

    metrics.write_text("{partial json\n", encoding="utf-8")
    before = metrics.read_bytes()
    with pytest.raises(ValueError, match="not valid JSONL"):
        reconcile_metrics_file(str(metrics), identity, 20, "Pretraining")
    assert metrics.read_bytes() == before


def test_addsub_v2_residual_gates_start_small_without_changing_parameter_budget():
    model = AddSubLanguageModel().eval()
    assert model.parameter_count() == 760340
    for block in model.blocks:
        for name in (
            "conv_residual_logit",
            "state_residual_logit",
            "expert_residual_logit",
        ):
            strength = torch.sigmoid(getattr(block, name)).item()
            assert strength == pytest.approx(0.1, abs=1e-6)


def test_addsub_v2_router_z_loss_is_finite_differentiable_and_reports_load():
    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=2,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    model = AddSubLanguageModel(config).train()
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    mask = torch.tensor(
        [[True, True, True, True, True, True, False, False],
         [True, True, True, True, True, False, False, False]]
    )
    result = model(tokens, attention_mask=mask)

    assert result["router_load"].shape == (config.n_layers, config.n_experts)
    assert not result["router_load"].requires_grad
    torch.testing.assert_close(
        result["router_load"].sum(dim=-1),
        torch.ones(config.n_layers),
    )
    assert torch.isfinite(result["router_z_loss"])
    loss = (
        result["logits"].float().square().mean()
        + config.moe_balance_coef * result["aux_loss"]
        + config.router_z_loss_coef * result["router_z_loss"]
    )
    loss.backward()
    for block in model.blocks:
        gradient = block.experts.router.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()


def test_addsub_v2_router_losses_ignore_right_padding_tokens():
    torch.manual_seed(2026)
    config = AddSubConfig(
        vocab_size=264,
        d_model=32,
        d_hidden=48,
        n_layers=1,
        n_experts=2,
        kernel_size=3,
        max_context=32,
        dropout=0.0,
    )
    experts = SparseTop1Experts(config).eval()
    valid = torch.randn(2, 3, config.d_model)
    padded = torch.cat(
        [valid, torch.randn(2, 4, config.d_model) * 100.0],
        dim=1,
    )
    mask = torch.tensor(
        [[True, True, True, False, False, False, False],
         [True, True, True, False, False, False, False]]
    )
    with torch.inference_mode():
        _, base_balance, base_z, base_load = experts(valid)
        _, pad_balance, pad_z, pad_load = experts(padded, token_mask=mask)
    torch.testing.assert_close(base_balance, pad_balance, rtol=0.0, atol=0.0)
    torch.testing.assert_close(base_z, pad_z, rtol=0.0, atol=0.0)
    torch.testing.assert_close(base_load, pad_load, rtol=0.0, atol=0.0)


def test_addsub_v2_router_and_residual_coefficients_are_validated():
    with pytest.raises(ValueError, match="residual_init"):
        AddSubConfig(residual_init=0.0).validate()
    with pytest.raises(ValueError, match="moe_balance_coef"):
        AddSubConfig(moe_balance_coef=-0.1).validate()
    with pytest.raises(ValueError, match="router_z_loss_coef"):
        AddSubConfig(router_z_loss_coef=float("inf")).validate()


def test_addsub_checkpoint_discovery_uses_numeric_order_and_rejects_ambiguous_names(tmp_path):
    from qmini_addsub.checkpoints import latest_checkpoint

    older = tmp_path / "checkpoint-step-9.pt"
    newer = tmp_path / "checkpoint-step-10.pt"
    older.touch()
    newer.touch()
    assert latest_checkpoint(tmp_path, "checkpoint") == (newer, 10)

    duplicate = tmp_path / "checkpoint-step-000000010.pt"
    duplicate.touch()
    with pytest.raises(ValueError, match="duplicate numeric checkpoint step"):
        latest_checkpoint(tmp_path, "checkpoint")


def test_addsub_checkpoint_discovery_fails_closed_on_malformed_or_metrics_only_output(tmp_path):
    from qmini_addsub.checkpoints import resolve_checkpoint

    malformed = tmp_path / "checkpoint-step-final.pt"
    malformed.touch()
    with pytest.raises(ValueError, match="malformed checkpoint filename"):
        resolve_checkpoint(tmp_path, "checkpoint", "metrics.jsonl")

    malformed.unlink()
    (tmp_path / "metrics.jsonl").write_text('{"step": 1}\\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metrics file exists but no checkpoint"):
        resolve_checkpoint(tmp_path, "checkpoint", "metrics.jsonl")


def test_addsub_checkpoint_resume_selects_latest_and_rejects_older_explicit_checkpoint(tmp_path):
    from qmini_addsub.checkpoints import resolve_checkpoint

    first = tmp_path / "checkpoint-step-000000001.pt"
    latest = tmp_path / "checkpoint-step-000000012.pt"
    first.touch()
    latest.touch()
    assert resolve_checkpoint(tmp_path, "checkpoint", "metrics.jsonl") == (latest, 12)
    with pytest.raises(ValueError, match="newer checkpoint"):
        resolve_checkpoint(tmp_path, "checkpoint", "metrics.jsonl", explicit=first)


def test_addsub_colab_notebook_uses_safe_checkpoint_resume():
    import json

    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "colab"
        / "QMini_AddSub_V1_From_Scratch.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "from qmini_addsub.checkpoints import resolve_checkpoint" in code
    assert "Automatically resuming pretraining" in code
    assert "Automatically resuming assistant SFT" in code
    assert "RUN_READINESS_GATE = False" in code
    assert "qmini_addsub.readiness" in code
    assert "'--checkpoint'" in code
    assert "metrics_filename='metrics.jsonl'" in code
    assert "sorted(PRETRAIN_DIR.glob" not in code
    assert "sorted(SFT_DIR.glob" not in code


def test_addsub_colab_keeps_cuda_visible_and_uses_the_actual_sft_metrics_filename():
    import json
    from pathlib import Path

    notebook_path = Path(__file__).resolve().parents[1] / "colab" / "QMini_AddSub_V1_From_Scratch.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    all_code = "\n".join(code)
    assert "REPO_BRANCH = 'main'" in code[0]
    assert "VERSION_ROOT = DRIVE_ROOT / 'trainability_addsub_main_v1'" in code[0]
    assert "CUDA_VISIBLE_DEVICES" not in all_code
    assert "os.environ" not in code[0]
    assert "if DEVICE.startswith('cuda') and not torch.cuda.is_available():" in all_code
    assert "metrics_path = SFT_DIR / 'sft_metrics.jsonl'" in all_code
    assert all_code.count("metrics_filename='sft_metrics.jsonl'") >= 4
    assert "metrics_path = SFT_DIR / 'metrics.jsonl'" not in all_code


def test_addsub_tokenizer_includes_approved_sft_training_messages(tmp_path):
    import io
    import json
    from scripts.train_qmini_addsub_tokenizer import append_sft_training_text

    sft_path = tmp_path / "sft_train.jsonl"
    record = {
        "source": "reviewed-source",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "How does a small computer store information?"},
            {"role": "assistant", "content": "It stores information as bits in memory."},
        ],
    }
    sft_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output = io.StringIO()
    stats = append_sft_training_text(output, sft_path, {"reviewed-source"})
    corpus = output.getvalue()
    assert stats["conversations"] == 1
    assert stats["messages"] == 2
    assert stats["bytes"] > 0
    assert "small computer store information" in corpus
    assert "bits in memory" in corpus
    assert "You are Q-Mini." in corpus


def test_addsub_tokenizer_rejects_unapproved_or_synthetic_sft_text(tmp_path):
    import io
    import json
    from scripts.train_qmini_addsub_tokenizer import append_sft_training_text

    sft_path = tmp_path / "sft_train.jsonl"
    record = {
        "source": "unapproved-source",
        "human_authored": True,
        "synthetic": False,
        "messages": [
            {"role": "user", "content": "A sufficiently clear user message."},
            {"role": "assistant", "content": "A sufficiently clear assistant response."},
        ],
    }
    sft_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not approved"):
        append_sft_training_text(io.StringIO(), sft_path, {"reviewed-source"})

    record["source"] = "reviewed-source"
    record["synthetic"] = True
    sft_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic or unverified"):
        append_sft_training_text(io.StringIO(), sft_path, {"reviewed-source"})


def test_addsub_checkpoint_resolver_uses_sft_metrics_filename_and_fails_closed(tmp_path):
    from qmini_addsub.checkpoints import resolve_checkpoint

    (tmp_path / "sft_metrics.jsonl").write_text('{"step": 10}\\n', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="metrics file exists but no checkpoint"):
        resolve_checkpoint(tmp_path, "sft-checkpoint", "sft_metrics.jsonl")



def test_metrics_reconciliation_accepts_explicit_horizon_growth(tmp_path):
    import json

    metrics = tmp_path / "metrics.jsonl"
    static_contract = {"batch_size": 1, "seed": 9}
    rows = [
        {
            "step": 1,
            "train_sha256": "a" * 64,
            "run_contract": {**static_contract, "planned_steps": 2},
            "schedule_total_steps": 2,
        },
        {
            "step": 2,
            "train_sha256": "a" * 64,
            "run_contract": static_contract,
            "schedule_total_steps": 2,
        },
        {
            "step": 3,
            "train_sha256": "a" * 64,
            "run_contract": static_contract,
            "schedule_total_steps": 5,
        },
    ]
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    discarded = reconcile_metrics_file(
        str(metrics),
        {"train_sha256": "a" * 64, "run_contract": static_contract},
        maximum_committed_step=2,
        run_name="Pretraining",
    )
    kept = [
        json.loads(line) for line in
        metrics.read_text(encoding="utf-8").splitlines()
    ]
    assert discarded == 1
    assert [row["step"] for row in kept] == [1, 2]


def test_metrics_reconciliation_rejects_decreasing_schedule_horizon(tmp_path):
    import json

    metrics = tmp_path / "metrics.jsonl"
    contract = {"batch_size": 1, "seed": 9}
    rows = [
        {"step": 1, "train_sha256": "a" * 64, "run_contract": contract, "schedule_total_steps": 5},
        {"step": 2, "train_sha256": "a" * 64, "run_contract": contract, "schedule_total_steps": 4},
    ]
    metrics.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="horizon cannot decrease"):
        reconcile_metrics_file(
            str(metrics),
            {"train_sha256": "a" * 64, "run_contract": contract},
            maximum_committed_step=2,
            run_name="SFT",
        )


def test_sft_zero_target_batch_fails_instead_of_skipping_optimizer_steps():
    from qmini_addsub.sft import _require_response_targets

    labels = torch.full((2, 8), -100, dtype=torch.long)
    with pytest.raises(ValueError, match="no assistant response targets"):
        _require_response_targets(labels, step=17)

    labels[0, 3] = 42
    assert _require_response_targets(labels, step=18) == 1


def test_generation_stops_a_fourgram_phrase_before_a_fifth_repeat():
    from qmini_addsub.generate import _would_exceed_fourgram_repeat

    three_repeats = [1, 2, 3, 4] * 3
    four_repeats = [1, 2, 3, 4] * 4
    five_repeats = [1, 2, 3, 4] * 5

    assert _would_exceed_fourgram_repeat(three_repeats, 1) is False
    assert _would_exceed_fourgram_repeat(four_repeats, 1) is False
    assert _would_exceed_fourgram_repeat(five_repeats, 1) is True
    assert _would_exceed_fourgram_repeat([1, 2, 3], 9) is False


def test_generation_fourgram_guard_does_not_block_a_different_continuation():
    from qmini_addsub.generate import _would_exceed_fourgram_repeat

    generated = [1, 2, 3, 4] * 4
    assert _would_exceed_fourgram_repeat(generated, 9) is False
