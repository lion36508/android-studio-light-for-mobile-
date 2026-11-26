import argparse
from types import SimpleNamespace

from qmini import cli


def test_cli_main_returns_subcommand_status(monkeypatch):
    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(func=lambda _args: 7),
    )

    assert cli.main() == 7


import pytest

from qmini.cli import _validate_thinking_steps


def test_cli_thinking_steps_follow_model_limit_two():
    _validate_thinking_steps(1, 2)
    _validate_thinking_steps(2, 2)
    with pytest.raises(ValueError, match="1..2"):
        _validate_thinking_steps(3, 2)


def test_cli_thinking_steps_allow_custom_model_limit_above_four():
    _validate_thinking_steps(5, 5)
    with pytest.raises(ValueError, match="1..5"):
        _validate_thinking_steps(6, 5)


def test_cli_thinking_steps_accept_model_default():
    _validate_thinking_steps(None, 2)


def test_tokenizer_cli_rejects_zero_validation_fraction_at_public_boundary(
    tmp_path, monkeypatch
):
    import sys
    from qmini.cli import main

    corpus = tmp_path / "cli-corpus.jsonl"
    corpus.write_text(
        '{"text":"First independent tokenizer document."}\n'
        '{"text":"Second independent tokenizer document."}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "tokenizer",
            "--data-dir", str(corpus),
            "--output", str(tmp_path / "tokenizer.json"),
            "--vocab-size", "300",
            "--min-frequency", "1",
            "--config-output", str(tmp_path / "config.json"),
            "--valid-fraction", "0",
        ],
    )
    with pytest.raises(ValueError, match="greater than 0 and below 0.5"):
        main()


def test_eval_conversations_cli_routes_through_structured_evaluator(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    import qmini.cli as cli

    model = SimpleNamespace(
        config=SimpleNamespace(max_thinking_steps=3),
    )
    tokenizer = object()
    expected_report = {"format": "qmini-conversation-evaluation-v1"}
    called = {}

    monkeypatch.setattr(
        cli, "load_model", lambda model_dir, device=None: (
            model, tokenizer, "cpu"
        )
    )

    def fake_evaluate(*args):
        called["args"] = args
        return expected_report

    monkeypatch.setattr(cli, "evaluate_conversation_file", fake_evaluate)
    monkeypatch.setattr(
        cli, "emit_report",
        lambda report, output: called.update({"report": report, "output": output}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "eval-conversations",
            "--model-dir", str(tmp_path / "model"),
            "--dataset", str(tmp_path / "conversations.jsonl"),
            "--thinking-steps", "3",
            "--output", str(tmp_path / "report.json"),
        ],
    )
    assert cli.main() == 0
    assert called["args"][0] is model
    assert called["args"][1] is tokenizer
    assert called["args"][2] == str(tmp_path / "conversations.jsonl")
    assert called["report"] is expected_report
    assert called["output"] == str(tmp_path / "report.json")



def test_training_cli_requires_an_explicit_total_step_target(monkeypatch):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "train",
            "--config", "configs/qmini_smoke.json",
            "--train-bin", "train.bin",
            "--valid-bin", "valid.bin",
            "--tokenizer", "tokenizer.json",
            "--output-dir", "run",
        ],
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2



def test_audit_is_config_only_without_tensor_allocation(monkeypatch, capsys):
    import sys
    import qmini.cli as cli

    def forbidden(*_args, **_kwargs):
        raise AssertionError("audit must not allocate model tensors")

    monkeypatch.setattr(cli, "QMiniMoE", forbidden)
    monkeypatch.setattr(
        sys, "argv",
        ["qmini", "audit", "--config", "configs/qmini_light_0.7m.json"],
    )
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert "705,742" in output
    assert "Tensor allocation: skipped" in output


def test_preflight_cli_reports_no_dataset_or_optimizer_update(monkeypatch, capsys):
    import json
    import sys
    from qmini import cli

    monkeypatch.setattr(
        sys, "argv",
        [
            "qmini", "preflight",
            "--config", "configs/qmini_smoke.json",
            "--device", "cpu",
            "--sequence-length", "8",
        ],
    )
    assert cli.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dataset_opened"] is False
    assert report["optimizer_step_performed"] is False
    assert report["weights_updated"] is False
    assert report["logits_shape"] == [1, 8, 128]


def test_generate_cli_serialized_prompt_requires_warning_and_passes_penalties(
    monkeypatch, capsys, tmp_path
):
    import sys
    from types import SimpleNamespace

    model = SimpleNamespace(
        config=SimpleNamespace(max_thinking_steps=2),
    )
    tokenizer = object()
    called = {}
    monkeypatch.setattr(cli, "load_model", lambda *_args, **_kwargs: (model, tokenizer, "cpu"))

    def fake_generate(*args, **kwargs):
        called["args"] = args
        called["kwargs"] = kwargs
        return "generated"

    monkeypatch.setattr(cli, "generate", fake_generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qmini", "generate",
            "--model-dir", str(tmp_path / "model"),
            "--prompt", "<|assistant|>trusted",
            "--serialized-prompt",
            "--frequency-penalty", "0.4",
            "--presence-penalty", "0.7",
        ],
    )
    assert cli.main() == 0
    assert "WARNING: --serialized-prompt trusts role/control-token markers" in capsys.readouterr().err
    assert called["kwargs"]["serialized_prompt"] is True
    assert called["kwargs"]["frequency_penalty"] == 0.4
    assert called["kwargs"]["presence_penalty"] == 0.7

def test_training_cli_forwards_allow_dataset_change(monkeypatch, tmp_path):
    import qmini.train as train_module
    import sys

    seen = {}

    def fake_train_model(*args, **kwargs):
        seen["kwargs"] = kwargs
        return tmp_path / "run"

    monkeypatch.setattr(train_module, "train_model", fake_train_model)
    base_argv = [
        "qmini",
        "train",
        "--config",
        "configs/qmini_smoke.json",
        "--train-bin",
        "train.bin",
        "--valid-bin",
        "valid.bin",
        "--tokenizer",
        "tokenizer.json",
        "--output-dir",
        "run",
        "--steps",
        "3",
    ]
    monkeypatch.setattr(sys, "argv", [*base_argv, "--allow-dataset-change"])
    assert cli.main() == 0
    assert seen["kwargs"]["allow_dataset_change"] is True
    monkeypatch.setattr(sys, "argv", base_argv)
    assert cli.main() == 0
    assert seen["kwargs"]["allow_dataset_change"] is False
