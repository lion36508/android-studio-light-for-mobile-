import json

from scripts.precision_audit import collect_report, main


def test_precision_audit_classifies_training_model_and_docs(tmp_path):
    (tmp_path / "qmini").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "qmini" / "train.py").write_text(
        'optimizer = torch.optim.AdamW(params)\n'
        'trainable_weight_policy = "fp32-master-q2-ste"\n',
        encoding="utf-8",
    )
    (tmp_path / "qmini" / "model.py").write_text(
        'return value.float()\n', encoding="utf-8"
    )
    (tmp_path / "docs" / "precision.md").write_text(
        "The row scale remains FP32.\n", encoding="utf-8"
    )

    report = collect_report(tmp_path)
    categories = {item["category"] for item in report["findings"]}

    assert report["schema"] == "qmini-precision-audit-v1"
    assert report["finding_count"] == 3
    assert "training_parameter_or_optimizer" in categories
    assert "model_numeric_or_parameter" in categories
    assert "documentation" in categories
    assert all(item["review_required"] for item in report["findings"])


def test_precision_audit_writes_json_and_enforces_selected_gate(tmp_path):
    source = tmp_path / "qmini"
    source.mkdir()
    (source / "train.py").write_text(
        'optimizer_state_dtype = torch.float32\n', encoding="utf-8"
    )
    output = tmp_path / "reports" / "precision.json"

    status = main([
        "--root", str(tmp_path),
        "--output", str(output),
        "--fail-on-category", "training_parameter_or_optimizer",
    ])

    assert status == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["finding_count"] == 1
    assert report["category_counts"]["training_parameter_or_optimizer"] == 1


def test_precision_audit_skips_generated_directories(tmp_path):
    source = tmp_path / "qmini"
    generated = tmp_path / ".venv" / "lib"
    source.mkdir()
    generated.mkdir(parents=True)
    (source / "clean.py").write_text("value = 1\n", encoding="utf-8")
    (generated / "third_party.py").write_text(
        "value = torch.float32\n", encoding="utf-8"
    )

    report = collect_report(tmp_path)
    assert report["files_scanned"] == 1
    assert report["finding_count"] == 0
