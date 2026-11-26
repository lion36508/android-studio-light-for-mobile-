from __future__ import annotations

import random

import pytest

from qmini_addsub.curriculum import SelfPacedCurriculumSampler


def _sampler(*, total_steps=200, warmup_steps=2, uniform_fraction=0.35):
    return SelfPacedCurriculumSampler(
        dataset_size=8,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        uniform_fraction=uniform_fraction,
    )


def test_batch_one_sampler_keeps_fractional_uniform_replay_quota():
    sampler = _sampler(total_steps=202, warmup_steps=2)
    rng = random.Random(917)
    uniform_count = 0
    post_warmup_steps = 0

    for step in range(202):
        indices, info = sampler.sample(1, rng)
        assert len(indices) == len(set(indices)) == 1
        if step >= 2:
            post_warmup_steps += 1
            uniform_count += info["uniform_samples"]
        sampler.update(indices, [0.5 + (indices[0] / 10.0)])

    assert uniform_count == 70
    assert uniform_count / post_warmup_steps == pytest.approx(0.35, abs=0.005)
    assert sampler.uniform_remainder == pytest.approx(0.0)


def test_sampler_warmup_then_uses_only_measured_losses():
    sampler = _sampler(total_steps=10, warmup_steps=1)
    rng = random.Random(31)

    warm_indices, warm_info = sampler.sample(2, rng)
    assert warm_info["mode"] == "uniform_warmup"
    assert len(warm_indices) == 2
    assert all(sampler.ema_loss[index] is None for index in range(sampler.dataset_size))
    sampler.update(warm_indices, [0.1, 2.0])

    focused_indices, focused_info = sampler.sample(2, rng)
    assert focused_info["mode"] in {"self_paced", "uniform_fallback"}
    assert len(focused_indices) == len(set(focused_indices)) == 2
    assert focused_info["seen_fraction"] == pytest.approx(0.25)
    assert focused_info["requested_uniform_fraction"] == pytest.approx(0.35)


def test_sampler_state_and_rng_resume_are_exact():
    sampler = _sampler(total_steps=20, warmup_steps=1)
    rng = random.Random(88)
    for _ in range(5):
        indices, _ = sampler.sample(1, rng)
        sampler.update(indices, [float(indices[0] + 1)])

    state = sampler.state_dict()
    rng_state = rng.getstate()
    restored = _sampler(total_steps=20, warmup_steps=1)
    restored.load_state_dict(state)
    restored_rng = random.Random()
    restored_rng.setstate(rng_state)

    expected_indices, expected_info = sampler.sample(1, rng)
    actual_indices, actual_info = restored.sample(1, restored_rng)
    assert actual_indices == expected_indices
    assert actual_info == expected_info


def test_sampler_allows_explicit_schedule_extension_but_not_shortening():
    old = _sampler(total_steps=20, warmup_steps=2)
    rng = random.Random(99)
    indices, _ = old.sample(1, rng)
    old.update(indices, [1.0])
    state = old.state_dict()

    extended = _sampler(total_steps=30, warmup_steps=2)
    extended.load_state_dict(state)
    assert extended.total_steps == 30

    shortened = _sampler(total_steps=10, warmup_steps=2)
    with pytest.raises(ValueError, match="only be extended"):
        shortened.load_state_dict(state)


@pytest.mark.parametrize("value", [0.0, 0.1, 0.34, 1.01, float("nan"), float("inf"), True])
def test_sampler_rejects_invalid_uniform_replay_fraction(value):
    with pytest.raises((TypeError, ValueError)):
        SelfPacedCurriculumSampler(
            dataset_size=8,
            total_steps=20,
            warmup_steps=2,
            uniform_fraction=value,
        )


def test_sampler_rejects_duplicate_and_invalid_loss_updates():
    sampler = _sampler()
    with pytest.raises(ValueError, match="duplicate"):
        sampler.update([1, 1], [0.5, 0.6])
    with pytest.raises(ValueError, match="finite and non-negative"):
        sampler.update([1], [float("nan")])
    with pytest.raises(ValueError, match="out of range"):
        sampler.update([8], [0.5])


def test_sampler_rejects_resume_state_with_invalid_ema_loss():
    sampler = _sampler()
    state = sampler.state_dict()
    state["ema_loss"][0] = float("nan")
    with pytest.raises(ValueError, match="invalid loss"):
        sampler.load_state_dict(state)


def test_sampler_rejects_step_beyond_saved_schedule_when_resumed_extended():
    original = _sampler(total_steps=20, warmup_steps=2)
    state = original.state_dict()
    state["step"] = 21

    extended = _sampler(total_steps=30, warmup_steps=2)
    with pytest.raises(ValueError, match="saved schedule horizon"):
        extended.load_state_dict(state)


def test_uniform_control_path_matches_random_sample_and_does_not_scan_ema():
    sampler = _sampler(total_steps=20, warmup_steps=2, uniform_fraction=1.0)
    rng = random.Random(123)
    expected_rng = random.Random(123)
    expected = expected_rng.sample(range(sampler.dataset_size), k=3)

    indices, info = sampler.sample(3, rng)

    assert indices == expected
    assert info["mode"] == "uniform_control"
    assert info["seen_fraction"] == 0.0
    assert sampler.step == 1
