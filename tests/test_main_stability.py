"""CPU-only numerical and boundary checks; no trained accuracy claim."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from spindle_power.load_prediction.measurement import Measurement
from spindle_power.load_prediction.main_stability import speed_support, stability_channels, apply_stability_columns


def measurement(speed, *, dt=.001):
    speed = np.asarray(speed, float)
    n = len(speed)
    return Measurement("synthetic", np.arange(n) * dt,
        dict(spindle_speed_rpm=speed, program_line_raw=np.ones(n), spindle_power_w=np.ones(n)),
        np.ones(n, bool), {"sample_period_s": dt}, np.arange(n, dtype=np.int64))


def qualify(record):
    return stability_channels(pd.DataFrame({"line": [1], "spindle_cmd_rpm": [8000.]}), record)


def test_constant_preserves_complete_two_filter_support():
    arrays, audit = speed_support(measurement(np.full(1001, 8000.)))
    assert audit["blocks"][0]["nominal_window"] == 101
    assert not arrays["numerical_support_valid"][:101].any()
    assert not arrays["numerical_support_valid"][-101:].any()
    assert arrays["numerical_support_valid"][101:-101].all()
    assert np.max(np.abs(arrays["rate_rpm_s"])) < 1e-6
    assert not arrays["transition_guard"].any()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 100, 101, 201, 202, 203])
def test_short_blocks_never_shrink_or_fallback(n):
    arrays, audit = speed_support(measurement(np.full(n, 8000.)))
    assert arrays["numerical_support_valid"].sum() == (1 if n == 203 else 0)
    if n < 203:
        assert np.isnan(arrays["rate_rpm_s"]).all()
        assert audit["blocks"][0]["status"] == "short_complete_support"


@pytest.mark.parametrize("jump", [-1000., -20., 20., 1000.])
def test_step_and_ring_zero_crossings_are_guarded(jump):
    speed = np.full(1601, 8000.)
    speed[800:] += jump
    _, arrays, _ = qualify(measurement(speed))
    assert arrays["transition_guard"][740:860].all()
    assert not arrays["raw_speed_qualified"][740:860].any()
    assert arrays["raw_speed_qualified"][200:500].all()


@pytest.mark.parametrize("slope", [-120., 120., -500., 500.])
def test_acceleration_and_braking_ramps_rejected(slope):
    t = np.arange(1001) * .001
    _, arrays, _ = qualify(measurement(8000. + slope * (t - .5)))
    assert not arrays["raw_speed_qualified"].any()
    np.testing.assert_allclose(arrays["rate_rpm_s"][101:-101], slope, atol=1e-7)


def test_raw_speed_band_spike_is_not_hidden_by_median_filter():
    speed = np.full(1001, 8000.)
    speed[500] = 9000.
    _, arrays, _ = qualify(measurement(speed))
    assert not arrays["raw_speed_band_valid"][500]
    assert not arrays["raw_speed_qualified"][500]
    assert arrays["raw_speed_qualified"][499]


@pytest.mark.parametrize("kind", ["valid", "time", "index", "explicit"])
def test_gaps_split_filter_and_guard_support(kind):
    record = measurement(np.full(1201, 8000.))
    record.values["spindle_speed_rpm"][450:600] += 20.
    breaks = np.zeros(1201, bool)
    if kind == "valid":
        record.valid[600] = False
    elif kind == "time":
        record.time_s[600:] += .01
    elif kind == "index":
        record.source_indices[600:] += 1
    else:
        breaks[600] = True
    arrays, audit = speed_support(record, break_before=breaks)
    assert len(audit["blocks"]) == 2
    second = audit["blocks"][1]["array_start"]
    assert not arrays["numerical_support_valid"][second:second + 101].any()
    assert not arrays["transition_guard"][second:].any()


def test_nonuniform_timing_is_explicitly_unsupported():
    record = measurement(np.full(1001, 8000.))
    record.time_s[500] += .00001
    arrays, audit = speed_support(record)
    assert not arrays["numerical_support_valid"].any()
    assert audit["blocks"][0]["status"] == "nonuniform_time"


def test_power_amplitude_not_an_input_and_indices_required():
    record = measurement(np.full(1001, 8000.))
    _, first, _ = qualify(record)
    record.values["spindle_power_w"] *= 1e6
    _, second, _ = qualify(record)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    record.source_indices = None
    with pytest.raises(ValueError, match="原始采样索引"):
        speed_support(record)


def test_interpolation_is_binary_linear_value_and_complete_support_is_strict():
    aligned = pd.DataFrame(dict(spindle_rate_rpm_s=[0.] * 4,
        spindle_rate_support_interpolated=[1., 1., .9999, 1.],
        spindle_transition_guard_interpolated=[0.] * 4,
        spindle_stable_interpolated=[1., .999, 1., .9989]))
    frame = pd.DataFrame({"aligned": [True] * 4})
    apply_stability_columns(frame, aligned)
    assert frame.spindle_stable.tolist() == [True, True, False, False]


def test_any_positive_interpolated_guard_rejects_target():
    aligned = pd.DataFrame(dict(spindle_rate_rpm_s=[0.], spindle_rate_support_interpolated=[1.],
        spindle_transition_guard_interpolated=[1e-9], spindle_stable_interpolated=[1.-1e-9]))
    frame = pd.DataFrame({"aligned": [True]})
    apply_stability_columns(frame, aligned)
    assert not frame.spindle_stable.any()


@pytest.mark.parametrize("change", ["repeat", "reverse"])
def test_nonincreasing_time_splits_without_resampling(change):
    record = measurement(np.full(1001, 8000.))
    record.time_s[500:] -= .001 if change == "repeat" else .01
    arrays, audit = speed_support(record)
    assert len(audit["blocks"]) == 2
    assert not arrays["numerical_support_valid"][399:601].any()


def test_quantized_constant_speed_positive_control():
    quantized = 8000. + 1.8310546875 * np.tile([0., 1., 0., -1.], 501)[:2001]
    _, arrays, _ = qualify(measurement(quantized))
    assert arrays["raw_speed_qualified"][300:-300].all()
    assert not arrays["transition_guard"][300:-300].any()


@pytest.mark.parametrize("dt", [.001, .002, .005])
@pytest.mark.parametrize("rate,passes", [(79., True), (81., False), (-79., True), (-81., False)])
def test_physical_rate_boundary_consistent_across_sampling_periods(dt, rate, passes):
    t = np.arange(int(1 / dt) + 1) * dt
    _, arrays, _ = qualify(measurement(8000. + rate * (t - .5), dt=dt))
    centre = (t >= .3) & (t <= .7)
    assert arrays["numerical_support_valid"][centre].all()
    assert arrays["raw_speed_qualified"][centre].all() == passes
    np.testing.assert_allclose(arrays["rate_rpm_s"][centre], rate, atol=1e-7)
