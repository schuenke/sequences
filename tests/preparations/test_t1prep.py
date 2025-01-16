"""Tests for the adiabatic T1 preparation block."""

import pypulseq as pp
import pytest
from sequences.preparations.t1prep import add_t1prep


def test_add_t1prep_system_defaults_if_none(system_defaults):
    """Test if system defaults are used if no system limits are provided."""
    _, block_duration1 = add_t1prep(system=system_defaults)
    _, block_duration2 = add_t1prep(system=None)

    assert block_duration1 == block_duration2


@pytest.mark.parametrize(('inversion_time', 'rf_duration'), [(0.01, 10e-3), (0.015, 20e-3)])
def test_add_t2prep_fail_on_short_inversion_time(system_defaults, inversion_time, rf_duration):
    """Test if function raises an error when desired inversion time is too short for given RF and spoiler durations."""
    seq = pp.Sequence(system=system_defaults)
    with pytest.raises(ValueError, match='Inversion time too short'):
        add_t1prep(seq=seq, system=system_defaults, inversion_time=inversion_time, rf_duration=rf_duration)


@pytest.mark.parametrize(
    ('inversion_time', 'rf_duration', 'add_spoiler', 'spoiler_ramp_time', 'spoiler_flat_time'),
    [
        (21e-3, 10.24e-3, True, 6e-4, 8.4e-3),
        (40e-3, 10.24e-3, True, 6e-4, 8.4e-3),
        (21e-3, 20.00e-3, True, 6e-4, 8.4e-3),
        (21e-3, 10.24e-3, False, 6e-4, 8.4e-3),
        (21e-3, 10.24e-3, True, 1e-3, 10e-3),
    ],
    ids=['defaults', 'longer_ti', 'longer_pulse', 'no_spoiler', 'longer_spoiler'],
)
def test_add_t1prep_duration(
    system_defaults, inversion_time, rf_duration, add_spoiler, spoiler_ramp_time, spoiler_flat_time
):
    """Ensure the default parameters are set correctly."""
    seq = pp.Sequence(system=system_defaults)

    seq, block_duration = add_t1prep(
        seq=seq,
        system=system_defaults,
        inversion_time=inversion_time,
        rf_duration=rf_duration,
        add_spoiler=add_spoiler,
        spoiler_ramp_time=spoiler_ramp_time,
        spoiler_flat_time=spoiler_flat_time,
    )

    manual_time_calc = (
        system_defaults.rf_dead_time  # dead time before 180° inversion pulse
        + rf_duration / 2  # half duration of 180° inversion pulse
        + inversion_time  # inversion time
    )

    assert sum(seq.block_durations.values()) == block_duration
    assert block_duration == pytest.approx(manual_time_calc)
