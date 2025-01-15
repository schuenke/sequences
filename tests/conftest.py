"""PyTest fixtures for the PTB-MR/sequences package."""

import pypulseq as pp
import pytest


@pytest.fixture(scope='function')
def system_defaults():
    """System defaults for sequence generation."""
    return pp.Opts(
        max_grad=30,
        grad_unit='mT/m',
        max_slew=120,
        slew_unit='T/m/s',
        rf_ringdown_time=30e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
