"""PyTest fixtures for the PTB-MR/sequences package."""

from copy import deepcopy

import pytest
from sequences.utils import sys_defaults


@pytest.fixture(scope='function')
def system_defaults():
    """System defaults for sequence generation."""
    return deepcopy(sys_defaults)
