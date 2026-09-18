import os

import pytest

from ropf.network import read_matpower

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASE = os.path.join(ROOT, "data", "case_ACTIVSg200.m")
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


@pytest.fixture(scope="session")
def net():
    return read_matpower(CASE)
