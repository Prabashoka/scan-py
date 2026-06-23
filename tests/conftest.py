"""Pytest fixtures for SCAN tests."""

from __future__ import annotations

import pytest

from scan import scan_cpd as _scan_cpd


@pytest.fixture
def scan_cpd():
    """Return the public detector function under test."""
    return _scan_cpd