"""Shared test fixtures and configuration.

Provides torch cleanup between test modules to prevent MPS resource
contention when running the full test suite in a single process.
"""

import gc

import pytest


@pytest.fixture(autouse=True)
def _cleanup_torch_after_test() -> None:  # type: ignore[misc]
    """Force garbage collection after each test to release GPU memory."""
    yield
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError, RuntimeError):
        pass
