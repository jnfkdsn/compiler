from __future__ import annotations

import time

import pytest

try:
    import pytest_benchmark.fixture  # type: ignore  # noqa: F401
except Exception:

    @pytest.fixture
    def benchmark():
        """Fallback benchmark fixture for environments without pytest-benchmark."""

        def _run(func, *args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            _run.last_elapsed_s = time.perf_counter() - start
            return result

        _run.last_elapsed_s = 0.0
        return _run
