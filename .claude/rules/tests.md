---
paths:
  - "tests/**/*.py"
---

# Test Conventions

- pytest is the test runner. Run with `make test` or `uv run pytest -x -q`.
- Test files mirror source: `src/kvguard/features.py` → `tests/test_features.py`.
- Use synthetic fixtures for unit tests — don't depend on real sweep result files.
- Parametrize tests for multiple compression ratios, press types, etc.
- Test edge cases: empty signals, zero tokens, NaN values, single-prompt datasets.
- Integration tests in `tests/test_integration.py` test the full pipeline.
- All tests must pass before committing: `make check` enforces this.
- Mark GPU-dependent tests with `@pytest.mark.skipif(not torch.cuda.is_available(), ...)`.
