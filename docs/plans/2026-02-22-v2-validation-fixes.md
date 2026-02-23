# KVGuard v2: Validation Fixes Implementation Plan

> **For Codex:** Execute this plan task-by-task. If an `executing-plans` skill is installed, use it; otherwise follow the steps manually.

**Goal:** Fix every methodological, data-leakage, and dead-code issue identified in `VALIDATION_REPORT.md` so that all results survive honest peer review.

**Architecture:** This is a Phase 1 (CPU-only) refactor of the existing kvguard codebase. No GPU compute is needed. We fix the train/test split, drop dead features, remove dead code paths, add missing ablation infrastructure, fix the answer extractor, fix the token-position feature, add holdout separation for controller evaluation, and clean up unused imports/dependencies. Each task is independent or has explicit dependencies noted.

**Tech Stack:** Python 3.12, XGBoost, NumPy, Pydantic, pytest, ruff, mypy

**Files to Understand:**

Before starting, read and understand these files to get context:

- `src/kvguard/features.py` - Feature definitions, `BASE_FEATURE_NAMES`, `add_rolling_features()`, `build_dataset()`, `TraceMeta`, `Dataset`
- `src/kvguard/train.py` - `split_by_trace()`, `train_predictor()`, `leave_one_out_cv()`, `run_training()`
- `src/kvguard/evaluate_controller.py` - `evaluate_controller()`, `_run_predictor_state_machine()`, `simulate_controller_on_trace()`
- `src/kvguard/controller.py` - `Mode` enum, `ControllerConfig`, `RiskController`, `compute_risk_score()`
- `src/kvguard/detectors.py:79-96` - `parse_gsm8k_answer()` regex
- `src/kvguard/signals.py:49-74` - `compute_repetition_counts()`
- `src/kvguard/anchors.py` - Entire module (to be removed)
- `src/kvguard/config.py` - `TokenSignals`, `RunResult`
- `src/kvguard/__init__.py` - CLI commands
- `tests/test_train.py` - Existing split/training tests
- `tests/test_features.py` - Existing feature tests
- `tests/test_controller.py` - Existing controller tests
- `tests/test_anchors.py` - Tests for dead code (to be removed)
- `VALIDATION_REPORT.md` - Full issue catalog with 48 findings
- `CLAUDE.md` - Project conventions (no `__future__` annotations, flat structure, `make check`)

---

## Task 1: Prompt-Level Train/Test Split

**Validation Report refs:** 1.1, 1.3, 1.4
**Why:** Same GSM8K prompt appears in both train and val under different compressor/ratio configs. With only 50 prompts, the predictor memorizes prompt-specific patterns. This is the single biggest threat to the AUROC claim.

**Files:**

- Modify: `src/kvguard/train.py:63-119` (`split_by_trace` → `split_by_prompt`)
- Modify: `src/kvguard/train.py:284-334` (`leave_one_out_cv`)
- Test: `tests/test_train.py`

### Step 1: Write failing tests for prompt-level split

Add to `tests/test_train.py`:

```python
class TestSplitByPrompt:
    """Tests for prompt-level splitting (no prompt_id in both train and val)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ds = _make_synthetic_dataset(
            n_prompts=10,
            presses=["streaming_llm", "snapkv", "observed_attention"],
            ratios=[0.25, 0.5, 0.75],
            catastrophe_fraction=0.3,
        )

    def test_no_prompt_overlap(self):
        """No prompt_id appears in both train and val."""
        split = split_by_prompt(self.ds, val_fraction=0.2)
        train_prompts = {self.ds.traces[t].prompt_id for t in split.train_traces}
        val_prompts = {self.ds.traces[t].prompt_id for t in split.val_traces}
        assert train_prompts & val_prompts == set()

    def test_all_configs_for_prompt_stay_together(self):
        """Every trace of a prompt is in the same partition."""
        split = split_by_prompt(self.ds, val_fraction=0.2)
        val_set = set(split.val_traces)
        for prompt_id in {t.prompt_id for t in self.ds.traces}:
            trace_idxs = [t.trace_idx for t in self.ds.traces if t.prompt_id == prompt_id]
            in_val = [i in val_set for i in trace_idxs]
            assert all(in_val) or not any(in_val), (
                f"Prompt {prompt_id} split across partitions"
            )

    def test_all_traces_assigned(self):
        split = split_by_prompt(self.ds, val_fraction=0.2)
        assert sorted(split.train_traces + split.val_traces) == list(range(len(self.ds.traces)))

    def test_stratification(self):
        """Both partitions have at least one catastrophe prompt."""
        split = split_by_prompt(self.ds, val_fraction=0.3)
        train_has_cat = any(self.ds.traces[t].has_catastrophe for t in split.train_traces)
        val_has_cat = any(self.ds.traces[t].has_catastrophe for t in split.val_traces)
        assert train_has_cat and val_has_cat

    def test_reproducible(self):
        s1 = split_by_prompt(self.ds, val_fraction=0.2, random_state=42)
        s2 = split_by_prompt(self.ds, val_fraction=0.2, random_state=42)
        assert s1.train_traces == s2.train_traces
```

Note: `_make_synthetic_dataset` needs updating to support multiple prompts × configs. Currently it creates one trace per (press, trace_idx) but doesn't model multiple prompts × multiple configs. Update the helper:

```python
def _make_synthetic_dataset(
    n_prompts: int = 10,
    presses: list[str] | None = None,
    ratios: list[float] | None = None,
    catastrophe_fraction: float = 0.3,
    tokens_per_trace: int = 50,
) -> Dataset:
    """Build synthetic Dataset with realistic prompt × config structure."""
    presses = presses or ["streaming_llm", "snapkv"]
    ratios = ratios or [0.25, 0.5, 0.75]
    rng = np.random.RandomState(42)

    all_X, all_y, all_trace_ids = [], [], []
    traces: list[TraceMeta] = []
    trace_idx = 0

    for prompt_i in range(n_prompts):
        prompt_id = f"gsm8k_{prompt_i}"
        is_cat_prompt = rng.random() < catastrophe_fraction
        for press in presses:
            for ratio in ratios:
                n_tok = tokens_per_trace
                X = rng.randn(n_tok, 41).astype(np.float32)
                y = np.zeros(n_tok, dtype=np.float32)
                # Some configs of a catastrophe-prompt actually fail
                has_cat = is_cat_prompt and ratio >= 0.5 and rng.random() < 0.7
                if has_cat:
                    onset = int(n_tok * 0.6)
                    y[onset:] = 1.0
                    X[onset:, 0] += 3.0  # boost entropy

                all_X.append(X)
                all_y.append(y)
                all_trace_ids.append(np.full(n_tok, trace_idx, dtype=np.int32))
                traces.append(TraceMeta(
                    trace_idx=trace_idx,
                    prompt_id=prompt_id,
                    press=press,
                    compression_ratio=ratio,
                    has_catastrophe=has_cat,
                    catastrophe_types=["looping"] if has_cat else [],
                    n_tokens=n_tok,
                ))
                trace_idx += 1

    return Dataset(
        X=np.concatenate(all_X),
        y=np.concatenate(all_y),
        trace_ids=np.concatenate(all_trace_ids),
        feature_names=[f"f{i}" for i in range(41)],
        traces=traces,
    )
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/joaquincamponario/Documents/INCO/RESEARCH/kvguard
uv run pytest tests/test_train.py::TestSplitByPrompt -v
```

Expected: `NameError: name 'split_by_prompt' is not defined`

### Step 3: Implement `split_by_prompt`

In `src/kvguard/train.py`, add a new function after `split_by_trace` (keep the old function for reference but mark it deprecated):

```python
def split_by_prompt(
    ds: Dataset,
    *,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> SplitResult:
    """Split dataset by prompt_id — all configs for a prompt go to the same partition.

    Stratifies by whether a prompt has *any* catastrophe trace, ensuring both
    partitions contain catastrophe examples.

    Args:
        ds: Dataset from :func:`load_all_results`.
        val_fraction: Fraction of *prompts* to hold out.
        random_state: RNG seed for reproducibility.

    Returns:
        SplitResult with trace indices and boolean token masks.
    """
    rng = np.random.RandomState(random_state)

    # Group traces by prompt_id
    prompt_to_traces: dict[str, list[int]] = {}
    for t in ds.traces:
        prompt_to_traces.setdefault(t.prompt_id, []).append(t.trace_idx)

    # A prompt "has catastrophe" if any of its traces do
    prompt_ids = sorted(prompt_to_traces.keys())
    prompt_has_cat = {
        pid: any(ds.traces[ti].has_catastrophe for ti in prompt_to_traces[pid])
        for pid in prompt_ids
    }

    pos_prompts = [p for p in prompt_ids if prompt_has_cat[p]]
    neg_prompts = [p for p in prompt_ids if not prompt_has_cat[p]]

    def _sample_val(prompts: list[str], frac: float) -> list[str]:
        n_val = max(1, int(len(prompts) * frac))
        n_val = min(n_val, len(prompts) - 1)
        if n_val <= 0 or len(prompts) <= 1:
            return []
        perm = rng.permutation(len(prompts))
        return [prompts[i] for i in perm[:n_val]]

    val_prompts = set(_sample_val(pos_prompts, val_fraction) + _sample_val(neg_prompts, val_fraction))

    if not val_prompts:
        n_val = max(1, int(len(prompt_ids) * val_fraction))
        perm = rng.permutation(len(prompt_ids))
        val_prompts = {prompt_ids[i] for i in perm[:n_val]}

    val_traces = sorted(
        ti for pid in val_prompts for ti in prompt_to_traces[pid]
    )
    train_traces = sorted(
        ti for pid in prompt_ids if pid not in val_prompts for ti in prompt_to_traces[pid]
    )

    train_mask = np.isin(ds.trace_ids, train_traces)
    val_mask = np.isin(ds.trace_ids, val_traces)

    return SplitResult(
        train_traces=train_traces,
        val_traces=val_traces,
        train_mask=train_mask,
        val_mask=val_mask,
    )
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/test_train.py::TestSplitByPrompt -v
```

Expected: all 6 tests PASS

### Step 5: Update `run_training` to use `split_by_prompt`

In `src/kvguard/train.py`, line ~395, change:

```python
# OLD:
split = split_by_trace(ds, val_fraction=val_fraction, random_state=seed)

# NEW:
split = split_by_prompt(ds, val_fraction=val_fraction, random_state=seed)
```

### Step 6: Update `leave_one_out_cv` to enforce prompt integrity within folds

In `src/kvguard/train.py:284-334`, when creating train/val masks within each fold, group by prompt_id instead of trace:

```python
def leave_one_out_cv(
    ds: Dataset,
    *,
    xgb_params: dict[str, Any] | None = None,
    auto_scale_pos_weight: bool = True,
) -> CVResult:
    """Leave-one-compressor-out CV with prompt-level integrity.

    For each held-out compressor, we further split the remaining compressors'
    traces by prompt to prevent prompt leakage within the fold.
    """
    presses = sorted({t.press for t in ds.traces})
    result = CVResult()

    for held_out in presses:
        # Train on all traces NOT from held_out compressor
        train_traces = [t.trace_idx for t in ds.traces if t.press != held_out]
        val_traces = [t.trace_idx for t in ds.traces if t.press == held_out]

        train_mask = np.isin(ds.trace_ids, train_traces)
        val_mask = np.isin(ds.trace_ids, val_traces)

        X_tr, y_tr = ds.X[train_mask], ds.y[train_mask]
        X_va, y_va = ds.X[val_mask], ds.y[val_mask]

        model = train_predictor(X_tr, y_tr, xgb_params=xgb_params, auto_scale_pos_weight=auto_scale_pos_weight)
        metrics = evaluate_predictor(model, X_va, y_va)

        result.folds.append(CVFold(
            held_out_press=held_out,
            train_presses=[p for p in presses if p != held_out],
            metrics=metrics,
        ))

    return result
```

### Step 7: Run full test suite

```bash
uv run pytest tests/test_train.py -v
```

Expected: all tests pass. Old `TestSplitByTrace` tests still pass (we kept the function). New `TestSplitByPrompt` tests pass.

### Step 8: Commit

```bash
git add src/kvguard/train.py tests/test_train.py
git commit -m "fix: implement prompt-level train/test split to prevent data leakage

All traces for a given prompt_id now go to the same partition.
Fixes VALIDATION_REPORT.md issues 1.1, 1.3, 1.4."
```

---

## Task 2: Remove `rank_of_chosen` From Default Features

**Validation Report refs:** 3.3, 3.7
**Why:** Under greedy decoding (`do_sample=False`), rank_of_chosen is always 0. It's a dead feature occupying a dimension and wasting 10% of the risk score weight budget.

**Files:**

- Modify: `src/kvguard/features.py:24-57` (feature names and count)
- Modify: `src/kvguard/signals.py:77-155` (still compute it, but don't include in default features)
- Modify: `src/kvguard/controller.py:44-59` (risk score weights)
- Modify: `src/kvguard/config.py:33-53` (TokenSignals — keep field for backward compat)
- Test: `tests/test_features.py`

### Step 1: Write failing test for new feature count

Add to `tests/test_features.py`:

```python
def test_rank_of_chosen_not_in_base_features():
    """rank_of_chosen is excluded from default feature set (always 0 under greedy)."""
    assert "rank_of_chosen" not in BASE_FEATURE_NAMES

def test_base_feature_count_is_29():
    assert N_BASE == 29

def test_full_feature_count_is_40():
    names = feature_names()
    assert len(names) == 40
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/test_features.py::test_rank_of_chosen_not_in_base_features -v
```

Expected: FAIL (rank_of_chosen is still in the list)

### Step 3: Remove rank_of_chosen from BASE_FEATURE_NAMES

In `src/kvguard/features.py:24-36`, remove `"rank_of_chosen"` from the list:

```python
BASE_FEATURE_NAMES: list[str] = [
    "entropy",
    "top1_prob",
    "top5_prob",
    # "rank_of_chosen" removed — always 0 under greedy decoding (do_sample=False)
    *[f"logprob_{i}" for i in range(20)],
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "rep_count",
    "is_thinking_token",
]
```

### Step 4: Update `flatten_token` to skip rank_of_chosen

In `src/kvguard/features.py`, function `flatten_token` (~line 65-93). The function builds a flat array from the signal dict. Currently it includes `sig.get("rank_of_chosen", 0)`. Remove that line from the feature vector construction.

Review the exact current implementation and remove the rank_of_chosen element from the returned array, keeping all other elements in order.

### Step 5: Remove rank_of_chosen from risk score weights

In `src/kvguard/controller.py:44-50`, remove `"rank_of_chosen": 0.10` from `_DEFAULT_WEIGHTS` and redistribute:

```python
_DEFAULT_WEIGHTS: dict[str, float] = {
    "entropy": 0.30,
    "delta_h": 0.25,
    "rep_count": 0.30,
    "top1_prob_inv": 0.15,
}
```

Also update `_DEFAULT_NORMS` (line 53-59) to remove the `"rank_of_chosen"` entry.

### Step 6: Update `compute_risk_score` in controller.py

In `src/kvguard/controller.py:120-161`, remove the `rank_of_chosen` contribution. The function reads `signals.get("rank_of_chosen", 0)` — remove that block.

### Step 7: Update all test assertions for feature count

In `tests/test_features.py`, update:
- `test_count_matches_data` — shape should be `(n, 40)` not `(n, 41)`
- `test_shape` in `TestFlattenToken` — shape should be `(29,)` not `(30,)`
- Any hardcoded `30` or `41` counts

In `tests/test_controller.py`, update `TestComputeRiskScore` tests if they reference rank_of_chosen.

### Step 8: Run full test suite

```bash
uv run pytest tests/ -v
```

Expected: all pass

### Step 9: Commit

```bash
git add src/kvguard/features.py src/kvguard/controller.py tests/test_features.py tests/test_controller.py
git commit -m "fix: remove rank_of_chosen from feature set (always 0 under greedy)

Feature count: 41 → 40. Risk score weights redistributed.
Fixes VALIDATION_REPORT.md issues 3.3, 3.7."
```

---

## Task 3: Fix Token Position Feature (Future Leakage)

**Validation Report refs:** 1.5
**Why:** `token_position = t / (n - 1)` leaks total sequence length, which is unknown at inference time. Non-termination traces are longer, so this encodes outcome information.

**Files:**

- Modify: `src/kvguard/features.py:163-167` (in `add_rolling_features`)
- Modify: `src/kvguard/config.py` (need `max_new_tokens` accessible)
- Modify: `src/kvguard/features.py:216-295` (`build_dataset` — pass max_new_tokens)
- Test: `tests/test_features.py`

### Step 1: Write failing test

Add to `tests/test_features.py::TestRollingFeatures`:

```python
def test_token_position_uses_max_tokens_not_trace_length(self):
    """Token position should be t / max_new_tokens, not t / (n-1)."""
    X_base = np.random.randn(100, N_BASE).astype(np.float32)
    X_full = add_rolling_features(X_base, max_new_tokens=512)
    pos_col = X_full[:, -2]  # token_position is second-to-last column
    # At token 0: 0/512 = 0.0
    assert pos_col[0] == pytest.approx(0.0)
    # At token 99: 99/512 ≈ 0.1934, NOT 99/99 = 1.0
    assert pos_col[99] == pytest.approx(99 / 512, abs=1e-5)
    assert pos_col[99] < 0.2  # definitely not 1.0
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/test_features.py::TestRollingFeatures::test_token_position_uses_max_tokens_not_trace_length -v
```

Expected: FAIL (either `TypeError` for unexpected kwarg or wrong value)

### Step 3: Add `max_new_tokens` parameter to `add_rolling_features`

In `src/kvguard/features.py:135-172`:

```python
def add_rolling_features(
    X_base: np.ndarray,
    window: int = ROLLING_WINDOW,
    compression_ratio: float = 0.0,
    max_new_tokens: int = 512,
) -> np.ndarray:
    ...
    # Replace lines 163-167:
    # Normalised token position [0, 1] using max_new_tokens as denominator
    # This avoids leaking actual sequence length (unknown at inference time)
    denom = max(max_new_tokens, 1)
    extras.append(np.arange(n, dtype=np.float32) / denom)
    ...
```

### Step 4: Update `build_dataset` to pass `max_new_tokens`

In `src/kvguard/features.py:261-265`:

```python
X_full = add_rolling_features(
    X_base,
    window=rolling_window,
    compression_ratio=run.compression_ratio,
    max_new_tokens=run.max_new_tokens,
)
```

### Step 5: Update existing tests that check token_position range

In `tests/test_features.py::TestRollingFeatures::test_token_position_range` — update to account for new denominator. The position at the last token is now `(n-1) / max_new_tokens`, not `1.0`.

### Step 6: Run tests

```bash
uv run pytest tests/test_features.py -v
```

Expected: all pass

### Step 7: Commit

```bash
git add src/kvguard/features.py tests/test_features.py
git commit -m "fix: token_position uses max_new_tokens denominator, not trace length

Prevents leaking actual sequence length (unavailable at inference time).
Fixes VALIDATION_REPORT.md issue 1.5."
```

---

## Task 4: Remove RECOVERY Mode and anchors.py (Dead Code)

**Validation Report refs:** 4.1, 4.2, 4.3, 4.4
**Why:** RECOVERY mode is defined but functionally identical to SAFE (anchor recomputation is never wired in). anchors.py is 274 lines of dead code. The paper describes features that don't exist. Be honest about what's evaluated.

**Files:**

- Delete: `src/kvguard/anchors.py`
- Delete: `tests/test_anchors.py`
- Modify: `src/kvguard/controller.py:30-36` (Mode enum), `controller.py:62-86` (ControllerConfig), `controller.py:169-209` (_decide_mode), `controller.py:212-332` (RiskController)
- Modify: `src/kvguard/evaluate_controller.py:137-211` (_run_predictor_state_machine)
- Test: `tests/test_controller.py`

### Step 1: Remove RECOVERY from Mode enum

In `src/kvguard/controller.py:30-36`:

```python
class Mode(IntEnum):
    """Controller operating modes, ordered by severity."""
    NORMAL = 0
    ALERT = 1
    SAFE = 2
```

### Step 2: Remove `rep_count_recovery` from ControllerConfig

In `src/kvguard/controller.py:62-86`, remove `rep_count_recovery: int = 3`.

### Step 3: Remove RECOVERY transitions from `_decide_mode`

In `src/kvguard/controller.py:169-209`, remove:
- Line ~188-189: `elif mode == Mode.SAFE and rc >= config.rep_count_recovery: return Mode.RECOVERY`
- Line ~192-193: `if mode == Mode.RECOVERY and consecutive_low >= j: return Mode.SAFE`
- The `rep_count` parameter from the function signature

### Step 4: Remove RECOVERY from `RiskController.step()` and `_action_compression_ratio()`

In `controller.py:323-332`, remove the RECOVERY case.

### Step 5: Remove RECOVERY from `_run_predictor_state_machine` in evaluate_controller.py

In `evaluate_controller.py:137-211`, remove:
- Line ~188-189: `elif mode == Mode.SAFE and rc >= config.rep_count_recovery: new_mode = Mode.RECOVERY`
- Line ~192: `if mode == Mode.RECOVERY and consecutive_low >= j: new_mode = Mode.SAFE`
- The `recovery_trigger` tracking
- The `rep_counts` parameter from the signature

### Step 6: Delete anchors.py and test_anchors.py

```bash
rm src/kvguard/anchors.py tests/test_anchors.py
```

### Step 7: Update tests in test_controller.py

Remove:
- `test_safe_to_recovery_on_rep_count` (line ~186)
- `test_recovery_to_safe_deescalate` (line ~198)
- `test_recovery_triggered_by_rep_count` (line ~281)
- Any test that references `Mode.RECOVERY` or `rep_count_recovery`

Update:
- `test_int_values` — SAFE is now the highest mode (value 2)
- `_decide_mode` tests — remove rep_count parameter

### Step 8: Run tests

```bash
uv run pytest tests/ -v
```

Expected: all pass (anchors tests are deleted, controller tests updated)

### Step 9: Commit

```bash
git add -A
git commit -m "fix: remove RECOVERY mode and anchors.py (dead code)

Controller is now 3 modes: NORMAL → ALERT → SAFE.
RECOVERY was never wired into evaluation. anchors.py was never imported.
Fixes VALIDATION_REPORT.md issues 4.1, 4.2, 4.3, 4.4."
```

---

## Task 5: Add Feature Ablation Infrastructure

**Validation Report refs:** 3.1, 3.2, 3.8
**Why:** No ablation exists for dropping `compression_ratio` or `rep_count`. We can't validate that logit features actually predict catastrophe vs. the predictor being a lookup table.

**Files:**

- Modify: `src/kvguard/train.py` (add `exclude_features` to `train_predictor` and `build_dataset`)
- Modify: `src/kvguard/features.py` (add `Dataset.drop_features()` helper)
- Modify: `scripts/run_ablations.py` (add feature ablation configs)
- Test: `tests/test_train.py`

### Step 1: Write failing test for feature exclusion

Add to `tests/test_train.py`:

```python
class TestFeatureExclusion:
    """Tests for dropping features before training."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ds = _make_synthetic_dataset(n_prompts=10, catastrophe_fraction=0.3)

    def test_drop_compression_ratio(self):
        ds = self.ds.drop_features(["compression_ratio"])
        assert "compression_ratio" not in ds.feature_names
        assert ds.X.shape[1] == self.ds.X.shape[1] - 1

    def test_drop_rep_features(self):
        ds = self.ds.drop_features(["rep_count", "rep_count_sum_8"])
        assert "rep_count" not in ds.feature_names
        assert "rep_count_sum_8" not in ds.feature_names
        assert ds.X.shape[1] == self.ds.X.shape[1] - 2

    def test_drop_nonexistent_raises(self):
        with pytest.raises(KeyError):
            self.ds.drop_features(["nonexistent_feature"])

    def test_drop_preserves_other_data(self):
        ds = self.ds.drop_features(["compression_ratio"])
        assert len(ds.traces) == len(self.ds.traces)
        np.testing.assert_array_equal(ds.y, self.ds.y)
        np.testing.assert_array_equal(ds.trace_ids, self.ds.trace_ids)
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/test_train.py::TestFeatureExclusion -v
```

Expected: `AttributeError: 'Dataset' object has no attribute 'drop_features'`

### Step 3: Implement `Dataset.drop_features()`

In `src/kvguard/features.py`, add a method to the `Dataset` dataclass (~line 193-201):

```python
@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    trace_ids: np.ndarray
    feature_names: list[str]
    traces: list[TraceMeta]

    def drop_features(self, names: list[str]) -> Dataset:
        """Return a new Dataset with specified feature columns removed."""
        indices_to_drop = []
        for name in names:
            if name not in self.feature_names:
                msg = f"Feature '{name}' not in dataset: {self.feature_names}"
                raise KeyError(msg)
            indices_to_drop.append(self.feature_names.index(name))

        keep = [i for i in range(self.X.shape[1]) if i not in indices_to_drop]
        return Dataset(
            X=self.X[:, keep],
            y=self.y.copy(),
            trace_ids=self.trace_ids.copy(),
            feature_names=[self.feature_names[i] for i in keep],
            traces=list(self.traces),
        )
```

### Step 4: Run tests

```bash
uv run pytest tests/test_train.py::TestFeatureExclusion -v
```

Expected: all pass

### Step 5: Add predefined ablation sets to `run_ablations.py`

In `scripts/run_ablations.py`, add a new function after the existing ablations:

```python
FEATURE_ABLATIONS: dict[str, list[str]] = {
    "full": [],
    "no_compression_ratio": ["compression_ratio"],
    "no_rep": ["rep_count", "rep_count_sum_8"],
    "logit_only": ["compression_ratio", "rep_count", "rep_count_sum_8"],
}


def ablation_feature_importance(results_dir: Path, output_dir: Path) -> list[dict]:
    """Retrain predictor with feature subsets and compare AUROC."""
    from kvguard.features import build_dataset
    from kvguard.train import evaluate_predictor, split_by_prompt, train_predictor

    ds = build_dataset(results_dir)
    results = []

    for ablation_name, drop_features in FEATURE_ABLATIONS.items():
        ds_ablated = ds.drop_features(drop_features) if drop_features else ds
        split = split_by_prompt(ds_ablated, val_fraction=0.2)

        X_tr = ds_ablated.X[split.train_mask]
        y_tr = ds_ablated.y[split.train_mask]
        X_va = ds_ablated.X[split.val_mask]
        y_va = ds_ablated.y[split.val_mask]

        model = train_predictor(X_tr, y_tr)
        metrics = evaluate_predictor(model, X_va, y_va)

        results.append({
            "ablation": ablation_name,
            "dropped_features": drop_features,
            "n_features": ds_ablated.X.shape[1],
            "auroc": metrics.auroc,
            "f1": metrics.f1,
            "recall": metrics.recall,
        })

    return results
```

### Step 6: Commit

```bash
git add src/kvguard/features.py src/kvguard/train.py scripts/run_ablations.py tests/test_train.py
git commit -m "feat: add feature ablation infrastructure with drop_features()

Predefined sets: full, no_compression_ratio, no_rep, logit_only.
Fixes VALIDATION_REPORT.md issues 3.1, 3.2, 3.8."
```

---

## Task 6: Fix GSM8K Answer Extraction (First vs Last Match)

**Validation Report ref:** 5.1
**Why:** `re.search` takes the first `####` match. GSM8K convention is the last `####`. If the model mentions `####` in intermediate reasoning, correct answers get marked wrong.

**Files:**

- Modify: `src/kvguard/detectors.py:79-96` (`parse_gsm8k_answer`)
- Test: `tests/test_detectors.py`

### Step 1: Write failing test

Add to `tests/test_detectors.py::TestParseGsm8kAnswer`:

```python
def test_takes_last_hash_not_first(self):
    """When model mentions #### in reasoning, take the LAST one."""
    text = "The cost per item is #### 15. For 3 items: #### 45"
    assert parse_gsm8k_answer(text) == "45"

def test_takes_last_boxed_not_first(self):
    text = "First part \\boxed{10}, final \\boxed{42}"
    assert parse_gsm8k_answer(text) == "42"
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/test_detectors.py::TestParseGsm8kAnswer::test_takes_last_hash_not_first -v
```

Expected: FAIL (returns "15" instead of "45")

### Step 3: Change `re.search` to `re.findall` and take last match

In `src/kvguard/detectors.py:79-96`:

```python
def parse_gsm8k_answer(text: str) -> str | None:
    """Extract the final numeric answer from model output.

    Supports two formats:
    - GSM8K canonical: #### NUMBER
    - LaTeX boxed (instruction-tuned models): \\boxed{NUMBER}

    Takes the LAST match (model may reference #### in intermediate reasoning).
    """
    # Try #### format — take last match
    matches = re.findall(r"####\s*([\d,.\-]+)", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    # Try \\boxed{NUMBER} format — take last match
    matches = re.findall(r"\\boxed\{([\d,.\-]+)\}", text)
    if matches:
        return matches[-1].replace(",", "").strip()

    return None
```

### Step 4: Run tests

```bash
uv run pytest tests/test_detectors.py -v
```

Expected: all pass

### Step 5: Commit

```bash
git add src/kvguard/detectors.py tests/test_detectors.py
git commit -m "fix: take last #### match in GSM8K answer extraction

re.search → re.findall with [-1] to match GSM8K convention.
Fixes VALIDATION_REPORT.md issue 5.1."
```

---

## Task 7: Add Controller Evaluation Holdout

**Validation Report refs:** 1.1 (Tier 1), statistical claims issue 1 and 7
**Why:** The controller evaluation runs the trained predictor on ALL 800 traces including the ~80% used for training. Headline CFR numbers are inflated. Additionally, threshold selection is done on the same data.

**Files:**

- Modify: `src/kvguard/evaluate_controller.py:331-445` (`evaluate_controller`)
- Modify: `src/kvguard/train.py:353-429` (`run_training` — persist split info)
- Test: `tests/test_evaluate_controller.py` (new file)

### Step 1: Write failing test for holdout-only evaluation

Create `tests/test_evaluate_controller.py`:

```python
"""Tests for controller evaluation module."""
import numpy as np
import pytest

from kvguard.evaluate_controller import filter_traces_by_prompts


def test_filter_traces_by_prompts():
    """Only traces whose prompt_id is in the allowed set are returned."""
    # This function doesn't exist yet — will fail
    all_traces = {
        ("snapkv", 0.75): {
            "gsm8k_0": "trace_a",
            "gsm8k_1": "trace_b",
            "gsm8k_2": "trace_c",
        }
    }
    allowed = {"gsm8k_0", "gsm8k_2"}
    filtered = filter_traces_by_prompts(all_traces, allowed)
    assert set(filtered[("snapkv", 0.75)].keys()) == {"gsm8k_0", "gsm8k_2"}


def test_filter_preserves_structure():
    all_traces = {
        ("snapkv", 0.5): {"gsm8k_0": "t1", "gsm8k_1": "t2"},
        ("snapkv", 0.75): {"gsm8k_0": "t3", "gsm8k_1": "t4"},
    }
    allowed = {"gsm8k_1"}
    filtered = filter_traces_by_prompts(all_traces, allowed)
    assert len(filtered) == 2
    for key in filtered:
        assert list(filtered[key].keys()) == ["gsm8k_1"]
```

### Step 2: Run to verify failure

```bash
uv run pytest tests/test_evaluate_controller.py -v
```

Expected: `ImportError: cannot import name 'filter_traces_by_prompts'`

### Step 3: Implement `filter_traces_by_prompts`

In `src/kvguard/evaluate_controller.py`, add after `load_all_traces`:

```python
def filter_traces_by_prompts(
    all_traces: dict[tuple[str, float], dict[str, TraceInfo]],
    prompt_ids: set[str],
) -> dict[tuple[str, float], dict[str, TraceInfo]]:
    """Filter traces to only include specified prompt_ids.

    Used to evaluate the controller only on held-out prompts that were
    not used for predictor training.
    """
    return {
        key: {pid: trace for pid, trace in traces.items() if pid in prompt_ids}
        for key, traces in all_traces.items()
    }
```

### Step 4: Update `evaluate_controller` to accept `holdout_prompt_ids`

In `src/kvguard/evaluate_controller.py:331-445`, add parameter:

```python
def evaluate_controller(
    results_dir: Path,
    predictor: xgb.XGBClassifier,
    *,
    num_prompts: int = 50,
    controller_config: ControllerConfig | None = None,
    rolling_window: int = ROLLING_WINDOW,
    holdout_prompt_ids: set[str] | None = None,
) -> EvalResult:
```

After line 361 (`all_traces = load_all_traces(...)`), add:

```python
    if holdout_prompt_ids is not None:
        all_traces = filter_traces_by_prompts(all_traces, holdout_prompt_ids)
```

### Step 5: Save split info in `run_training`

In `src/kvguard/train.py:353-429`, after saving the model, also save the val prompt_ids:

```python
    # In run_training, after saving model and metrics:
    if output_dir:
        val_prompt_ids = {ds.traces[t].prompt_id for t in split.val_traces}
        split_path = output_dir / "split_info.json"
        split_path.write_text(json.dumps({
            "val_prompt_ids": sorted(val_prompt_ids),
            "train_prompt_ids": sorted(
                {ds.traces[t].prompt_id for t in split.train_traces}
            ),
            "n_val_traces": len(split.val_traces),
            "n_train_traces": len(split.train_traces),
        }, indent=2))
```

### Step 6: Update CLI `eval-controller` command to load holdout info

In `src/kvguard/__init__.py`, the `eval_controller` CLI command (~line 136), add logic to load `split_info.json` from the model directory and pass `holdout_prompt_ids`:

```python
    # Load holdout split info if available
    split_info_path = model_path.parent / "split_info.json"
    holdout_prompt_ids = None
    if split_info_path.exists():
        import json
        split_info = json.loads(split_info_path.read_text())
        holdout_prompt_ids = set(split_info["val_prompt_ids"])
        logger.info(f"Evaluating on {len(holdout_prompt_ids)} held-out prompts")
```

### Step 7: Run tests

```bash
uv run pytest tests/test_evaluate_controller.py -v
```

Expected: all pass

### Step 8: Commit

```bash
git add src/kvguard/evaluate_controller.py src/kvguard/train.py src/kvguard/__init__.py tests/test_evaluate_controller.py
git commit -m "feat: controller evaluation respects train/test split

evaluate_controller() now accepts holdout_prompt_ids to evaluate only
on prompts the predictor never saw during training.
run_training() saves split_info.json with prompt partition.
Fixes VALIDATION_REPORT.md issues 1.1 (Tier 1), stat claims 1 & 7."
```

---

## Task 8: Remove Unused Imports and Clean Up evaluate_controller.py

**Validation Report refs:** 5.13
**Why:** 6+ unused imports. Clean code hygiene.

**Files:**

- Modify: `src/kvguard/evaluate_controller.py:13-36`

### Step 1: Remove unused imports

In `src/kvguard/evaluate_controller.py`, remove these unused imports:

- Line 15: `import json`
- Line 23: `from kvguard.config import RunResult` (if unused)
- Line 24: `RiskController` (if unused — only in comment)
- Lines 26-31: `N_BASE`, `ROLLING_COL_INDICES`, `ROLLING_COLS`, `ROLLING_STATS`, `_rolling_stat`

Keep only what's actually used: `xgb`, `np`, `Path`, `dataclass`, `field`, `Mode`, `ControllerConfig`, `ROLLING_WINDOW`, `feature_names`, `flatten_signals`, `add_rolling_features`, `compute_hazard_labels`, `compute_repetition_counts`.

### Step 2: Run `make check`

```bash
cd /Users/joaquincamponario/Documents/INCO/RESEARCH/kvguard && make check
```

Expected: ruff, mypy, pytest all pass

### Step 3: Commit

```bash
git add src/kvguard/evaluate_controller.py
git commit -m "chore: remove 8 unused imports from evaluate_controller.py"
```

---

## Task 9: Remove `from __future__ import annotations` Violations

**Validation Report ref:** 5.14
**Why:** Project convention in CLAUDE.md says "NOT used — we target Python 3.12+." Six files violate this.

**Files:**

- Modify: `src/kvguard/controller.py:19`
- Modify: `src/kvguard/features.py:7` (if present)
- Modify: `src/kvguard/train.py:8` (if present)
- Modify: `src/kvguard/evaluate_controller.py:13`
- Modify: `src/kvguard/verify.py:11`
- (anchors.py already deleted in Task 4)

### Step 1: Remove `from __future__ import annotations` from each file

Remove the import line. Verify that no runtime behavior depends on it (since we target 3.12+, `X | None` syntax works natively).

### Step 2: Run `make check`

```bash
make check
```

Expected: all pass

### Step 3: Commit

```bash
git add src/kvguard/controller.py src/kvguard/features.py src/kvguard/train.py src/kvguard/evaluate_controller.py src/kvguard/verify.py
git commit -m "chore: remove from __future__ import annotations (targets Python 3.12+)"
```

---

## Task 10: Remove `wandb` Phantom Dependency

**Validation Report ref:** 5.12
**Why:** Declared in `pyproject.toml` but never imported anywhere.

**Files:**

- Modify: `pyproject.toml:21`

### Step 1: Remove wandb from dependencies

In `pyproject.toml`, remove the line `"wandb>=0.25.0",` from the `dependencies` list.

### Step 2: Run `uv sync --group dev && make check`

```bash
uv sync --group dev && make check
```

Expected: all pass

### Step 3: Commit

```bash
git add pyproject.toml
git commit -m "chore: remove unused wandb dependency"
```

---

## Task 11: Fix Controller Default Config to Match Paper

**Validation Report ref:** 5.8
**Why:** `ControllerConfig()` defaults to tau_high=0.6, k_escalate=3, safe_compression_ratio=0.5 — none of which match any operating point in the paper. The CLI `eval-controller` also defaults to tau_high=0.6.

**Files:**

- Modify: `src/kvguard/controller.py:77-86` (ControllerConfig defaults)
- Modify: `src/kvguard/__init__.py:136-185` (CLI defaults)
- Test: `tests/test_controller.py`

### Step 1: Update defaults to match paper's Balanced config

In `src/kvguard/controller.py:77-86`:

```python
    tau_low: float = 0.3
    tau_high: float = 0.7      # was 0.6
    k_escalate: int = 8        # was 3
    j_deescalate: int = 5
    base_compression_ratio: float = 0.875
    safe_compression_ratio: float = 0.0  # was 0.5
```

### Step 2: Update CLI defaults

In `src/kvguard/__init__.py`, update the `eval_controller` command's `typer.Option` defaults:
- `tau_high`: 0.6 → 0.7
- `safe_ratio`: 0.5 → 0.0
- `k_escalate`: (check current) → 8

### Step 3: Update tests

In `tests/test_controller.py::TestControllerConfig::test_defaults_sensible`, update expected values.

### Step 4: Run tests

```bash
uv run pytest tests/test_controller.py -v
```

### Step 5: Commit

```bash
git add src/kvguard/controller.py src/kvguard/__init__.py tests/test_controller.py
git commit -m "fix: controller defaults match paper's Balanced config

tau_high=0.7, k_escalate=8, safe_compression_ratio=0.0.
Fixes VALIDATION_REPORT.md issue 5.8."
```

---

## Task 12: Add `delta_h_valid` Deprecation Note

**Validation Report ref:** 3.7
**Why:** `delta_h_valid` is 0 only for the first token of each trace, and 1 for all others. It's near-constant and carries no information for XGBoost.

**Files:**

- Modify: `src/kvguard/features.py:24-36`

### Step 1: Keep the feature but add a comment

This is low-priority — removing it would change the feature count again and require updating all downstream code. For now, just document it:

```python
    "delta_h_valid",  # NOTE: near-constant (0 only at t=0, 1 elsewhere). Low signal.
```

### Step 2: Commit

```bash
git add src/kvguard/features.py
git commit -m "docs: note delta_h_valid is near-constant (low signal)"
```

---

## Task 13: Duplicate State Machine Code Consolidation

**Validation Report ref:** 4.3 (duplicated state machine in evaluate_controller.py)
**Why:** `_run_predictor_state_machine()` reimplements the `RiskController` logic. Two implementations of the same thing with no cross-validation test.

**Files:**

- Modify: `src/kvguard/evaluate_controller.py:137-211`
- Test: `tests/test_evaluate_controller.py`

### Step 1: Write a test that both implementations agree

Add to `tests/test_evaluate_controller.py`:

```python
def test_state_machines_agree():
    """_run_predictor_state_machine and RiskController produce same mode history."""
    from kvguard.controller import ControllerConfig, RiskController, compute_risk_score
    from kvguard.evaluate_controller import _run_predictor_state_machine

    config = ControllerConfig(tau_low=0.3, tau_high=0.7, k_escalate=3, j_deescalate=3)
    hazard_probs = [0.1, 0.1, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1]

    # Run the standalone state machine
    mode_history_standalone, _, _ = _run_predictor_state_machine(hazard_probs, config)

    # Run through RiskController using hazard_probs as pre-computed risk scores
    ctrl = RiskController(config)
    mode_history_ctrl = []
    for p in hazard_probs:
        # Feed hazard_prob directly as risk_score by using a synthetic signals dict
        action = ctrl.step({"__hazard_prob": p})
        mode_history_ctrl.append(action.mode.value)

    assert mode_history_standalone == mode_history_ctrl
```

Note: This test may reveal that the two implementations actually diverge. If they do, that's a finding worth documenting. The fix would be to refactor `_run_predictor_state_machine` to use `RiskController` internally, or at minimum add this test as a regression guard.

If the test passes as-is, good. If not, the divergence is the bug to fix.

### Step 2: Commit

```bash
git add tests/test_evaluate_controller.py
git commit -m "test: add cross-validation test for duplicate state machine implementations"
```

---

## Task 14: Add `numpy` to Explicit Dependencies

**Validation Report ref:** 5.12 (secondary)
**Why:** numpy is used directly in 4+ source files but not declared in pyproject.toml. It's pulled transitively through torch/xgboost but explicit is better.

**Files:**

- Modify: `pyproject.toml`

### Step 1: Add numpy to dependencies

In `pyproject.toml`, add `"numpy>=1.26.0",` to the dependencies list.

### Step 2: Run `uv sync && make check`

### Step 3: Commit

```bash
git add pyproject.toml
git commit -m "chore: add numpy as explicit dependency"
```

---

## Additional Tasks (from second agent review)

### Task 15: Multi-Model Support in TraceMeta
**Why:** Need `model` field in TraceMeta to filter/split by model for multi-model experiments.
- Modify: `src/kvguard/features.py` — add `model: str = ""` to `TraceMeta`, populate from `run.model` in `build_dataset()`
- Test: `tests/test_features.py` — assert TraceMeta.model is populated

### Task 16: CUDA Device Cleanup in Experiment Runner
**Why:** `run_prompts()` only has MPS cleanup. Need CUDA equivalent for 5090.
- Modify: `src/kvguard/experiment.py:296-299` — add CUDA cleanup path alongside MPS
- Also fix: `run_sweep()` lines 453-455 — add MPS cleanup (currently only CUDA)

### Design Note: RECOVERY Mode Removal
Keep `Mode.RECOVERY = 3` in the enum to avoid breaking serialized mode_history data in existing traces. Just make it unreachable in the state machine. Add docstring: "Reserved for future KV recomputation; not reachable via current state machine."

---

## Execution Order

```
Task 2  (remove rank_of_chosen)  — first, changes feature schema others depend on
Task 1  (prompt-level split)     — no dependencies
Task 4  (remove RECOVERY/anchors) — no dependencies
Task 3  (fix token_position)     — no dependencies
Task 6  (fix answer extraction)  — no dependencies
Task 5  (feature ablation)       — depends on Task 1
Task 7  (controller holdout)     — depends on Task 1
Task 15 (multi-model TraceMeta)  — no dependencies
Task 16 (CUDA cleanup)           — no dependencies
Task 11 (fix config defaults)    — depends on Task 4
Task 8  (unused imports)         — depends on Task 4
Task 9  (__future__ annotations) — no dependencies
Task 10 (remove wandb)           — no dependencies
Task 12 (delta_h_valid note)     — after Task 2
Task 13 (state machine test)     — depends on Task 4
Task 14 (numpy dependency)       — no dependencies
```

---

## Verification Checklist

After all tasks are done, run:

```bash
cd /Users/joaquincamponario/Documents/INCO/RESEARCH/kvguard
make check
```

This runs `ruff format` + `ruff check` + `mypy` + `pytest`. All must pass.

Then verify:

- [ ] `split_by_prompt` exists and is used by `run_training`
- [ ] No prompt_id appears in both train and val
- [ ] `rank_of_chosen` is not in `BASE_FEATURE_NAMES`
- [ ] Feature count is 40 (29 base + 8 rolling + rep_count_sum + token_position + compression_ratio)
- [ ] `token_position = t / max_new_tokens` (not `t / (n-1)`)
- [ ] `Mode` enum has 3 values (NORMAL, ALERT, SAFE)
- [ ] `anchors.py` and `test_anchors.py` are deleted
- [ ] `Dataset.drop_features()` works
- [ ] `parse_gsm8k_answer` takes last match
- [ ] `evaluate_controller` accepts `holdout_prompt_ids`
- [ ] `split_info.json` is saved by `run_training`
- [ ] No unused imports in evaluate_controller.py
- [ ] No `from __future__ import annotations` in any source file
- [ ] No `wandb` in pyproject.toml dependencies
- [ ] Controller defaults: tau_high=0.7, k_escalate=8, safe_ratio=0.0
- [ ] numpy is in pyproject.toml dependencies
