# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-11-08

### Added
- **SDC Risk Metrics - SUDA**:
  - Implemented the Special Uniques Detection Algorithm (`pymasq.metrics.suda`) calculating disclosure risk via Minimal Sample Uniques (MSUs) and factorial subset weighting, with unit tests (`tests/metrics/test_suda.py`).
- **SDC Mitigation - Rank Swapping**:
  - Implemented standard rank swapping mitigation for continuous and ordinal variables in `pymasq.mitigations.rank_swap`. Swaps records within a bounded rank window $P = \max(1, \lfloor N \cdot p \rfloor)$.
  - Supports both fraction (`0 < p <= 1`) and percentage (`1 < p <= 100`) parameter scales.
  - Strictly preserves column multisets, sums, and means while breaking exact record linkages.
  - Handled NaN positions defensively, added seed determinism, and added unit test coverage (`tests/mitigations/test_rank_swap.py`).
- **Security & Data Privacy**:
  - Added path traversal protection in `pymasq.datasets.load_data` using `Path.resolve()` and `target_path.is_relative_to(base_dir)`.
  - Upgraded cache HMAC integrity validation in `pymasq.utils.cache` from MD5 to SHA-256 with fallback verification for legacy caches and timing-attack resistant `hmac.compare_digest`.
  - Added support for `PYMASQ_CACHE_KEY` and `PYMASQ_CACHE_DIR` environment variables for secure cache configuration.
  - Eliminated plain-text dumps of raw dataset rows in cache description metadata headers.
  - Restricted cache clearing to `.pkl` files to avoid recursive directory tree deletion.
- **Modular Dependency Extras**:
  - Split heavy optional dependencies in `pyproject.toml` into granular extras: `[dl]`, `[automl]`, `[sa]`, `[nlp]`, `[stats]`, and `[all]`.
  - Added lazy import guarding with informative `ImportError` messages when optional packages are not installed.

### Changed
- **Vectorization & Performance Optimizations**:
  - Optimized `indiv_risk` calculation by computing risk scores across unique frequency pairs and eliminating expensive whole-DataFrame self-merge.
  - Vectorized Jensen-Shannon divergence (`pymasq.metrics.jensen_shannon`) using `np.histogram(..., bins=bins)` instead of pandas cut/merge loops.
  - Vectorized equivalence class calculation in `pymasq.utils.freq_calc` and `pymasq.mitigations.utils.freq_calc` via `transform("size")`, eliminating memory allocations and guaranteeing row-index alignment.
  - Vectorized cell-by-cell hashing (`pymasq.mitigations.hashing`) by pre-hashing unique values and mapping back via `.map()`.
  - Vectorized PRAM transition probability scaling and index selection (`pymasq.mitigations.pram`).
  - Precomputed 2D rotation planes outside the column loop in geometric transformations (`pymasq.mitigations.geom_transform`).
  - Cached inspected metric signatures in `self._metric_signatures` and avoided redundant copies of `target` in the optimizer evaluation loop (`pymasq.optimizations._base`), yielding ~27% speedup.
- **Documentation & Sphinx Build Fixes**:
  - Fixed Sphinx toctree inclusion in `docs/source/modules.rst` ensuring `pymasq` is correctly rendered in table of contents.
  - Resolved Sphinx docstring formatting errors (unexpected indentation, definition lists, unclosed backticks, footnote targets) across `pymasq.kve`, `pymasq.metrics`, `pymasq.mitigations`, `pymasq.optimizations`, `pymasq.preprocessing`, and `pymasq.datasets`.
  - Upgraded warning messages logged via `logger.info("WARNING: ...")` to proper `logger.warning(...)` across `pymasq.mitigations.shuffle` and `pymasq.optimizations._base`.
- **Defensive Edge-Case Hardening**:
  - Added list length validation in `pymasq.mitigations.substitute` to raise `InputError` if `from_val` and `to_val` lists have unequal lengths (> 1).
  - Raised `NotInRangeError` for out-of-bounds probability in `pymasq.datasets.rand_cat_change`.
  - Eliminated caller DataFrame and Index mutation in `_reverse_map` in `pymasq.mitigations.shuffle`.
  - Hardened `is_identical` / `_is_identical` to safely handle empty Series and Series containing `NaN` values without `IndexError` or false negatives.
  - Added input validation for missing column lookups in `as_dataframe`, raising `InputError`.
  - Updated `preprocess.py` to include `"string"` alongside `"object", "category"` to comply with Pandas 3.0+ migration.
  - Set explicit `scoring="accuracy"` and `l1_ratios=(0.0,)` in `LogisticRegressionCV` to preempt Scikit-Learn 1.10+ deprecations.

### Fixed
- **SDC Risk Metrics Mathematical Operators & Typing**:
  - Corrected bitwise XOR (`^`) to exponentiation (`**`) across 8 equations in `indiv_risk_approx` and `indiv_risk_exact` in `pymasq.metrics.risk_scores`.
  - Added `Union[int, float]` support for `samp_freq` and `pop_freq` in `indiv_risk_exact` to prevent runtime `BeartypeCallHintParamViolation` with float64 Series inputs.
  - Fixed `is_l_diverse` list-to-int comparison bug (`len(vals) >= l`).
  - Corrected $k$-anonymity inequality condition to `val >= k`.
- **RNG Seed Unpacking**:
  - Fixed signed 32-bit unpack (`"i"`) to unsigned (`"I"`) in `set_seed()`, preventing negative seed generation.
  - Fixed timestamp token padding format in `new_token()`.
- **Plotting**:
  - Fixed invalid `hovername` keyword to `hover_name` in `px.scatter`.
- **Security in Tests**:
  - Replaced unsafe `yaml.FullLoader` with `yaml.safe_load` in integration tests.

## [1.0.1] - 2026-11-07

### Added
- **Python Compatibility**: Added official support for Python 3.10, 3.11, 3.12, and 3.13 (`requires-python = ">=3.10"`).
- **Modern CI/CD Workflows**:
  - Continuous integration via GitHub Actions testing across Python 3.10, 3.11, and 3.12 with Ruff lint checks (`.github/workflows/ci.yml`).
  - Automated documentation build and deployment to GitHub Pages (`gh-pages` branch) upon push to `main` (`.github/workflows/docs.yml`).
  - Automated package building (sdist & wheel) and release publication to PyPI with OpenID Connect (OIDC) Trusted Publishing (`.github/workflows/publish.yml`).
- **Documentation Extras**: Added `doc` optional dependency extra in `pyproject.toml` (`sphinx>=7.0`, `sphinx_rtd_theme>=2.0`, `sphinx-mdinclude>=0.5.3`).
- **API Documentation**: Added documentation module for `pymasq.models`.

### Changed
- **Packaging Modernization**: Migrated build configuration and project metadata to PEP 621 compliant `pyproject.toml`.
- **Dependency Upgrades**:
  - `pandas>=2.0`
  - `numpy>=1.26`
  - `scikit-learn>=1.2`
  - `tensorflow>=2.10`
  - `tpot>=1.1.0`
  - `SALib>=1.4`
  - `statsmodels>=0.14`
  - `matplotlib>=3.5`
  - `plotly>=4.11.0`
  - `pytest>=7.4`
  - `beartype>=0.5.1`
- **Linting & Formatting**: Adopted `ruff` as primary linter replacing deprecated `flake8`.
- **Architecture**: Modernized subpackage naming from `pymasq.classifiers` to `pymasq.models` and aligned top-level exports.

### Fixed
- **Pandas 2.0+ Compatibility**: Resolved `LossySetitemError` when assigning floating point values to integer columns during microaggregation.
- **TPOT Integration**: Resolved keyword argument deprecation (`scoring` -> `scorers`) and prevented worker exhaustion in multi-threaded environments.
- **KVE Multiprocessing**: Handled semaphore leak and disk space issues during large permutation feature exploration runs.
- **Metrics Numerical Stability**: Guarded against divide-by-zero in MAPE and fixed regression/AUC scoring edge-cases with binary target labels.
- **Type Annotations**: Corrected runtime Beartype type hints for infinite iteration limits and arbitrary keyword arguments.
- **Sphinx Documentation Build**: Fixed module resolution paths in `conf.py` and rendered Markdown README and Changelog cleanly with `sphinx-mdinclude`.

