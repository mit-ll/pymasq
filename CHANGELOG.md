# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

