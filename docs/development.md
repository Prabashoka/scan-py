# Development

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip maturin
python -m pip install -e ".[test]"
maturin develop --release
```

## Test

```bash
python -m pytest tests
```

## Lint

```bash
python -m ruff check .
```

## Build

```bash
maturin build --release
```

## Project Layout

```text
src/                 Rust/PyO3 backend
python/scan/         Python package
python/scan/detector.py
python/scan/result.py
python/scan/plotting.py
python/scan/metrics.py
python/scan/simulator.py
tests/               Pytest suite
docs/                Markdown documentation
examples/            Script examples
```

## Notes

- Keep the Python import name as `scan` even though the PyPI distribution name is `scan-py`.
- Run `maturin develop --release` after Rust changes so Python imports the rebuilt extension.
- Public functions should be exported from `python/scan/__init__.py`.