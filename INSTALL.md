# Installation Guide

## Install with pip

```bash
pip install .
pip install .[dev]  # Also installs dev tools (pytest, black, flake8, isort, mypy)
```

## Handling `flash-attn` Installation Issues

If `flash-attn` fails due to **PEP 517 build issues**, try:

```bash
pip install --upgrade pip setuptools wheel
pip install flash-attn --no-build-isolation
```
