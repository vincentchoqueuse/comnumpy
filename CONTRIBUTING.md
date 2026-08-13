# Contributing to comnumpy

We welcome contributions! Whether you're fixing a bug, improving the documentation, or developing a new submodule, your input is valuable.

## Before you start: the two normative documents

[`ARCHITECTURE.md`](ARCHITECTURE.md) is the decision record — it is normative:
the code conforms to a decision `Dxx` or amends it explicitly, keeping its
number and stating the reason. Source comments that mention "decision D25" or
"D40a" refer to it. [`CONVENTIONS.md`](CONVENTIONS.md) holds what you need
day to day: tensor layouts, axis categories, error-message shape, and how to
observe a signal inside a chain (decision D42 — chains contain communication
blocks only; use `taps` and `wiring`, never a recorder or scope block).

A pull request that contradicts a decision is fine, provided it says which one
and why — that is how the document evolves.

## Setting Up for Development

Fork the repository and clone your fork:

```bash
git clone https://github.com/<your-username>/comnumpy.git
cd comnumpy
```

Create a virtual environment and install in editable mode:

```bash
conda create -n comnumpy-dev python=3.11
conda activate comnumpy-dev
pip install -e .
pip install -r requirements.txt
```

Run the tests to make sure everything works:

```bash
make test
```

### Type checking against the right NumPy

`pyright` is a blocking check, and its verdict depends on which NumPy is
installed: the 1.26 stubs require `ndarray` to carry type arguments, the
2.x stubs do not. CI lints against the newest NumPy, so an environment
pinned to the oldest supported one reports hundreds of errors that are
not real -- and, worse, can stay silent on one that is.

Keep a second environment for the check:

```bash
python -m venv .venv-lint && .venv-lint/bin/pip install -e .
pyright --pythonpath .venv-lint/bin/python
```

## Guidelines

- **Code style**: Follow PEP 8 formatting. Run `make lint` to check.
- **Tests**: Add tests for new features or fixes in the `tests/` directory.
- **Docstrings**: Use the NumPy/SciPy docstring style.
- **Pull requests**: Keep them focused and concise. One feature or fix per PR.
- **Commit messages**: Write clear, descriptive commit messages.

## Adding a New Submodule

We encourage contributors to develop self-contained submodules for new communication models or signal processing tools. A submodule typically includes:

1. A dedicated directory under `src/comnumpy/` (e.g., `src/comnumpy/mymodule/`)
2. An `__init__.py` file
3. Python modules implementing your algorithms as `Processor` subclasses
4. Unit tests in `tests/mymodule/`
5. Documentation under `docs/documentation/mymodule/` and optionally `docs/tutorials/`

Use existing submodules like `optical` or `mimo` as templates.

## Reporting Issues

If you encounter a bug or have a feature request, please [open an issue](https://github.com/vincentchoqueuse/comnumpy/issues). Include:

- A clear description of the problem or proposal
- Steps to reproduce (for bugs)
- Your Python version and operating system
