# Contributing to comnumpy

We welcome contributions! Whether you're fixing a bug, improving the documentation, or developing a new submodule, your input is valuable.

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
5. Documentation under `docs/documentation/mymodule/` and optionally `docs/examples/`

Use existing submodules like `optical` or `mimo` as templates.

## Reporting Issues

If you encounter a bug or have a feature request, please [open an issue](https://github.com/vincentchoqueuse/comnumpy/issues). Include:

- A clear description of the problem or proposal
- Steps to reproduce (for bugs)
- Your Python version and operating system
