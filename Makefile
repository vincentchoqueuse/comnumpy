# Define the Python interpreter
PYTHON ?= python3

# Define directories
DOCS_DIR = docs
TESTS_DIR = tests
SRC_DIR = src

# Run all tests
test:
	$(PYTHON) -m unittest discover $(TESTS_DIR)

# Build the documentation
docs:
	make -C $(DOCS_DIR) html

# Clean up build artifacts
clean:
	rm -rf $(DOCS_DIR)/_build/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

# Build the project
build:
	$(PYTHON) setup.py sdist bdist_wheel

# Install the project in editable mode
install:
	$(PYTHON) -m pip install -e .

# Run linting and formatting checks
lint:
	flake8 $(SRC_DIR) $(TESTS_DIR)

.PHONY: test docs clean build install lint