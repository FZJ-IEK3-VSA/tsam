# Contributing to tsam

Thank you for your interest in contributing to tsam! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setting Up Your Development Environment

1. **Clone the repository**

   ```bash
   git clone https://github.com/FZJ-IEK3-VSA/tsam.git
   cd tsam
   ```

2. **Create a virtual environment and install dependencies**

   Using uv (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[develop]"
   ```

   Using pip:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e ".[develop]"
   ```

3. **Set up pre-commit hooks**

   ```bash
   pre-commit install
   ```

   This will automatically run linting and formatting checks before each commit.

## Code Quality

We use modern Python tools to maintain code quality:

### Linting and Formatting

[Ruff](https://docs.astral.sh/ruff/) is used for both linting and formatting:

```bash
# Check for linting issues
ruff check src/ test/

# Auto-fix linting issues
ruff check src/ test/ --fix

# Format code
ruff format src/ test/
```

### Type Checking

[Mypy](https://mypy.readthedocs.io/) is used for static type checking:

```bash
mypy src/tsam/
```

### Running All Checks

You can run all checks at once using pre-commit:

```bash
pre-commit run --all-files
```

## Testing

We use [pytest](https://docs.pytest.org/) for testing:

```bash
# Run all tests
uv run pytest test/

# Run tests with coverage
uv run pytest test/ --cov=tsam

# Run tests in parallel (faster)
uv run pytest test/ -n auto

# Run a specific test file
uv run pytest test/test_averaging.py

# Run tests matching a pattern
uv run pytest test/ -k "test_k_means"
```

### Writing Tests

- Place tests in the `test/` directory
- Name test files with the `test_` prefix
- Name test functions with the `test_` prefix
- Use descriptive names that explain what is being tested

## Making Changes

### Branching Strategy

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with clear messages

3. Push your branch and create a pull request to `develop`

### Commit Messages

Write clear, concise commit messages that explain what changes were made and why:

- Use the imperative mood ("Add feature" not "Added feature")
- Keep the first line under 50 characters
- Add more detail in the body if needed

### Pull Request Guidelines

1. Ensure all tests pass
2. Ensure linting and type checks pass
3. Update documentation if needed
4. Add tests for new functionality
5. Keep pull requests focused on a single change

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) guidelines (enforced by Ruff)
- Use meaningful variable and function names
- Add docstrings to public functions and classes
- Keep functions focused and reasonably sized

## Documentation

Documentation is built using [Sphinx](https://www.sphinx-doc.org/) and hosted on [Read the Docs](https://tsam.readthedocs.io/).

### Building Documentation Locally

```bash
cd docs
make html
```

The built documentation will be in `docs/build/html/`.

## Questions?

If you have questions, feel free to:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the [documentation](https://tsam.readthedocs.io/)

## License

By contributing to tsam, you agree that your contributions will be licensed under the MIT License.
