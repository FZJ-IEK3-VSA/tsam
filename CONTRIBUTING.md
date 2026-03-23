# Contributing to ETHOS.TSAM

Thank you for your interest in contributing to ETHOS.TSAM! This document provides guidelines and instructions for contributing.

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

We use [Conventional Commits](https://www.conventionalcommits.org/). PR titles are validated against this format and used to generate the changelog.

Common prefixes:

| Prefix | When to use | Example |
|---|---|---|
| `feat:` | New feature | `feat: add hourly resolution support` |
| `fix:` | Bug fix | `fix: correct weight normalization` |
| `docs:` | Documentation only | `docs: update installation guide` |
| `build:` | Build system / dependencies | `build: bump pandas to 2.2` |
| `ci:` | CI configuration | `ci: add Python 3.13 to matrix` |
| `refactor:` | Code change that neither fixes a bug nor adds a feature | `refactor: extract clustering logic` |
| `test:` | Adding or updating tests | `test: add segmentation edge cases` |

Use `!` after the prefix (e.g. `feat!:`) for breaking changes.

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

Documentation is built using [MkDocs](https://www.mkdocs.org/) with [Material for MkDocs](https://squidfun.github.io/mkdocs-material/) and hosted on [Read the Docs](https://tsam.readthedocs.io/).

### Building Documentation Locally

```bash
mkdocs serve
```

The documentation will be available at `http://127.0.0.1:8000/`.

## Releasing

Releases are automated via [release-please](https://github.com/googleapis/release-please) and GitHub Actions.

### Regular releases

1. Merge (squash) PRs with conventional commit titles into `develop`
2. Merge (merge) `develop` into `master`
3. release-please automatically opens a PR with version bump + CHANGELOG update
4. Merge the release-please PR → a git tag is created → package is published to PyPI

### Pre-releases

Pre-releases can be published from any branch by pushing a tag:

```bash
git tag v4.1.0-rc.1
git push origin v4.1.0-rc.1
```

This creates a GitHub Release marked as pre-release and publishes to PyPI.

### Hotfix / manual releases

Tag any commit and push it to trigger a release:

```bash
git tag v4.0.1
git push origin v4.0.1
```

Note: manual releases skip the CHANGELOG update (which is managed by release-please).

## Questions?

If you have questions, feel free to:
- Open an issue on GitHub
- Check existing issues and discussions
- Review the [documentation](https://tsam.readthedocs.io/)

## License

By contributing to ETHOS.TSAM, you agree that your contributions will be licensed under the MIT License.
