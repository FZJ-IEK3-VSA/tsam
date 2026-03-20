# Installation

It is recommended to install tsam within its own environment. If you are not familiar with python environments, please consider reading some [external documentation](https://realpython.com/python-virtual-environments-a-primer/).

**Quick Install (Recommended)**

The fastest way to install tsam is using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install tsam
```

**Alternative Installation Methods**

Using pip:

```bash
pip install tsam
```

Using conda-forge:

```bash
conda install tsam -c conda-forge
```

**Creating an Isolated Environment**

With uv (recommended):

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install tsam
```

With conda/mamba:

```bash
mamba create -n tsam_env python pip
mamba activate tsam_env
pip install tsam
```

## Local Installation for Development

Clone the repository:

```bash
git clone https://github.com/FZJ-IEK3-VSA/tsam.git
cd tsam
```

**Using uv (Recommended)**

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[develop]"
```

**Using conda**

```bash
conda env create --file=environment.yml
conda activate tsam_dev
pip install -e ".[develop]"
```

## Development Tools

tsam uses modern Python development tools for code quality:

**Linting and Formatting with Ruff**

[Ruff](https://docs.astral.sh/ruff/) is used for fast linting and formatting:

```bash
# Check for issues
ruff check src/ test/

# Auto-fix issues
ruff check src/ test/ --fix

# Format code
ruff format src/ test/
```

**Type Checking with Mypy**

[Mypy](https://mypy.readthedocs.io/) is used for static type checking:

```bash
mypy src/tsam/
```

**Pre-commit Hooks**

Pre-commit hooks automatically run linting and formatting on every commit:

```bash
# Install pre-commit
uv pip install pre-commit

# Set up hooks (run once after cloning)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

**Running Tests**

Tests are run using pytest:

```bash
# Run all tests
uv run pytest test/

# Run tests with coverage
uv run pytest test/ --cov=tsam

# Run tests in parallel
uv run pytest test/ -n auto
```

## Installation of an Optimization Solver

Some clustering algorithms in tsam are based on Mixed-Integer Linear Programming. An appropriate solver accessible by [Pyomo](https://github.com/Pyomo/pyomo/) is required.

**HiGHS (Default)**

[HiGHS](https://github.com/ERGO-Code/HiGHS) is installed by default and works well for most use cases.

**Commercial Solvers**

For better performance on large problems, commercial solvers are recommended if you have access to a license:

* [Gurobi](https://www.gurobi.com/)
* [CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio)
