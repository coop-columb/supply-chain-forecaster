# Contributing to Supply Chain Forecaster

Thank you for your interest in contributing to the Supply Chain Forecaster project! This document provides guidelines and instructions for setting up your development environment and submitting contributions.

## Development Setup

### Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- pip
- Git

### Setting Up Your Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/supply-chain-forecaster.git
   cd supply-chain-forecaster
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality standards are maintained. The hooks run automatically when you commit changes, but you can also run them manually:

```bash
pre-commit run --all-files
```

The pre-commit hooks check:
- Code formatting with Black
- Import sorting with isort
- Linting with flake8
- Type checking with mypy
- YAML file validity
- Trailing whitespace and end-of-file newlines
- Debug statements
- Large files

## Code Style Guidelines

- **Black**: Code should be formatted according to Black's default style.
- **isort**: Imports should be sorted according to the Black-compatible profile.
- **Docstrings**: Use Google-style docstrings for functions and classes.
- **Type Annotations**: Use type annotations where appropriate.

## Testing

Always run tests before submitting a pull request:

```bash
pytest
```

For test coverage:

```bash
pytest --cov=./ --cov-report=term
```

## Pull Request Process

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run the pre-commit hooks on all files (`pre-commit run --all-files`)
5. Run the tests to ensure they pass (`pytest`)
6. Commit your changes with a descriptive commit message
7. Push to your branch (`git push origin feature/your-feature-name`)
8. Open a Pull Request against the main branch

## Commit Message Guidelines

Follow these guidelines for commit messages:
- Use the imperative mood (e.g., "Add feature" not "Added feature")
- Start with a type prefix: feat, fix, docs, style, refactor, test, or chore
- Include a scope in parentheses if applicable
- Keep the first line under 72 characters
- Separate the subject from the body with a blank line
- Use the body to explain what and why, not how

Example:
```
feat(dashboard): implement time series data downsampling

Implement intelligent time series downsampling to improve chart rendering 
performance. This reduces the number of points displayed when viewing 
large datasets while preserving visual fidelity.
```

## Documentation

Update documentation when making changes:
- Update README.md if you change installation or usage instructions
- Update API documentation if you change API behavior
- Add or update docstrings for functions and classes
- Create or update example files as needed

## Questions?

If you have any questions or need help, please open an issue on GitHub or contact the project maintainers.