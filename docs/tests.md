# Testing Guide

This document provides information on how to run the tests for the Supply Chain Forecaster system.

## Test Structure

The test suite is organized into the following categories:

- **Unit Tests**: Tests for individual functions and methods
- **Integration Tests**: Tests for interactions between components
- **API Tests**: Tests for API endpoints
- **End-to-End Tests**: Tests for entire workflows

## Running Tests

### Running All Tests

```bash
pytest tests/
```

### Running with Coverage

```bash
pytest --cov=./ --cov-report=term --cov-report=html tests/
```

### Running Specific Test Categories

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# API tests
pytest tests/api/

# End-to-end tests
pytest tests/e2e/
```

### Running a Specific Test File

```bash
pytest tests/unit/test_models.py
```

### Running a Specific Test Function

```bash
pytest tests/unit/test_models.py::test_prophet_model
```

## Test Configuration

Tests use a separate configuration defined in `tests/conftest.py`. This configuration ensures that tests don't interfere with your development or production environment.

## Writing Tests

When adding new features, please follow these guidelines for writing tests:

- **Unit Tests**: Create tests for individual functions and methods in the `tests/unit/` directory.
- **Integration Tests**: Create tests for interactions between components in the `tests/integration/` directory.
- **API Tests**: Create tests for API endpoints in the `tests/api/` directory.
- **End-to-End Tests**: Create tests for entire workflows in the `tests/e2e/` directory.

## Test Data

Test data is stored in the `tests/data/` directory. This includes CSV files and other resources needed for testing.

## Mock Objects

Mock objects are used to simulate dependencies in unit tests. These are defined in the `tests/mocks/` directory.

## Continuous Integration

The test suite is automatically run on GitHub Actions when code is pushed to the repository. See the `.github/workflows/ci.yml` file for details.

## Next Steps

After setting up and running the tests, you can proceed to the [Installation Guide](installation/installation.md) to learn how to install and configure the system.