@echo off
:: Script to set up the development environment for Supply Chain Forecaster on Windows

echo Setting up development environment for Supply Chain Forecaster...

:: Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

:: Install pre-commit hooks
echo Installing pre-commit hooks...
pre-commit install

:: Run pre-commit hooks on all files to ensure everything is clean
echo Running pre-commit hooks on all files...
pre-commit run --all-files
if %ERRORLEVEL% NEQ 0 (
    echo Some pre-commit hooks failed. You may need to fix these issues before committing.
    echo You can run 'pre-commit run --all-files' again after making changes.
)

echo Development environment setup complete!
echo.
echo To activate this environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To run tests, use:
echo   pytest
echo.
echo To check code formatting, use:
echo   black --check .
echo   isort --check --profile black .
echo.
echo Happy coding!