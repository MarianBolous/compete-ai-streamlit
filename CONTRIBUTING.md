# Contributing to CompeteAI

Thank you for considering contributing to CompeteAI! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read and understand its contents before contributing.

We are committed to making participation in this project a harassment-free experience for everyone, regardless of level of experience, gender, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, or religion.

Examples of unacceptable behavior include the use of sexualized language or imagery, derogatory comments or personal attacks, trolling, public or private harassment, insults, or other unprofessional conduct.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/compete-ai-streamlit.git
   cd compete-ai-streamlit
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```
4. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. **Make your changes** in the appropriate files
2. **Run the linting tools** to ensure code quality:
   ```bash
   flake8 .
   black .
   isort .
   ```
3. **Run tests** to ensure functionality:
   ```bash
   pytest
   ```
4. **Update documentation** as needed
5. **Commit your changes** with a descriptive commit message:
   ```bash
   git commit -m "feat: add new feature X"
   ```
   We follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages.

## Pull Request Process

1. **Update your fork** with the latest changes from the main repository:
   ```bash
   git remote add upstream https://github.com/MarianBolous/compete-ai-streamlit.git
   git fetch upstream
   git rebase upstream/main
   ```
2. **Push your branch** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
3. **Create a Pull Request** through the GitHub interface
4. **Address review comments** if any are provided

## Coding Standards

We adhere to the following coding standards:

- Follow [PEP 8](https://pep8.org/) style guide for Python code
- Use [Black](https://black.readthedocs.io/) for code formatting
- Sort imports using [isort](https://pycqa.github.io/isort/)
- Add type hints to function signatures
- Write docstrings for all functions, classes, and modules
- Use descriptive variable names

## Testing

- Write unit tests for all new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage of new code

## Documentation

- Update documentation for any changed functionality
- Document new features thoroughly
- Keep the README.md up to date

## Issue Reporting

If you find a bug or have a suggestion for improvement:

1. Check if the issue already exists in the [GitHub Issues](https://github.com/MarianBolous/compete-ai-streamlit/issues)
2. If not, create a new issue with a descriptive title and detailed information
3. Include:
   - Steps to reproduce the issue
   - Expected behavior
   - Actual behavior
   - Screenshots if applicable
   - Environment information (OS, Python version, etc.)

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license.

---

Thank you for contributing to CompeteAI! Your efforts help make this project better for everyone.