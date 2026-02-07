# Contributing Guide

Thank you for your interest in contributing to the Multi-Sensor Perception Pipeline!

## Development Setup

1. Fork and clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for all public functions and classes
- Maximum line length: 100 characters

## Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with clear commit messages
3. Add tests for new functionality
4. Update documentation if needed
5. Submit a pull request with a clear description

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
