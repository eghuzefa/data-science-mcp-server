# Contributing to Engineer Your Data

We welcome contributions from data engineers and BI professionals! This guide helps you get started.

## Getting Started

1. **Fork and clone** the repository
2. **Install dependencies**: `pip install -e .`
3. **Run tests**: `python -m pytest` (ensure all 161 tests pass)
4. **Create a branch** for your feature/fix

## Development Guidelines

### Code Standards
- Follow existing code patterns and structure
- Add type hints for new functions
- Use async/await for tool implementations
- Include comprehensive error handling

### Tool Development
- Inherit from `BaseTool` class
- Implement required methods: `name`, `description`, `get_schema()`, `execute()`
- Add tools to registry in `src/server.py`
- Use pandas for data operations where possible

### Testing Requirements
- **Write tests** for all new tools and functionality
- Follow existing test patterns in `tests/tools/`
- Ensure 100% test pass rate
- Include edge cases and error scenarios

### Documentation
- Update README.md if adding new tool categories
- Add docstrings for all classes and methods
- Include usage examples in tool descriptions

## Contribution Types

**🔧 New Tools**: Data connectors, transformations, or analysis tools
**📊 Visualizations**: New chart types or export formats
**🧪 Testing**: Improve test coverage or add integration tests
**📝 Documentation**: Improve guides, examples, or API docs
**🐛 Bug Fixes**: Fix issues or improve error handling

## Pull Request Process

1. **Test thoroughly**: All tests must pass
2. **Update documentation**: README, docstrings, examples
3. **Follow commit format**: `feat:`, `fix:`, `docs:`, `test:`
4. **Describe changes**: Clear PR description with examples
5. **Review feedback**: Address reviewer comments promptly

## Code Review Checklist

- [ ] All tests pass (`python -m pytest`)
- [ ] New tools added to server registry
- [ ] Documentation updated
- [ ] Error handling implemented
- [ ] Type hints included
- [ ] Examples provided

## Questions?

Open an issue for questions about:
- Architecture decisions
- Tool design patterns
- Testing approaches
- Documentation improvements

Thank you for contributing to Engineer Your Data! 🚀