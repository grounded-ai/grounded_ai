# TODOs

## Development Tasks

- [ ] Write unit tests
- [ ] Update model to provide reasoning outputs
- [ ] Create pypi test package
- [ ] Create pypi test package (real)
- [ ] Update readme
- [ ] Create release on gh
- [ ] See about actions like ruff linting on PR

## Additional Considerations

### Documentation & Examples
- [ ] Add docstrings to all public methods and classes
- [ ] Create example notebooks in `/examples` directory
- [ ] Add API reference documentation (Sphinx/MkDocs)
- [ ] Add troubleshooting guide for common issues

### Code Quality & Standards
- [ ] Add type checking with mypy
- [ ] Set up pre-commit hooks (black, isort, flake8)
- [ ] Add code coverage reporting
- [ ] Create contributing guidelines (CONTRIBUTING.md)

### Package Management
- [ ] Pin dependency versions in requirements files
- [ ] Add optional dependencies for dev/test environments
- [ ] Create environment.yml for conda users
- [ ] Add security scanning for dependencies

### Error Handling & Validation
- [ ] Add comprehensive error handling for model loading failures
- [ ] Validate input data formats more robustly
- [ ] Add graceful fallbacks for CUDA/memory issues
- [ ] Implement retry logic for model inference failures

### Performance & Monitoring
- [ ] Add performance benchmarks
- [ ] Implement caching for repeated evaluations
- [ ] Add memory usage monitoring
- [ ] Consider async/batch processing for large datasets

### User Experience
- [ ] Add progress bars for long-running evaluations
- [ ] Create CLI interface for command-line usage
- [ ] Add configuration file support (YAML/JSON)
- [ ] Implement result export formats (CSV, JSON, Excel)

### Security & Compliance
- [ ] Add input sanitization for prompts
- [ ] Implement rate limiting for API usage
- [ ] Add audit logging for evaluations
- [ ] Consider GDPR/privacy compliance for text processing