# Changelog: Store Insights AI

## [Unreleased]

### Added
- Integration with LLaVA v1.5-7b model for image analysis
- Requirements.txt with updated dependencies
- Comprehensive README.md with installation and usage instructions
- Improved documentation including feature design and current state
- Python path fix to properly import LLaVA modules
- LLaVA directory README and setup instructions

### Changed
- Replaced MiniCPM and Phi models with LLaVA model in inference.py
- Updated image processing pipeline to work with LLaVA
- Modified response generation and parsing for LLaVA's output format
- Changed model references throughout the codebase
- Updated main README with detailed LLaVA setup instructions

### Removed
- MiniCPM and Phi model-specific code
- Unused dependencies related to previous models

## [0.1.0] - Initial Version

### Added
- Basic Streamlit application structure
- Dashboard for displaying metrics and insights
- Gender Demographics module for analyzing customer demographics
- Queue Management module for analyzing checkout counters
- Mock data fallback for systems without GPU 