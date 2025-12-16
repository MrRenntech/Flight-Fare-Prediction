# Changelog

## [Unreleased] - 2024-05-22

### Added
-   `HowItWorks.txt`: Comprehensive guide to the project internals.
-   `CHANGELOG.md`: This file.
-   `requirements.txt`: Dependency list.
-   `launch.bat`: Click-to-run script.
-   `install_prereq.bat`: Setup script.
-   `columns.json`: Metadata artifact for robust feature alignment.

### Changed
-   **Refactored `Flight model.py`**:
    -   Modularized into functions (`load_data`, `preprocess_data`, `train_model`).
    -   Added logic to save `columns.json` alongside the model.
    -   Cleaned up redundant code.
-   **Refactored `main.py`**:
    -   **Dynamic Feature Encoding**: Replaced hardcoded `if-else` chains with a robust DataFrame alignment method using `columns.json`. This ensures the input to the prediction model exactly matches the training features.
    -   Improved error handling and code readability.
-   **Documentation**: Updated `README.md` with better structure and instructions.
