# `Adapt`: A real-time weather radar data processing platform designed to guide adaptive scan strategy.

** Alpha version – currently in development and internal testing. **

## Quickstart

1. **Clone the repository:**
   ```sh
   git clone https://github.com/RBhupi/adapt.git
   cd adapt
   ```

2. **Create the environment:**
   - With mamba/conda:
     ```sh
     mamba env create -f environment.yml
     mamba activate adapt
     ```
   - Or with pip:
     ```sh
     pip install -r requirements.txt
     pip install -e .
     ```

3. **Edit config:**
   - Open `scripts/user_config.py` and set the `BASE_DIR` variable to your desired output directory.

4. **Run the pipeline:**
   ```sh
   python scripts/run_nexrad_pipeline.py scripts/user_config.py
    ```


[![codecov](https://codecov.io/gh/RBhupi/adapt/branch/main/graph/badge.svg)](https://codecov.io/gh/RBhupi/adapt)


