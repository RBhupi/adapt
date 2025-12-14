# Configuration

- **config.py**: Core configuration settings for the pipeline.
- **environment.yml**: Dependency management via conda/micromamba.
- **pyproject.toml**: Package metadata and optional dependencies.

## Key Configuration Files

### config.py
Contains settings for radar data processing, thresholds, and output paths.

### environment.yml
Specifies all dependencies (system libraries, Python packages) needed to run ADAPT. This ensures consistent environments across development, CI, and deployment.

### pyproject.toml
Defines package metadata, dependencies, and optional feature sets (e.g., `[project.optional-dependencies].dev` for development tools).

See the source code for detailed configuration options.
