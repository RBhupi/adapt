# Pipeline Overview

ADAPT follows a modular radar processing pipeline: **Downloader → Loader + Processor → Visualization**.

## Pipeline Stages

### 1. Downloader
Downloads radar data from NEXRAD or other sources.
- Supports real-time and historical data.
- Handles data queuing and retries.
- Located in `src/adapt/radar/downloader.py`.

### 2. Loader
Loads and parses radar data into memory.
- Converts raw radar files into xarray Datasets.
- Handles multiple file formats (HDF5, NetCDF).
- Located in `src/adapt/radar/loader.py`.

### 3. Processor
Segments and analyzes radar cells.
- **Segmenter**: Identifies radar cells using morphological operations.
- **Cell Analyzer**: Extracts cell properties (centroids, areas, reflectivity).
- **Cell Projector**: Projects cells onto geographic coordinates.
- Located in `src/adapt/radar/` and `src/adapt/pipeline/`.

### 4. Visualization
Plots results and generates output.
- Displays radar reflectivity and segmented cells.
- Generates publication-quality figures.
- Located in `src/adapt/visualization/plotter.py`.

## Orchestration

The **Orchestrator** coordinates the pipeline stages and the **Processor** manages job queuing and execution. See the pipeline module for details.
