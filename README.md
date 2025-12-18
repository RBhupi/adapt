# `Adapt`: Adaptive Radar Data Processing Platform

> A real-time weather radar data processing platform designed for adaptive scan strategy research.

**Status:** Alpha (in development, not fully tested)

## 🎯 What is ADAPT?

**ADAPT** processes NEXRAD Level-II radar data in real-time or batch modes to:

1. **Download** NEXRAD files from AWS S3 (automatic streaming architecture)
2. **Regrid** to Cartesian coordinates using Cressman interpolation (ARM Py-ART)
3. **Segment** storm cells using reflectivity thresholding and morphology
4. **Project** cell motion using optical flow
5. **Analyze** cell properties (area, centroid, reflectivity, motion)
6. **Persist** results to NetCDF, SQLite, and PNG visualizations

### Architecture

```
Input: AWS S3 NEXRAD
    ↓
┌─────────────────────────────────────────┐
│ Realtime/Historical Downloader Thread   │ ← Stream latest or date range
└─────────────────────────────────────────┘
    ↓ (Queue)
┌─────────────────────────────────────────┐
│ Processor Thread (Parallel)             │
│  ├─ Load & Regrid (PyART)              │
│  ├─ Segment (Threshold + Morphology)   │
│  ├─ Project (Optical Flow)             │
│  └─ Analyze (Cell Statistics)          │
└─────────────────────────────────────────┘
    ├─ Output: NetCDF (gridded + analysis)
    ├─ Output: SQLite (cell database)
    └─→ Plotter Queue
        ↓
┌─────────────────────────────────────────┐
│ Plotter Thread                          │
│  ├─ Reflectivity visualization         │
│  └─ Optical flow overlay               │
└─────────────────────────────────────────┘
    └─ Output: PNG plots

Final Exports:
  ✓ Gridded NetCDF
  ✓ Analysis NetCDF (segmentation + projections)
  ✓ SQLite database (cell stats)
  ✓ PNG plots
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/RBhupi/adapt.git
cd adapt
```

### 2. Create Environment

**Option A: Conda/Mamba (Recommended)**

```bash
conda env create -f environment.yml -n adapt_env
conda activate adapt_env
```

**Option B: Pip with venv**

```bash
python -m venv adapt_env
source adapt_env/bin/activate  # On Windows: adapt_env\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 3. Configure

Edit `scripts/user_config.py` to set:
- `base_dir`: Output directory for results
- `radar_id`: Target NEXRAD radar (e.g., "KMOB", "KHTX")
- `mode`: "realtime" or "historical"
- Thresholds and processing parameters

### 4. Run Pipeline

**Realtime mode (last 60 minutes, continuous):**
```bash
python scripts/run_nexrad_pipeline.py scripts/user_config.py
```

**Historical mode (specific date range):**
```bash
python scripts/run_nexrad_pipeline.py scripts/user_config.py \
  --mode historical \
  --start-time 2024-06-15T00:00:00Z \
  --end-time 2024-06-15T02:00:00Z
```

**Override radar ID:**
```bash
python scripts/run_nexrad_pipeline.py scripts/user_config.py --radar-id KMOB
```

## 📦 Installation Details

### System Requirements

- **Python:** 3.8+
- **OS:** macOS, Linux, Windows (with WSL2 recommended)
- **RAM:** 4 GB minimum (>8 GB recommended for batch processing)
- **Disk:** Fast SSD for NetCDF I/O (temporal data causes HDF5 congestion)

### Dependencies

**Core Libraries:**
- `numpy`, `xarray`, `pandas`: Numerical computing
- `arm_pyart`: ARM Py-ART (loading & regridding)
- `opencv-python`: Optical flow (Farneback)
- `scikit-image`: Morphological operations
- `pydantic>=2.0`: Configuration validation
- `nexradaws`: AWS S3 NEXRAD inventory

**I/O & Visualization:**
- `netCDF4`, `h5py`: Binary data formats
- `matplotlib`, `cartopy`: Plotting & map projections
- `contextily`: Basemaps

**Development:**
- `pytest`, `pytest-cov`: Testing
- `sphinx`, `sphinx-rtd-theme`: Docs (for GitHub Pages)

### Troubleshooting

**HDF5 errors in plots:** This is benign (Py-ART checking file types). Plots still generate correctly.


**Cartopy projection errors:** Try:
```bash
conda install -c conda-forge cartopy=0.21  # Specific version
```

## 🧪 Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suite

```bash
pytest tests/schemas/test_config_resolution.py -v      # Config tests
pytest tests/radar/test_downloader_*.py -v            # Downloader tests
pytest tests/radar/test_projector_*.py -v             # Projector tests
pytest -m unit                                         # Fast unit tests only
```

### Test Coverage

```bash
pytest --cov=src/adapt --cov-report=html tests/
open htmlcov/index.html
```

**Current Status:** ✅ 111/111 tests passing

## 📊 Output Files

For each NEXRAD scan, ADAPT generates:

```
output_dir/
├── nexrad/              # Raw Level-II files (streamed from AWS)
├── gridnc/              # Gridded NetCDF (regridded data)
├── analysis/            # Analysis results
│   ├── *_analysis.nc    # Segmentation + projections
│   ├── *_cells_statistics.db    # SQLite database
│   └── *_cells_statistics.parquet # Analytics export
├── plots/               # Visualization PNG files
├── logs/                # Pipeline logs
└── tmp/                 # Temporary processing files
```

### Database Schema

The SQLite database contains one row per detected cell:

| Column | Type | Description |
|--------|------|-------------|
| time | TIMESTAMP | Scan time |
| cell_label | INT | Cell ID |
| cell_area_sqkm | FLOAT | Cell area in km2 |
| max_dbz | FLOAT | Maximum reflectivity |
| mean_dbz | FLOAT | Mean reflectivity |
| heading_x, heading_y | FLOAT | Motion vectors |
| projection_1..5 | FLOAT | Forward motion forecasts |
| (60+ columns total) | ... | Centroids, dimensions, etc. |

## 🔧 Configuration System

ADAPT uses a 3-layer configuration hierarchy:

```
ParamConfig (expert defaults in src/adapt/schemas/param.py)
    ↓ override
UserConfig (user-facing settings in scripts/user_config.py)
    ↓ override
CLIConfig (command-line args)
    ↓
InternalConfig (validated, immutable runtime config)
```

**Field Aliases** allow friendly config names:
```python
threshold=35    # Maps to: segmenter.threshold=35.0
radar_id="KMOB"     # Maps to: downloader.radar_id="KMOB"
min_cellsize_gridpoint=10    # Maps to: segmenter.min_cellsize_gridpoint=10
```

## 📚 Documentation

### Build Docs

```bash
cd docs
make html
open build/html/index.html
```

### GitHub Pages Deployment

Docs build automatically in GitHub Actions (see `.github/workflows/docs.yml`).

## 🤝 Contributing

1. Fork repository
2. Create branch: `git checkout -b feature/your-feature`
3. Write tests: `pytest tests/`
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature/your-feature`
6. Open Pull Request

## 📈 Performance

**Typical performance** (KMOB radar, 2024 data):
- Per-file processing: 5-10 seconds
- Realtime mode: ~40 files/hour
- Historical batch: 13 files in ~90 seconds
- Cell detection: 0-10 cells per scan (threshold-dependent)

## 🙏 Acknowledgments

Built with support from the U.S. Department of Energy as part of the Atmospheric Radiation Measurement (ARM) Research Facility, an Office of Science user facility.

## 📝 License

See [LICENSE](LICENSE) file

## 📧 Contact

Bhupendra Raut (@RBhupi) 

