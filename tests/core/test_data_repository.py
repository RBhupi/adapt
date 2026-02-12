"""Tests for DataRepository artifact management."""

import sqlite3
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from adapt.core import DataRepository, ProductType


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def temp_base_dir():
    """Create temporary base directory."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def repository(temp_base_dir):
    """Create DataRepository instance."""
    repo = DataRepository(
        run_id="test1234",
        base_dir=temp_base_dir,
        radar_id="KDIX"
    )
    yield repo
    repo.close()


@pytest.fixture
def sample_dataset():
    """Create a sample xarray Dataset."""
    return xr.Dataset({
        'reflectivity': xr.DataArray(
            np.random.randn(10, 10).astype(np.float32),
            dims=['y', 'x'],
            coords={
                'y': np.arange(10) * 1000.0,
                'x': np.arange(10) * 1000.0,
            }
        )
    })


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame."""
    return pd.DataFrame({
        'cell_label': [1, 2, 3],
        'cell_area_sqkm': [100.0, 200.0, 150.0],
        'reflectivity_max': [45.5, 52.3, 48.1],
    })


# =========================================================================
# Test: Catalog Initialization
# =========================================================================


class TestCatalogInitialization:
    """Test catalog database creation."""

    def test_catalog_created(self, repository, temp_base_dir):
        """Catalog SQLite file should be created."""
        catalog_path = temp_base_dir / "catalog" / "test1234_catalog.sqlite"
        assert catalog_path.exists()

    def test_directory_structure_created(self, repository, temp_base_dir):
        """All required directories should be created."""
        expected_dirs = [
            temp_base_dir / "catalog",
            temp_base_dir / "KDIX" / "nexrad",
            temp_base_dir / "KDIX" / "gridnc",
            temp_base_dir / "KDIX" / "analysis",
            temp_base_dir / "KDIX" / "plots",
            temp_base_dir / "logs",
        ]
        for d in expected_dirs:
            assert d.exists(), f"Directory not created: {d}"

    def test_tables_created(self, repository):
        """All required tables should exist."""
        conn = repository._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "artifacts" in tables
        assert "processing_status" in tables
        assert "runs" in tables

    def test_run_registered(self, repository):
        """Current run should be registered."""
        conn = repository._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (repository.run_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row['status'] == 'running'
        assert row['radar_id'] == 'KDIX'


# =========================================================================
# Test: Artifact Registration
# =========================================================================


class TestArtifactRegistration:
    """Test artifact registration."""

    def test_register_artifact(self, repository, temp_base_dir):
        """Should register artifact and return ID."""
        # Create dummy file
        file_path = temp_base_dir / "KDIX" / "nexrad" / "20260211" / "test.nc"
        file_path.parent.mkdir(parents=True)
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.NEXRAD_RAW,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test",
            parent_ids=[],
            metadata={"test": True}
        )

        assert len(artifact_id) == 16

    def test_register_artifact_without_scan_time(self, repository, temp_base_dir):
        """Should register artifact without scan_time."""
        file_path = temp_base_dir / "test_file.db"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.CELLS_DB,
            file_path=file_path,
            producer="test"
        )

        assert len(artifact_id) == 16

    def test_query_artifacts(self, repository, temp_base_dir):
        """Should query registered artifacts."""
        file_path = temp_base_dir / "test.nc"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        results = repository.query(product_type=ProductType.GRIDDED_NC)
        assert len(results) == 1
        assert results[0]['artifact_id'] == artifact_id

    def test_query_by_time_range(self, repository, temp_base_dir):
        """Should filter by time range."""
        # Register two artifacts with different times
        file1 = temp_base_dir / "file1.nc"
        file2 = temp_base_dir / "file2.nc"
        file1.touch()
        file2.touch()

        repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file1,
            scan_time=datetime(2026, 2, 11, 10, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )
        repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file2,
            scan_time=datetime(2026, 2, 11, 14, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        # Query for afternoon only
        results = repository.query(
            product_type=ProductType.GRIDDED_NC,
            time_range=(
                datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2026, 2, 11, 16, 0, 0, tzinfo=timezone.utc)
            )
        )
        assert len(results) == 1

    def test_get_artifact(self, repository, temp_base_dir):
        """Should retrieve artifact by ID."""
        file_path = temp_base_dir / "test.nc"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.ANALYSIS_NC,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )

        artifact = repository.get_artifact(artifact_id)
        assert artifact is not None
        assert artifact['product_type'] == ProductType.ANALYSIS_NC
        assert artifact['producer'] == "processor"


# =========================================================================
# Test: Processing Status
# =========================================================================


class TestProcessingStatus:
    """Test processing status tracking."""

    def test_get_unprocessed(self, repository, temp_base_dir):
        """Should return artifacts not processed by plugin."""
        file_path = temp_base_dir / "test.nc"
        file_path.touch()

        repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        pending = repository.get_unprocessed(
            product_type=ProductType.GRIDDED_NC,
            plugin_name="segmenter"
        )
        assert len(pending) == 1

    def test_mark_complete(self, repository, temp_base_dir):
        """Should mark artifact as processed."""
        file_path = temp_base_dir / "test.nc"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        repository.mark_complete(artifact_id, "segmenter", "completed")

        pending = repository.get_unprocessed(
            product_type=ProductType.GRIDDED_NC,
            plugin_name="segmenter"
        )
        assert len(pending) == 0

    def test_mark_failed(self, repository, temp_base_dir):
        """Should mark artifact as failed with error message."""
        file_path = temp_base_dir / "test.nc"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.GRIDDED_NC,
            file_path=file_path,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        repository.mark_complete(
            artifact_id,
            "segmenter",
            status="failed",
            error_message="Test error"
        )

        # Check status was recorded
        conn = repository._get_connection()
        cursor = conn.execute(
            "SELECT * FROM processing_status WHERE artifact_id = ?",
            (artifact_id,)
        )
        row = cursor.fetchone()
        assert row['status'] == 'failed'
        assert row['error_message'] == 'Test error'


# =========================================================================
# Test: Write Operations
# =========================================================================


class TestWriteOperations:
    """Test atomic write operations."""

    def test_write_netcdf(self, repository, sample_dataset):
        """Should write NetCDF and register artifact."""
        artifact_id = repository.write_netcdf(
            ds=sample_dataset,
            product_type=ProductType.ANALYSIS_NC,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        # Verify file exists
        artifact = repository.get_artifact(artifact_id)
        assert artifact is not None
        assert Path(artifact['file_path']).exists()

        # Verify filename pattern
        filename = Path(artifact['file_path']).name
        assert "test1234" in filename  # run_id
        assert "analysis" in filename
        assert filename.endswith(".nc")

    def test_write_netcdf_gridded(self, repository, sample_dataset):
        """Should write gridded NetCDF with correct path."""
        artifact_id = repository.write_netcdf(
            ds=sample_dataset,
            product_type=ProductType.GRIDDED_NC,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="loader"
        )

        artifact = repository.get_artifact(artifact_id)
        file_path = Path(artifact['file_path'])

        # Check directory structure: root/RADAR_ID/gridnc/YYYYMMDD/
        assert "KDIX" in str(file_path)
        assert "gridnc" in str(file_path)
        assert "20260211" in str(file_path)
        assert "gridded" in file_path.name

    def test_write_parquet(self, repository, sample_dataframe):
        """Should write Parquet and register artifact."""
        artifact_id = repository.write_parquet(
            df=sample_dataframe,
            product_type=ProductType.CELLS_PARQUET,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        # Verify file exists
        artifact = repository.get_artifact(artifact_id)
        assert artifact is not None
        assert Path(artifact['file_path']).exists()

        # Verify metadata includes row count
        import json
        metadata = json.loads(artifact['metadata'])
        assert metadata['row_count'] == 3

    def test_get_or_create_cells_db(self, repository):
        """Should create cells database."""
        artifact_id = repository.get_or_create_cells_db(
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )

        artifact = repository.get_artifact(artifact_id)
        assert artifact is not None
        assert artifact['product_type'] == ProductType.CELLS_DB
        assert Path(artifact['file_path']).exists()

    def test_get_or_create_cells_db_reuse(self, repository):
        """Should reuse existing cells database."""
        id1 = repository.get_or_create_cells_db(
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )
        id2 = repository.get_or_create_cells_db(
            scan_time=datetime(2026, 2, 11, 13, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )

        assert id1 == id2

    def test_write_sqlite_table(self, repository, sample_dataframe):
        """Should write DataFrame to SQLite table."""
        # Create cells db first
        db_artifact_id = repository.get_or_create_cells_db(
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )

        # Write table
        repository.write_sqlite_table(
            df=sample_dataframe,
            table_name='cells',
            artifact_id=db_artifact_id
        )

        # Verify data was written
        artifact = repository.get_artifact(db_artifact_id)
        with sqlite3.connect(artifact['file_path']) as conn:
            df_read = pd.read_sql("SELECT * FROM cells", conn)
            assert len(df_read) == 3


# =========================================================================
# Test: Data Access
# =========================================================================


class TestDataAccess:
    """Test data access operations."""

    def test_open_dataset(self, repository, sample_dataset):
        """Should open NetCDF as xarray Dataset."""
        artifact_id = repository.write_netcdf(
            ds=sample_dataset,
            product_type=ProductType.ANALYSIS_NC,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        opened_ds = repository.open_dataset(artifact_id)
        assert 'reflectivity' in opened_ds.data_vars
        opened_ds.close()

    def test_open_dataset_invalid_type(self, repository, temp_base_dir):
        """Should raise error for non-NetCDF artifact."""
        file_path = temp_base_dir / "test.db"
        file_path.touch()

        artifact_id = repository.register_artifact(
            product_type=ProductType.CELLS_DB,
            file_path=file_path,
            producer="test"
        )

        with pytest.raises(ValueError, match="Cannot open as dataset"):
            repository.open_dataset(artifact_id)

    def test_open_table_parquet(self, repository, sample_dataframe):
        """Should open Parquet as DataFrame."""
        artifact_id = repository.write_parquet(
            df=sample_dataframe,
            product_type=ProductType.CELLS_PARQUET,
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="test"
        )

        opened_df = repository.open_table(artifact_id)
        assert len(opened_df) == 3
        assert 'cell_label' in opened_df.columns

    def test_open_table_sqlite(self, repository, sample_dataframe):
        """Should open SQLite table as DataFrame."""
        db_artifact_id = repository.get_or_create_cells_db(
            scan_time=datetime(2026, 2, 11, 12, 0, 0, tzinfo=timezone.utc),
            producer="processor"
        )
        repository.write_sqlite_table(
            df=sample_dataframe,
            table_name='cells',
            artifact_id=db_artifact_id
        )

        opened_df = repository.open_table(db_artifact_id, table_name='cells')
        assert len(opened_df) == 3

    def test_open_nonexistent_artifact(self, repository):
        """Should raise error for nonexistent artifact."""
        with pytest.raises(ValueError, match="Artifact not found"):
            repository.open_dataset("nonexistent")


# =========================================================================
# Test: Lifecycle
# =========================================================================


class TestLifecycle:
    """Test repository lifecycle."""

    def test_finalize_run(self, repository):
        """Should mark run as complete."""
        repository.finalize_run("completed")

        conn = repository._get_connection()
        cursor = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?",
            (repository.run_id,)
        )
        row = cursor.fetchone()
        assert row['status'] == 'completed'
        assert row['end_time'] is not None

    def test_context_manager(self, temp_base_dir):
        """Should work as context manager."""
        with DataRepository(
            run_id="ctx12345",
            base_dir=temp_base_dir,
            radar_id="KHTX"
        ) as repo:
            assert repo.run_id == "ctx12345"

    def test_generate_run_id(self):
        """Should generate valid run IDs."""
        run_id = DataRepository.generate_run_id()
        assert len(run_id) == 8
        assert run_id.isalnum()


# =========================================================================
# Test: Path Generation
# =========================================================================


class TestPathGeneration:
    """Test path generation methods."""

    def test_generate_plot_path(self, repository):
        """Should generate correct plot path."""
        path = repository.generate_plot_path(
            plot_type="reflectivity",
            scan_time=datetime(2026, 2, 11, 12, 30, 45, tzinfo=timezone.utc)
        )

        assert "KDIX" in str(path)
        assert "plots" in str(path)
        assert "20260211" in str(path)
        assert "reflectivity" in path.name
        assert "123045" in path.name  # HHMMSS
        assert "test1234" in path.name  # run_id
        assert path.suffix == ".png"
