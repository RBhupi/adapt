"""
Directory setup for radar pipeline.

Supports flat directory structure (YYYYMMDD), good for multiday analysis.
- NEXRAD files keep original AWS names (no renaming)
- Date-first hierarchy: YYYYMMDD/RADAR
- Consistent across all output types
- Timestamps in filenames for easy sorting

Author: Bhupendra Raut
"""

import os
from pathlib import Path
from datetime import datetime, timezone


def setup_output_directories(base_output_dir=None):
    """
    Set up organized output directory structure.
    
    Parameters
    ----------
    base_output_dir : str or Path, optional
        Base output directory. If None, prompts user for input.
    
    Returns
    -------
    dict
        Dictionary with paths: 'base', 'gridnc', 'analysis', 'logs'
    """
    
    if base_output_dir is None:
        print("\n" + "=" * 70)
        print("RADAR PIPELINE - OUTPUT DIRECTORY SETUP")
        print("=" * 70)
        print("\nCurrent location: ", Path.cwd())
        print("\nDefault options:")
        print("  1. Current directory: ./output")
        print("  2. Home directory: ~/radar_output")
        print("  3. Custom path")
        
        choice = input("\nSelect option (1/2/3) [default=1]: ").strip() or "1"
        
        if choice == "1":
            base_output_dir = Path.cwd() / "output"
        elif choice == "2":
            base_output_dir = Path.home() / "radar_output"
        elif choice == "3":
            path_input = input("Enter custom path (use ~ for home): ").strip()
            base_output_dir = Path(path_input).expanduser()
        else:
            base_output_dir = Path.cwd() / "output"
    
    base_output_dir = Path(base_output_dir).expanduser().resolve()
    
    # Create subdirectories
    directories = {
        "base": base_output_dir,
        "nexrad": base_output_dir / "nexrad",
        "gridnc": base_output_dir / "gridnc",
        "analysis": base_output_dir / "analysis",
        "plots": base_output_dir / "plots",
        "logs": base_output_dir / "logs",
    }
    
    # Create all directories
    for key, path in directories.items():
        path.mkdir(parents=True, exist_ok=True)
    
    print("\nOutput directories created:")
    for key, path in directories.items():
        print(f"  {key:12s}: {path}")
    print("=" * 70 + "\n")
    
    return directories


def get_nexrad_path(output_dirs, radar_id, filename, scan_time=None):
    """
    Get new NEXRAD Level-II file path (preserves original AWS names).
    
    Parameters
    ----------
    output_dirs : dict
        Output directories from setup_output_directories()
    radar_id : str
        Radar station ID (e.g., 'KDIX')
    filename : str
        Original AWS filename (e.g., 'KDIX20251126_221706_V06')
    scan_time : datetime or str, optional
        Scan timestamp for YYYYMMDD directory. If None, extracted from filename.
    
    Returns
    -------
    Path
        Full path: nexrad/YYYYMMDD/RADAR_ID/original_filename
    
    Example
    -------
    >>> get_nexrad_path(dirs, 'KDIX', 'KDIX20251126_221706_V06')
    Path('output/nexrad/20251126/KDIX/KDIX20251126_221706_V06')
    """
    # Extract date from filename if not provided
    if scan_time is None:
        # Parse from filename: KDIX20251126_221706_V06 -> 20251126
        try:
            date_str = filename.split('_')[0][-8:]  # Last 8 chars before underscore
            scan_time = datetime.strptime(date_str, '%Y%m%d')
        except:
            scan_time = datetime.now(timezone.utc)
    elif isinstance(scan_time, str):
        scan_time = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
    
    # Create YYYYMMDD/RADAR_ID directory structure
    date_str = scan_time.strftime("%Y%m%d")
    output_dir = output_dirs["nexrad"] / date_str / radar_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir / filename


def get_netcdf_path(output_dirs, radar_id, filename, scan_time=None):
    """
    Get organized NetCDF file path (mirrors NEXRAD structure).
    
    Parameters
    ----------
    output_dirs : dict
        Output directories from setup_output_directories()
    radar_id : str
        Radar station ID (e.g., 'KDIX')
    filename : str
        Base filename (with or without .nc extension)
    scan_time : datetime or str, optional
        Scan timestamp. If None, extracted from filename.
    
    Returns
    -------
    Path
        Full path: gridnc/YYYYMMDD/RADAR_ID/filename.nc
    
    Example
    -------
    >>> get_netcdf_path(dirs, 'KDIX', 'KDIX20251126_221706_V06')
    Path('output/gridnc/20251126/KDIX/KDIX20251126_221706_V06.nc')
    """
    # Extract date from filename if not provided
    if scan_time is None:
        try:
            date_str = filename.split('_')[0][-8:]
            scan_time = datetime.strptime(date_str, '%Y%m%d')
        except:
            scan_time = datetime.now(timezone.utc)
    elif isinstance(scan_time, str):
        scan_time = datetime.fromisoformat(scan_time.replace('Z', '+00:00'))
    
    # Create YYYYMMDD/RADAR_ID directory structure
    date_str = scan_time.strftime("%Y%m%d")
    output_dir = output_dirs["gridnc"] / date_str / radar_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure .nc extension
    if not filename.endswith('.nc'):
        filename = filename + '.nc'
    
    return output_dir / filename


def get_analysis_path(output_dirs, radar_id=None, analysis_type="parquet", timestamp=None, filename=None):
    """
    Get organized analysis file path (YYYYMMDD/RADAR_ID structure).
    
    Parameters
    ----------
    output_dirs : dict
        Output directories from setup_output_directories()
    radar_id : str, optional
        Radar station ID (e.g., 'KDIX')
    analysis_type : str
        File type: 'parquet', 'csv', 'db', or 'netcdf'
    timestamp : datetime or str, optional
        Timestamp for directory. If None, uses current time.
    filename : str, optional
        Custom filename. If None, generates from radar_id and timestamp.
    
    Returns
    -------
    Path
        Full path: analysis/YYYYMMDD/RADAR_ID/filename.ext
    
    Example
    -------
    >>> get_analysis_path(dirs, radar_id='KDIX', analysis_type='parquet')
    Path('output/analysis/20251126/KDIX/KDIX_cells_properties.parquet')
    
    >>> get_analysis_path(dirs, radar_id='KDIX', analysis_type='db', 
    ...                   filename='KDIX_cells_statistics.db')
    Path('output/analysis/20251126/KDIX/KDIX_cells_statistics.db')
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    # Create YYYYMMDD/RADAR_ID directory
    date_str = timestamp.strftime("%Y%m%d")
    if radar_id:
        date_dir = output_dirs["analysis"] / date_str / radar_id
    else:
        date_dir = output_dirs["analysis"] / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        ext = analysis_type if not analysis_type.startswith('.') else analysis_type[1:]
        if radar_id:
            filename = f"{radar_id}_cells_properties.{ext}"
        else:
            filename = f"combined_cells_properties.{ext}"
    
    return date_dir / filename


def get_plot_path(output_dirs, radar_id=None, plot_type="reflectivity", timestamp=None, scan_time=None):
    """
    Get organized plot file path (YYYYMMDD/RADAR_ID structure).
    
    Parameters
    ----------
    output_dirs : dict
        Output directories from setup_output_directories()
    radar_id : str, optional
        Radar station ID (e.g., 'KDIX')
    plot_type : str
        Type of plot: 'reflectivity', 'velocity', 'segmentation'
    timestamp : datetime or str, optional
        Timestamp for file naming (scan time). If None, uses current time.
    scan_time : datetime or str, optional
        Alias for timestamp (for consistency with other functions)
    
    Returns
    -------
    Path
        Full path: plots/YYYYMMDD/RADAR_ID/RADAR_ID_plottype_HHMMSS.png
    
    Example
    -------
    >>> get_plot_path(dirs, radar_id='KDIX', plot_type='reflectivity')
    Path('output/plots/20251126/KDIX/KDIX_reflectivity_221706.png')
    """
    # Use scan_time if provided (takes precedence)
    if scan_time is not None:
        timestamp = scan_time
    
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    # Create YYYYMMDD/RADAR_ID directory
    date_str = timestamp.strftime("%Y%m%d")
    if radar_id:
        date_dir = output_dirs["plots"] / date_str / radar_id
    else:
        date_dir = output_dirs["plots"] / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    
    time_str = timestamp.strftime("%H%M%S")
    
    if radar_id:
        filename = f"{radar_id}_{plot_type}_{time_str}.png"
    else:
        filename = f"combined_{plot_type}_{time_str}.png"
    
    return date_dir / filename



def get_log_path(output_dirs, radar_id=None):
    """
    Get organized log file path.
    
    Parameters
    ----------
    output_dirs : dict
        Output directories from setup_output_directories()
    radar_id : str, optional
        Radar station ID
    
    Returns
    -------
    Path
        Full path to log file
    """
    log_dir = output_dirs["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if radar_id:
        filename = f"pipeline_{radar_id}_{timestamp}.log"
    else:
        filename = f"pipeline_latest.log"
    
    return log_dir / filename


if __name__ == "__main__":
    """Test the directory structure functions."""
    print("\n" + "=" * 70)
    print("TESTING DIRECTORY STRUCTURE")
    print("=" * 70)
    
    # Test setup
    dirs = setup_output_directories()
    
    # Test NEXRAD path (preserves original names)
    nexrad_path = get_nexrad_path(dirs, "KDIX", "KDIX20251126_221706_V06")
    print(f"\nNEXRAD path: {nexrad_path}")
    
    # Test NetCDF path (mirrors NEXRAD)
    nc_path = get_netcdf_path(dirs, "KDIX", "KDIX20251126_221706_V06")
    print(f"NetCDF path: {nc_path}")
    
    # Test analysis path
    analysis_path = get_analysis_path(dirs, "KDIX", "parquet")
    print(f"Analysis path: {analysis_path}")
    
    # Test SQLite path
    db_path = get_analysis_path(dirs, "KDIX", "db", 
                                filename="KDIX_cells_statistics.db")
    print(f"SQLite path: {db_path}")
    
    # Test plot path
    plot_path = get_plot_path(dirs, "KDIX", "reflectivity")
    print(f"Plot path: {plot_path}")
    
    # Test log path
    log_path = get_log_path(dirs, "KDIX")
    print(f"Log path: {log_path}")
    
    print("\n" + "=" * 70)
    print("DIRECTORY STRUCTURE VERIFIED")
    print("=" * 70)
