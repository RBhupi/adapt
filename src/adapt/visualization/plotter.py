"""
Radar plotting module. I don't know if this should be in the 
orchetrator or a separate  thing altogether. This did not turn out the way I thought but Keeping it for now.
"""

import threading
import queue
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import contextily as ctx
    from pyproj import Transformer
    CONTEXTILY_AVAILABLE = True
except ImportError:
    CONTEXTILY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RadarPlotter:
    """Refactored radar plotter with modular functions."""
    
    def __init__(self, config: Dict = None, show_plots: bool = False):
        """Init plotter with config."""
        self.config = config or {}
        viz_config = self.config.get("visualization", {}) 
        
        # Plot configuration
        self.dpi = viz_config.get("dpi", 200)
        self.figsize = tuple(viz_config.get("figsize", (20, 10)))
        self.output_format = viz_config.get("output_format", "png")
        
        # Basemap configuration
        self.use_basemap = viz_config.get("use_basemap", True)
        self.basemap_alpha = viz_config.get("basemap_alpha", 0.6)
        
        # Style configuration
        self.seg_linewidth = viz_config.get("seg_linewidth", 1)
        self.proj_linewidth = viz_config.get("proj_linewidth", 0.8)
        self.proj_alpha = viz_config.get("proj_alpha", 0.6)
        self.flow_scale = viz_config.get("flow_scale", 1.0)
        self.flow_subsample = viz_config.get("flow_subsample", 10)
        
        # Reflectivity thresholds
        self.min_refl = viz_config.get("min_reflectivity", 0)
        self.vmin = viz_config.get("refl_vmin", 10)
        self.vmax = viz_config.get("refl_vmax", 50)
        
        if self.use_basemap and not CONTEXTILY_AVAILABLE:
            logger.warning("Basemap requested but contextily not installed")
            self.use_basemap = False
        
        logger.info(f"✓ RadarPlotter initialized (format={self.output_format}, dpi={self.dpi})")
    
    def _get_var_name(self, var_key: str, default: str) -> str:
        """Get variable name from config."""
        return self.config.get("global", {}).get("var_names", {}).get(var_key, default)
    
    def _get_coord_name(self, coord_key: str, default: str) -> str:
        """Get coordinate name from config."""
        return self.config.get("global", {}).get("coord_names", {}).get(coord_key, default)
    
    def _extract_timestamp(self, ds: xr.Dataset) -> datetime:
        """Extract timestamp from dataset."""
        if 'time' not in ds.coords:
            return datetime.now()
        
        try:
            time_val = ds.coords['time'].values
            if np.ndim(time_val) == 0:
                return pd.Timestamp(time_val).to_pydatetime()
            else:
                return pd.Timestamp(time_val[0]).to_pydatetime()
        except Exception:
            return datetime.now()
    
    def _get_coordinates_km(self, ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
        """Get x, y coordinates in km."""
        y_name = self._get_coord_name("y", "y")
        x_name = self._get_coord_name("x", "x")
        
        y_coords = ds[y_name].values / 1000  # Convert m to km
        x_coords = ds[x_name].values / 1000
        
        return x_coords, y_coords
    
    def _mask_reflectivity(self, refl: np.ndarray) -> np.ma.MaskedArray:
        """Apply thresholding to reflectivity."""
        refl_float = refl.astype(float)
        return np.ma.masked_where(
            (refl_float < self.min_refl) | np.isnan(refl_float),
            refl_float
        )
    
    def _setup_figure(self) -> Tuple[plt.Figure, plt.Axes, plt.Axes]:
        """Create figure with two subplots."""
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=self.figsize,
            dpi=self.dpi
        )
        return fig, ax1, ax2
    
    def _get_radar_location(self, ds: xr.Dataset) -> Tuple[float, float]:
        """Extract radar lat/lon from dataset."""
        def extract_float(val):
            if isinstance(val, xr.DataArray):
                return float(val.values)
            return float(val)
        
        lat = extract_float(
            ds.attrs.get('radar_latitude',
                        ds.coords.get('radar_latitude',
                                     ds.attrs.get('origin_latitude', 0)))
        )
        lon = extract_float(
            ds.attrs.get('radar_longitude',
                        ds.coords.get('radar_longitude',
                                     ds.attrs.get('origin_longitude', 0)))
        )
        return lat, lon
    
    def _add_basemap(self, ax: plt.Axes, ds: xr.Dataset, x_coords: np.ndarray, y_coords: np.ndarray) -> None:
        """Add OpenStreetMap basemap to axis."""
        if not self.use_basemap or not CONTEXTILY_AVAILABLE:
            return
        
        try:
            radar_lat, radar_lon = self._get_radar_location(ds)
            
            # Set CRS for azimuthal equidistant (km units)
            crs_str = f"+proj=aeqd +lat_0={radar_lat} +lon_0={radar_lon} +x_0=0 +y_0=0 +datum=WGS84 +units=km"
            
            ax.set_xlim(x_coords.min(), x_coords.max())
            ax.set_ylim(y_coords.min(), y_coords.max())
            
            ctx.add_basemap(
                ax,
                crs=crs_str,
                source=ctx.providers.OpenStreetMap.Mapnik,
                alpha=self.basemap_alpha,
                attribution=False,
                zoom='auto'
            )
        except Exception as e:
            logger.warning(f"Could not add basemap: {e}")
    
    def _plot_reflectivity_field(
        self,
        ax: plt.Axes,
        refl: np.ma.MaskedArray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> matplotlib.image.AxesImage:
        """Plot reflectivity pcolormesh."""
        return ax.pcolormesh(
            x_coords,
            y_coords,
            refl,
            cmap='ChaseSpectral',
            vmin=self.vmin,
            vmax=self.vmax,
            shading='auto',
            zorder=1
        )
    
    def _add_colorbar(self, ax: plt.Axes, im: matplotlib.image.AxesImage) -> None:
        """Add colorbar to axis."""
        cbar = plt.colorbar(im, ax=ax, label='Reflectivity (dBZ)', fraction=0.046, pad=0.04)
    
    def _plot_heading_yectors(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> bool:
        """Plot optical flow arrows on Panel 1."""
        heading_x_name = self._get_var_name("heading_x", "heading_x")
        heading_y_name = self._get_var_name("heading_y", "heading_y")
        
        if heading_x_name not in ds.data_vars or heading_y_name not in ds.data_vars:
            return False
        
        heading_x = ds[heading_x_name].values
        heading_y = ds[heading_y_name].values
        
        if np.all(np.isnan(heading_x)):
            logger.debug("Optical flow not plotted (all NaN - first frame)")
            return False
        
        # Subsample for clarity
        y_indices = np.arange(0, len(y_coords), self.flow_subsample)
        x_indices = np.arange(0, len(x_coords), self.flow_subsample)
        
        Y_sub = y_coords[y_indices]
        X_sub = x_coords[x_indices]
        U_sub = heading_x[np.ix_(y_indices, x_indices)]
        V_sub = heading_y[np.ix_(y_indices, x_indices)]
        
        X_mesh, Y_mesh = np.meshgrid(X_sub, Y_sub)
        
        ax.quiver(
            X_mesh, Y_mesh,
            U_sub, V_sub,
            color='#333333',
            alpha=0.7,
            scale=self.flow_scale,
            scale_units='xy',
            width=0.002,
            headwidth=3,
            headlength=4,
            zorder=45
        )
        
        logger.info(f"✓ Plotted optical flow field ({len(y_indices)}x{len(x_indices)} vectors, scale={self.flow_scale})")
        return True
    
    def _plot_segmentation_contours(
        self,
        ax: plt.Axes,
        labels: xr.DataArray,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> None:
        """Plot thin black contours for segmented cells."""
        labels_data = labels.values
        unique_labels = np.unique(labels_data)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return
        
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Plot each cell individually with binary mask for clean contours
        for cell_id in unique_labels:
            cell_mask = (labels_data == cell_id).astype(float)
            ax.contour(
                x_grid, y_grid,
                cell_mask,
                levels=[0.5],
                colors='black',
                linewidths=self.seg_linewidth,
                alpha=0.9,
                zorder=50
            )
    
    def _plot_projection_contours(
        self,
        ax: plt.Axes,
        ds: xr.Dataset,
        x_coords: np.ndarray,
        y_coords: np.ndarray
    ) -> None:
        """Plot thin transparent gray contours for projections."""
        proj_name = self._get_var_name("cell_projections", "cell_projections")
        frame_offset_name = self._get_coord_name("frame_offset", "frame_offset")
        
        if proj_name not in ds.data_vars:
            return
        
        proj_da = ds[proj_name]
        if frame_offset_name not in proj_da.dims:
            return
        
        y_grid, x_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        linestyles = ['dashed', 'dashdot', 'dotted']
        base_width = self.proj_linewidth
        
        num_frames = len(proj_da[frame_offset_name])
        
        # Skip frame_offset=0 (registration), plot future projections
        for proj_idx in range(1, num_frames):
            labels_proj = proj_da.isel({frame_offset_name: proj_idx}).values
            
            if np.all(np.isnan(labels_proj)):
                continue
            
            unique_proj = np.unique(labels_proj)
            unique_proj = unique_proj[unique_proj > 0]
            
            if len(unique_proj) == 0:
                continue
            
            style_idx = (proj_idx - 1) % len(linestyles)
            linewidth = base_width * (1 - 0.1 * (proj_idx - 1))
            
            # Plot each cell individually to avoid matplotlib contour quirks
            for cell_id in unique_proj:
                # Create binary mask for this specific cell
                cell_mask = (labels_proj == cell_id).astype(float)
                
                # Plot single contour at 0.5 level (boundary of cell)
                ax.contour(
                    x_grid, y_grid,
                    cell_mask,
                    levels=[0.5],
                    colors='#555555',
                    linewidths=linewidth,
                    linestyles=linestyles[style_idx],
                    alpha=self.proj_alpha,
                    zorder=40
                )
    
    def _format_axis(
        self,
        ax: plt.Axes,
        title: str,
        timestamp: datetime,
        radar_id: str
    ) -> None:
        """Format axis with labels and title."""
        ax.set_xlabel('Distance from Radar - X (km)', fontsize=11)
        ax.set_ylabel('Distance from Radar - Y (km)', fontsize=11)
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        time_str = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        ax.set_title(
            f'{radar_id} {title}\n{time_str}',
            fontsize=12,
            fontweight='bold',
            pad=10
        )
    
    def _add_flow_legend(self, ax: plt.Axes) -> None:
        """Add legend for optical flow vectors."""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='>', color='#333333',
                   linewidth=0, markersize=4, alpha=0.7,
                   label='Flow')
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=10,
            framealpha=0.9
        )
    
    def _save_figure(self, fig: plt.Figure, output_path: Path) -> str:
        """Save figure in configured format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure correct extension
        output_file = output_path.with_suffix(f'.{self.output_format}')
        
        fig.savefig(
            output_file,
            dpi=self.dpi,
            bbox_inches='tight',
            format=self.output_format
        )
        
        plt.close(fig)
        logger.info(f"✓ Plot saved: {output_file}")
        
        return str(output_file)
    
    def plot_reflectivity_with_cells(
        self,
        ds: xr.Dataset,
        frame_offset: int = 0,
        output_path: Optional[Path] = None,
    ) -> str:
        """Two-panel plot: reflectivity + projections."""
        # Extract metadata
        radar_id = ds.attrs.get('radar_id', 'RADAR')
        timestamp = self._extract_timestamp(ds)
        
        # Get reflectivity
        refl_name = self._get_var_name("reflectivity", "reflectivity")
        refl = ds[refl_name].values
        refl_masked = self._mask_reflectivity(refl)
        
        # Get coordinates
        x_coords, y_coords = self._get_coordinates_km(ds)
        
        # Get labels
        labels_name = self._get_var_name("cell_labels", "cell_labels")
        labels = ds[labels_name]
        
        # Create figure
        fig, ax1, ax2 = self._setup_figure()
        
        # ============================================================
        # PANEL 1: Full Reflectivity + Flow Vectors
        # ============================================================
        im1 = self._plot_reflectivity_field(ax1, refl_masked, x_coords, y_coords)
        self._add_colorbar(ax1, im1)
        self._add_basemap(ax1, ds, x_coords, y_coords)
        
        flow_plotted = self._plot_heading_yectors(ax1, ds, x_coords, y_coords)
        
        self._format_axis(ax1, 'Reflectivity + Motion Vectors', timestamp, radar_id)
        
        if flow_plotted:
            self._add_flow_legend(ax1)
        
        # ============================================================
        # PANEL 2: Segmented Cells + Projections
        # ============================================================
        # Mask reflectivity to show only segmented cells
        labels_mask = labels.values > 0
        refl_segmented = np.ma.masked_where(~labels_mask, refl)
        refl_segmented = np.ma.masked_where(refl_segmented < self.min_refl, refl_segmented)
        
        im2 = self._plot_reflectivity_field(ax2, refl_segmented, x_coords, y_coords)
        self._add_colorbar(ax2, im2)
        self._add_basemap(ax2, ds, x_coords, y_coords)
        
        # Add segmentation contours (thin black lines)
        self._plot_segmentation_contours(ax2, labels, x_coords, y_coords)
        
        # Add projection contours (thin transparent gray lines)
        self._plot_projection_contours(ax2, ds, x_coords, y_coords)
        
        self._format_axis(ax2, 'Segmented Cells + Projections', timestamp, radar_id)
        
        # ============================================================
        # Save Figure
        # ============================================================
        plt.tight_layout()
        
        if output_path is None:
            output_path = Path(f"/tmp/radar_plot_{timestamp.strftime('%Y%m%d_%H%M%S')}.{self.output_format}")
        
        return self._save_figure(fig, Path(output_path))
    
    def plot_from_netcdf(
        self,
        segmentation_nc: Path,
        output_path: Optional[Path] = None,
    ) -> str:
        """Plot from analysis NetCDF file."""
        import time
        
        max_retries = 5
        retry_delay = 0.1
        
        seg_path = Path(segmentation_nc)
        
        # Wait for file to exist
        for attempt in range(max_retries):
            if seg_path.exists() and seg_path.stat().st_size > 0:
                break
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                raise FileNotFoundError(f"File not found: {segmentation_nc}")
        
        # Try to open NetCDF
        seg_ds = None
        for attempt in range(max_retries):
            try:
                seg_ds = xr.open_dataset(segmentation_nc)
                break
            except (OSError, FileNotFoundError, RuntimeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise RuntimeError(f"Failed to open NetCDF: {segmentation_nc}") from e
        
        # Validate required variables
        labels_name = self._get_var_name("cell_labels", "cell_labels")
        refl_name = self._get_var_name("reflectivity", "reflectivity")
        
        if labels_name not in seg_ds.data_vars:
            raise ValueError(f"Missing variable: {labels_name}")
        if refl_name not in seg_ds.data_vars:
            raise ValueError(f"Missing variable: {refl_name}")
        
        try:
            plot_file = self.plot_reflectivity_with_cells(
                ds=seg_ds,
                frame_offset=0,
                output_path=output_path,
            )
        finally:
            seg_ds.close()
        
        return plot_file


class PlotterThread(threading.Thread):
    """Threaded radar plotter from queue."""
    
    def __init__(
        self,
        input_queue: queue.Queue,
        output_dirs: Dict,
        config: Dict = None,
        show_plots: bool = False,
        name: str = "RadarPlotter",
    ):
        """Init plotter thread with queue."""
        super().__init__(name=name, daemon=True)
        
        self.input_queue = input_queue
        self.output_dirs = output_dirs
        self.config = config or {}
        self.show_plots = show_plots
        
        self.plotter = RadarPlotter(config=config, show_plots=show_plots)
        self.running = True
        
        logger.info(f"✓ {name} initialized")
    
    def run(self):
        """Thread main loop."""
        logger.info(f"{self.name} started")
        
        while self.running:
            try:
                item = self.input_queue.get(timeout=1.0)
                
                if item is None:
                    logger.info(f"{self.name} received shutdown signal")
                    break
                
                self._process_item(item)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name}: {e}", exc_info=True)
        
        logger.info(f"{self.name} stopped")
    
    def _process_item(self, item: Dict):
        """Process plot item from queue."""
        try:
            seg_nc = item.get('segmentation_nc')
            radar_id = item.get('radar_id', 'RADAR')
            timestamp = item.get('timestamp', datetime.now())
            
            if not seg_nc or not Path(seg_nc).exists():
                logger.warning(f"Segmentation file not found: {seg_nc}")
                return
            
            # Get file_id for tracker
            file_id = Path(seg_nc).stem.replace('_analysis', '').replace('_segmentation', '')
            
            # Use helper for consistent paths
            from adapt.setup_directories import get_plot_path
            
            output_path = get_plot_path(
                output_dirs=self.output_dirs,
                radar_id=radar_id,
                plot_type='reflectivity',
                scan_time=timestamp
            )
            
            plot_file = self.plotter.plot_from_netcdf(
                segmentation_nc=seg_nc,
                output_path=output_path,
            )
            
            logger.info(f"✓ {radar_id} plot saved: {plot_file}")
            
            # Update tracker
            tracker = self.config.get("file_tracker")
            if tracker and plot_file:
                tracker.mark_stage_complete(file_id, "plotted", path=Path(plot_file))
            
        except Exception as e:
            logger.exception(f"Error processing plot item: {e}")
            
            tracker = self.config.get("file_tracker")
            if tracker:
                file_id = Path(item.get('segmentation_nc', '')).stem.replace('_analysis', '').replace('_segmentation', '')
                if file_id:
                    tracker.mark_stage_complete(file_id, "plotted", error=str(e))
    
    def stop(self):
        """Stop plotter thread."""
        self.running = False
        self.input_queue.put(None)


if __name__ == "__main__":
    print("✓ RadarPlotter loaded.")
