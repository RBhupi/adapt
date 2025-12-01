import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RadarCellSegmenter:
    """Config-driven segmentation for threaded pipelines."""

    def __init__(self, config: dict):
        """Store config.
        
        Parameters
        ----------
        config : dict
            Segmenter config with 'global' section for var_names/coord_names,
            and segmentation parameters like 'threshold', 'method', etc.
        """
        self.config = config
        self.method = config.get("method", "threshold")
        
        # Pre-load config parameters
        self.threshold = config.get("threshold", 30)
        # @TODO I used closing, may not required now. I will test it again later.
        self.kernel_size = config.get("closing_kernel", (2, 2))
        self.filter_by_size = config.get("filter_by_size", True)
        self.min_gridpoints = config.get("min_cellsize_gridpoint", 5)
        self.max_gridpoints = config.get("max_cellsize_gridpoint", None)

        logger.info("RadarCellSegmenter initialized: method=%s, threshold=%s", 
                    self.method, self.threshold)

    def segment(self, ds: xr.Dataset) -> xr.Dataset:
        """Segment and attach labels."""
        if self.method == "threshold":
            return self._segment2D_threshold(ds)
        else:
            raise ValueError(f"Unknown segmentation method: {self.method}")

    def _segment2D_threshold(self, ds: xr.Dataset) -> xr.Dataset:
        """Threshold and label 2D reflectivity data.
        
        Expects 2D dataset (already sliced at z-level by processor).
        """
        # Get global settings
        global_cfg = self.config.get("global", {})
        var_names = global_cfg.get("var_names", {})

        # Get variable names
        refl_name = var_names.get("reflectivity", "reflectivity")
        labels_name = var_names.get("cell_labels", "cell_labels")

        # Extract reflectivity (already 2D)
        refl = ds[refl_name].values

        binary_mask = refl > self.threshold
        labels = self._binary_to_labels(
            binary_mask, 
            self.kernel_size, 
            self.filter_by_size, 
            self.min_gridpoints, 
            self.max_gridpoints
        )

        # Get z_level from dataset attrs (set by processor)
        z_level_m = ds.attrs.get("z_level_m", 2000)

        # Build attrs dict, excluding None values (NetCDF can't serialize None)
        attrs = {
            "long_name": "Cell segmentation labels",
            "units": "1",
            "method": self.method,
            "threshold_dbz": self.threshold,
            "z_level_m": z_level_m,
            "min_cellsize_gridpoint": self.min_gridpoints,
        }
        if self.max_gridpoints is not None:
            attrs["max_cellsize_gridpoint"] = self.max_gridpoints

        labels_da = xr.DataArray(
            labels,
            dims=("y", "x"),
            coords={"y": ds.y, "x": ds.x},
            attrs=attrs
        )

        # we attach labels to original dataset
        ds_out = ds.copy()
        ds_out[labels_name] = labels_da
        logger.debug(f"Labels attached: var={labels_name}, shape={labels.shape}, max={labels.max()}")

        return ds_out

    def _binary_to_labels(self, binary_mask: np.ndarray, kernel_size: tuple,
                          filter_by_size: bool, min_gridpoints: int, max_gridpoints: int) -> np.ndarray:
        """Morphology, label, filter."""
        from skimage.morphology import closing, footprint_rectangle
        from skimage.measure import label

        closed_mask = closing(binary_mask, footprint_rectangle(kernel_size))

        labels = label(closed_mask)

        # if there are any cells, filter and/or renumber
        if labels.max() > 0:
            labels = self._filter_and_relabel(labels, filter_by_size, min_gridpoints, max_gridpoints)

        return labels.astype(np.int32)

    def _filter_and_relabel(self, labels: np.ndarray, filter_by_size: bool,
                             min_gridpoints: int, max_gridpoints: int) -> np.ndarray:
        """Filter, renumber by size."""
        labels_unique, counts = np.unique(labels, return_counts=True)
        keep_mask = labels_unique > 0

        if filter_by_size:
            if min_gridpoints > 1:
                keep_mask &= (counts >= min_gridpoints)
                num_small = np.sum((labels_unique > 0) & (counts < min_gridpoints))
                if num_small > 0:
                    logger.debug(f"Removed {num_small} small (< {min_gridpoints})")

            if max_gridpoints is not None:
                keep_mask &= (counts <= max_gridpoints)
                num_large = np.sum((labels_unique > 0) & (counts > max_gridpoints))
                if num_large > 0:
                    logger.debug(f"Removed {num_large} large (> {max_gridpoints})")

        labels_to_keep = labels_unique[keep_mask]
        labels_renumbered = self._relabel_by_size(labels, labels_to_keep, counts)

        num_kept = len(labels_to_keep)
        num_removed = len(labels_unique) - 1 - num_kept
        if filter_by_size and num_removed > 0:
            logger.debug(f"Kept {num_kept}, removed {num_removed}")

        return labels_renumbered

    def _relabel_by_size(self, labels: np.ndarray, labels_to_keep: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Renumber: largest=1."""
        keep_indices = np.isin(np.arange(len(counts)), labels_to_keep)
        keep_counts = counts[keep_indices]

        sort_indices = np.argsort(-keep_counts)
        labels_sorted = labels_to_keep[sort_indices]

        old_to_new = np.zeros(labels.max() + 1, dtype=np.int32)
        old_to_new[labels_sorted] = np.arange(1, len(labels_sorted) + 1)

        return old_to_new[labels]
