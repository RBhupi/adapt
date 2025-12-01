"""Radar cell projection using optical flow."""

import logging
import numpy as np
import xarray as xr
import cv2
from scipy.spatial import Delaunay
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)
# TODO: Implement other projection methods in future.




class RadarCellProjector:
    """Config-driven optical flow projection.
    
    The return of this process will be the second ds from the list, with 
    cell_projections added. The index 0 of the cell_projection is registration 
    from previous frame to current frame. The index from 1 onwards are future projections. 
    The calling class will get the second ds, with the output attached to it as return.
    """

    def __init__(self, config: dict):
        """Store config."""
        self.config = config

    def project(self, ds_list):
        """Project cells forward.
        
        Args:
            ds_list: list of two ds received from segmenter, each containing
                     cell_labels as segmentation labels
                     
        Returns:
            The second (latest) ds with cell_projections added
        """
        if self.config.get("method") == "adapt_default":
            return self._project_opticalflow(ds_list)
        else:
            raise ValueError(f"Unknown projection method: {self.config.get('method')}")


    def _project_opticalflow(self, ds_list):
        """Compute optical flow and project cell labels.

        Receives ds with cell_labels from segmenter. The reflectivity
        slice extraction is handled by processor before calling this.
        """
        nan_fill = self.config.get("nan_fill_value", 0)
        max_interval_minutes = self.config.get("max_time_interval_minutes", 30)
        max_proj_steps = min(self.config.get("max_projection_steps", 1), 10)
        flow_params = self.config.get("flow_params", {
            "pyr_scale": 0.5,
            "levels": 3,
            "winsize": 10,
            "iterations": 3,
            "poly_n": 5,
            "poly_sigma": 1.2,
            "flags": 0
        })

        time_diff = self._validate_datasets(ds_list, max_interval_minutes)

        if time_diff is None:
            # Return second ds without projections
            return ds_list[1]

        logger.debug(f"Computing flow: {time_diff:.1f} min interval")

        # Get reflectivity from ds (already at correct z-level from processor)
        # Reflectivity is always 2D at the configured z-level
        var_names = self.config.get("global", {}).get("var_names", {})
        refl_var = var_names.get("reflectivity", "reflectivity")
        
        refl1 = np.nan_to_num(ds_list[0][refl_var].values, nan=nan_fill).astype(np.float32)
        refl2 = np.nan_to_num(ds_list[1][refl_var].values, nan=nan_fill).astype(np.float32)

        refl1_norm, refl2_norm = self._normalize(refl1, refl2)
        flow = cv2.calcOpticalFlowFarneback(refl1_norm, refl2_norm, None, **flow_params)

        # Get cell_labels from segmenter output
        # For registration (offset=0): project labels from t-1 (ds_list[0]) to t0 (ds_list[1])
        # For future projections (offset=1, 2, 3, n): project labels from t0 (ds_list[1]) forward
        labels_prev = ds_list[0]["cell_labels"].values.astype(np.int32)
        labels_curr = ds_list[1]["cell_labels"].values.astype(np.int32)

        # Generate projections:
        # - First projection (offset=0) is registration: t-1 → t0 (uses labels from t-1)
        # - Subsequent projections (offset=1,2,...) are future: t0→t1, t1→t2, etc. (uses labels from t0)
        # So total projections = max_proj_steps + 1 (1 for registration + N for future)

        labels_proj_list = []

        # Registration - project t-1 labels to t0 position (1 step)
        registration = self._project_frames(labels_prev, flow, n_steps=1)
        labels_proj_list.append(registration[0])
        
        # Future projections - project current labels (t0) forward (n steps)
        # Each pixel carries its original flow value and uses accumulated displacement.
        # @TODO I have removed more complecated  logic of using flow at new positions for each step, 
        # because some cells did not move in noisy radar data during the test. I will test it again later.
        future_projections = self._project_frames(labels_curr, flow, n_steps=max_proj_steps)
        for i in range(max_proj_steps):
            labels_proj_list.append(future_projections[i])

        # Add cell_projections to the second (latest) ds
        ds_out = ds_list[1].copy()
        
        # Stack projections along frame_offset dimension
        # frame_offset=0: registration (projection from t-1 to t0)
        # frame_offset=1,2,...: future projections from t0
        if labels_proj_list:
            frame_offsets = list(range(len(labels_proj_list)))  # 0, 1, 2, ... (0=registration)
            projections = np.stack(labels_proj_list, axis=0)
            
            ds_out["cell_projections"] = xr.DataArray(
                projections,
                dims=["frame_offset", "y", "x"],
                coords={
                    "frame_offset": frame_offsets,
                    "y": ds_out.y,
                    "x": ds_out.x,
                },
                attrs={"description": "Projected cell labels"}
            )
            
            # Also store flow field
            ds_out["heading_x"] = xr.DataArray(
                flow[:, :, 0].astype(np.float32),
                dims=["y", "x"],
                coords={"y": ds_out.y, "x": ds_out.x},
                attrs={"units": "pixels/frame", "description": "Heading in x direction"}
            )
            ds_out["heading_y"] = xr.DataArray(
                flow[:, :, 1].astype(np.float32),
                dims=["y", "x"],
                coords={"y": ds_out.y, "x": ds_out.x},
                attrs={"units": "pixels/frame", "description": "heading in y direction"}
            )
            
            logger.info(f"✓ Added cell_projections with {len(labels_proj_list)} projection steps")

        return ds_out
    
    def _project_frames(self, labels_src, flow, n_steps=1):
        """Project labels for multiple steps, carrying flow with each pixel.
        
        Key concept: Flow is computed at original pixel positions (t-1→t0).
        Each pixel carries its original flow value and uses it for all projection steps.
        This is correct because flow represents the motion vector AT that pixel's location.
        
        Args:
            labels_src: Source labels (H, W) at time t
            flow: Optical flow field (H, W, 2) in pixels/frame
            n_steps: Number of projection steps to compute
            
        Returns:
            projections: (n_steps, H, W) array with projected labels
        """
        H, W = labels_src.shape
        projections = np.full((n_steps, H, W), fill_value=0, dtype=np.int32)

        unique_labels = np.unique(labels_src[labels_src > 0])

        # Sort by area (smallest first) to prevent large cells from overwriting small ones
        label_areas = []
        for label_val in unique_labels:
            area = np.sum(labels_src == label_val)
            label_areas.append((label_val, area))
        label_areas.sort(key=lambda x: x[1])

        # For each cell, extract pixels with their flow values ONCE
        for label_val, _ in label_areas:
            mask = labels_src == label_val
            y_indices, x_indices = np.where(mask)

            # Extract flow at ORIGINAL positions - this travels with the pixel
            cell_pixels = []
            for idx in range(len(y_indices)):
                y = y_indices[idx]
                x = x_indices[idx]

                # Get flow at this pixel's ORIGINAL location
                fx = flow[y, x, 0]
                fy = flow[y, x, 1]

                # If flow is invalid, use zero (pixel doesn't move)
                if not np.isfinite(fx) or not np.isfinite(fy):
                    fx, fy = 0.0, 0.0

                cell_pixels.append((y, x, fx, fy))

            # Project this cell for all steps using SAME flow values
            for step_idx in range(n_steps):
                step = step_idx + 1  # Steps are 1-indexed (1, 2, 3, ...)
                
                for y, x, fx, fy in cell_pixels:
                    # Accumulated displacement: original_pos + flow * step
                    new_x = x + fx * step
                    new_y = y + fy * step

                    new_x_int = int(np.round(new_x))
                    new_y_int = int(np.round(new_y))

                    # Only place if within bounds
                    if 0 <= new_x_int < W and 0 <= new_y_int < H:
                        projections[step_idx, new_y_int, new_x_int] = label_val

        # Fill holes in projected cells using concave hull for each step
        for step_idx in range(n_steps):
            for label_val in unique_labels:
                label_mask = projections[step_idx] == label_val

                if not label_mask.any():
                    continue

                filled_mask = self._fill_concave_hull(label_mask, alpha=0.1)
                projections[step_idx][filled_mask > 0] = label_val

        return projections



    def _validate_datasets(self, ds_list, max_interval_minutes):
        """Validate dataset appropriateness."""
        if len(ds_list) != 2:
            raise ValueError(f"Need exactly 2 datasets, got {len(ds_list)}")

        # Handle both scalar and array time coordinates (2D datasets have scalar time)
        time1_val = ds_list[0].time.values
        time2_val = ds_list[1].time.values
        time1 = time1_val if np.ndim(time1_val) == 0 else time1_val[0]
        time2 = time2_val if np.ndim(time2_val) == 0 else time2_val[0]
        
        time_diff_minutes = (time2 - time1) / np.timedelta64(1, 'm')

        if abs(time_diff_minutes) > max_interval_minutes:
            logger.warning(f"⚠️ Time interval {time_diff_minutes:.1f} min exceeds max {max_interval_minutes} min. Skipping projection.")
            return None

        return time_diff_minutes

    def _normalize(self, refl1, refl2):
        """Normalize to uint8."""
        vmin = min(refl1.min(), refl2.min())
        vmax = max(refl1.max(), refl2.max())

        if vmax > vmin:
            refl1_norm = np.uint8(255 * (refl1 - vmin) / (vmax - vmin))
            refl2_norm = np.uint8(255 * (refl2 - vmin) / (vmax - vmin))
        else:
            refl1_norm = np.uint8(refl1)
            refl2_norm = np.uint8(refl2)

        return refl1_norm, refl2_norm

    def _fill_concave_hull(self, label_mask, alpha=0.1):
        """Fill concave hull using alpha shapes.

        Args:
            label_mask: Binary mask of projected points
            alpha: Controls tightness (lower = tighter, higher = more convex)
                   Typical range: 0.05-0.3
        """
        if not label_mask.any():
            return label_mask

        # Get coordinates of projected points
        points = np.argwhere(label_mask)

        if len(points) < 4:
            # Too few points for triangulation, use dilation
            kernel = np.ones((3, 3), dtype=np.uint8)
            return binary_dilation(label_mask, structure=kernel).astype(np.uint8)

        # Swap to (x, y) for Delaunay
        points = points[:, [1, 0]]

        try:
            # Compute Delaunay triangulation
            tri = Delaunay(points)

            # Create output mask
            filled = np.zeros_like(label_mask, dtype=np.uint8)
            H, W = label_mask.shape

            # Filter triangles by circumradius (alpha shape)
            for simplex in tri.simplices:
                # Get triangle vertices
                pts = points[simplex]

                # Compute circumradius
                a = np.linalg.norm(pts[1] - pts[0])
                b = np.linalg.norm(pts[2] - pts[1])
                c = np.linalg.norm(pts[0] - pts[2])

                # Semi-perimeter
                s = (a + b + c) / 2.0
                area = np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))

                if area > 1e-10:
                    circumradius = (a * b * c) / (4.0 * area)

                    # Keep triangle if circumradius < 1/alpha
                    if circumradius < 1.0 / alpha:
                        # Fill triangle using cv2
                        triangle = pts.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillConvexPoly(filled, triangle, 1)

            return filled.astype(np.uint8)

        except Exception as e:
            logger.warning(f"Concave hull failed: {e}, falling back to dilation")
            kernel = np.ones((3, 3), dtype=np.uint8)
            return binary_dilation(label_mask, structure=kernel).astype(np.uint8)

    def _apply_closing(self, labels_proj):
        """Apply closing."""
        from scipy.ndimage import binary_closing

        labels_closed = labels_proj.copy()
        unique_labels = np.unique(labels_proj[~np.isnan(labels_proj)])

        for label in unique_labels:
            if label <= 0:
                continue

            label_mask = (labels_proj == label) & ~np.isnan(labels_proj)
            kernel = np.ones((3, 3), dtype=np.uint8)
            closed = binary_closing(label_mask.astype(np.uint8), structure=kernel)
            labels_closed[closed > 0] = label

        return labels_closed

