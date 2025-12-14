from pathlib import Path
import numpy as np
import xarray as xr


def write_fake_segmentation_netcdf(
    path: Path,
    with_labels: bool = True,
):
    data_vars = {
        "reflectivity": (("y", "x"), np.ones((4, 4), dtype="float32")),
    }

    if with_labels:
        labels = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
        ], dtype="int32")
        data_vars["cell_labels"] = (("y", "x"), labels)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "y": np.arange(4),
            "x": np.arange(4),
        },
        attrs={
            "z_level_m": 2000,
            "radar_id": "TEST",
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    ds.close()

    return path
