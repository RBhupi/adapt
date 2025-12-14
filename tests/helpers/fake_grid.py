import numpy as np
import xarray as xr


def make_fake_grid_ds(
    time_len=1,
    z_levels=(0, 1000, 2000),
    shape=(5, 5),
    variables=("reflectivity",),
):
    """
    Create a minimal Py-ART-like grid dataset suitable
    for processor integration tests.
    """

    time = np.array(["2025-01-01"], dtype="datetime64[ns]")
    z = np.array(z_levels)
    y = np.arange(shape[0])
    x = np.arange(shape[1])

    data_vars = {}
    for var in variables:
        data = np.random.rand(time_len, len(z), shape[0], shape[1]) * 50
        data_vars[var] = (("time", "z", "y", "x"), data)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": time, "z": z, "y": y, "x": x},
        attrs={
            "radar_latitude": 40.0,
            "radar_longitude": -100.0,
            "radar_altitude": 100.0,
        },
    )

    return ds
