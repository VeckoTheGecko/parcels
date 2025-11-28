import numpy as np
import pytest
import xarray as xr
import xgcm

from parcels._core.xgrid import _DEFAULT_XGCM_KWARGS
from parcels._datasets.structured.generic import datasets
from parcels._datasets.utils import to_strict_array
from tests.utils import assert_backed_by_strict_array


def test_left_indexed_dataset():
    """Checks that 'ds_2d_left' is right indexed on all variables."""
    ds = datasets["ds_2d_left"]
    grid = xgcm.Grid(ds, **_DEFAULT_XGCM_KWARGS)

    for _axis_name, axis in grid.axes.items():
        for pos, _dim_name in axis.coords.items():
            assert pos in ["left", "center"]


def test_right_indexed_dataset():
    """Checks that 'ds_2d_right' is right indexed on all variables."""
    ds = datasets["ds_2d_right"]
    grid = xgcm.Grid(ds, **_DEFAULT_XGCM_KWARGS)
    for _axis_name, axis in grid.axes.items():
        for pos, _dim_name in axis.coords.items():
            assert pos in ["center", "right"]


@pytest.mark.parametrize(
    "ds",
    [
        xr.Dataset({"var": (("x", "y"), np.arange(20).reshape(4, 5))}),
    ],
)
def test_to_strict_array_api(ds):
    """Updates the arrays used in a dataset to use the strict array API.

    Ensures when the dataset is used during testing that no non-strict array API features are used.
    """
    ds = to_strict_array(ds)
    xr.testing.assert_equal(ds, ds)
    assert_backed_by_strict_array(ds)
