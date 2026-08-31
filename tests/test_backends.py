import io
from pathlib import Path
from typing import Literal

import pytest
import xarray as xr

import parcels
import parcels.tutorial

BackendT = Literal["WindowedArray", "Dask", "Zarr", "NumPy", "CachedChunkArray"]
BACKENDS = {"WindowedArray", "Dask", "Zarr", "NumPy", "CachedChunkArray"}


@pytest.fixture(scope="module")
def nemo_dataset() -> xr.Dataset:
    ds_u = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/U")
    ds_v = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/V")
    ds_w = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/W")
    ds_coords = parcels.tutorial.open_dataset("NemoNorthSeaORCA025-N006_data/mesh_mask")[["glamf", "gphif"]]

    ds_fset = parcels.convert.nemo_to_sgrid(
        fields={"U": ds_u["uo"], "V": ds_v["vo"], "W": ds_w["wo"]},
        coords=ds_coords,
    )
    return ds_fset


@pytest.fixture(scope="module")
def nemo_results(tmp_parquet, nemo_dataset) -> tuple[xr.Dataset, Path]:
    run_simulation(nemo_dataset, tmp_parquet, "NumPy")
    return nemo_dataset, tmp_parquet


def assert_fieldset_backend(fset: parcels.FieldSet, backend: BackendT):
    # a bit of a hacky way to check for the backend.... probably better for us to change how backends are stored
    buf = io.StringIO()
    fset.describe(buf)
    return backend in buf.getvalue()


def run_simulation(ds: xr.Dataset, output_path: Path, backend: BackendT) -> Path:
    if backend == "Zarr":
        raise NotImplementedError("Doesn't work at this level of execution. Also will likely remove Zarr backend.")

    if backend == "NumPy":
        ds.load()

    fset = parcels.FieldSet.from_sgrid_conventions(ds)

    if backend == "WindowedArray":
        fset.to_windowed_arrays()
    if backend == "CachedChunkArray":
        fset.to_cached_chunk_arrays()

    assert_fieldset_backend(fset, backend)

    # TODO Create the particleset by seeding 1000 particles
    ...

    # TODO Advect the particles using a RK4 advection kernel, saving the results to the Parquet file
    ...

    return output_path


@pytest.mark.parametrize(
    "backend",
    BACKENDS
    - {
        "NumPY",  # reference point
        "Zarr",  # not supported
    },
)
def test_nemo_identical_across_backends(nemo_results, tmp_parquet, backend):
    ds = nemo_results[0]
    ref_parquet = nemo_results[1]

    assert str(ref_parquet) != str(tmp_parquet)  # just covering my bases with Pytest fixture usage

    run_simulation(ds, tmp_parquet, backend)

    # TODO: Compare the tmp_parquet with the reference parquet
