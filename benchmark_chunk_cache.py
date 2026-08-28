"""Benchmark: plain dask vs windowed arrays vs cached chunk arrays.

Uses the ds_2d_left_agrid.zarr dataset with 10,000 particles.
"""

import time as time_mod

import numpy as np
import xarray as xr

import parcels
import parcels._sgrid as sgrid


def make_fieldset(ds: xr.Dataset) -> parcels.FieldSet:
    """Build a FieldSet from the 2D left A-grid zarr dataset."""
    ds = ds.copy()
    ds["lon"].attrs["units"] = "m"
    ds["lat"].attrs["units"] = "m"
    ds = ds.pipe(
        sgrid._attach_sgrid_metadata,
        sgrid.SGrid2DMetadata(
            cf_role="grid_topology",
            topology_dimension=2,
            node_dimensions=("XG", "YG"),
            node_coordinates=("lon", "lat"),
            face_dimensions=(
                sgrid.FaceNodePadding("XC", "XG", sgrid.Padding.LOW),
                sgrid.FaceNodePadding("YC", "YG", sgrid.Padding.LOW),
            ),
            vertical_dimensions=(sgrid.FaceNodePadding("ZC", "ZG", sgrid.Padding.LOW),),
        ),
    )
    return parcels.FieldSet.from_sgrid_conventions(
        ds,
        vector_fields={"UV": ("U_A_grid", "V_A_grid")},
        skip_field_data_validation=True,
    )


def delete_on_boundary(particles, fieldset):
    """Delete particles that hit the boundary instead of erroring."""
    particles.state = np.where(
        particles.state == parcels.StatusCode.ErrorOutOfBounds,
        parcels.StatusCode.Delete,
        particles.state,
    )


def run_simulation(fieldset, ds, npart, label):
    """Run a simulation and return elapsed time."""
    np.random.seed(42)
    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.Particle,
        t=np.full(npart, ds.time.values[0]),
        z=np.full(npart, 1),
        y=np.random.uniform(1.0, 5.0, npart),
        x=np.random.uniform(1.0, 5.0, npart),
    )

    t0 = time_mod.perf_counter()
    pset.execute(
        [parcels.kernels.AdvectionRK2, delete_on_boundary],
        runtime=np.timedelta64(100, "ms"),
        dt=np.timedelta64(10, "ms"),
    )
    elapsed = time_mod.perf_counter() - t0
    alive = np.sum(pset.state != parcels.StatusCode.Delete)
    print(f"  {label}: {elapsed:.3f}s ({alive}/{npart} particles alive)")
    return elapsed


def main():
    zarr_path = "../xarray-interpolation/datasets/ds_2d_left_agrid.zarr"
    npart = 10_000

    print(f"Loading dataset from {zarr_path}")
    ds = xr.open_zarr(zarr_path, consolidated=False)
    print(f"  shape: {dict(ds.dims)}")
    print(f"  chunks: U_A_grid {ds['U_A_grid'].encoding.get('chunks', 'N/A')}")

    # --- 1. Plain dask ---
    print("\n1. Plain dask")
    fieldset_dask = make_fieldset(ds)
    run_simulation(fieldset_dask, ds, npart, "plain dask")

    # --- 2. Windowed arrays ---
    print("\n2. Windowed arrays")
    fieldset_windowed = make_fieldset(ds)
    fieldset_windowed.to_windowed_arrays()
    run_simulation(fieldset_windowed, ds, npart, "windowed")

    # --- 3. Cached chunk arrays ---
    print("\n3. Cached chunk arrays")
    fieldset_cached = make_fieldset(ds)
    fieldset_cached.to_cached_chunk_arrays()
    run_simulation(fieldset_cached, ds, npart, "cached chunks")


if __name__ == "__main__":
    main()
