"""Module controlling the writing of ParticleSets to Zarr file."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import cftime
import numpy as np
import xarray as xr
import zarr
from zarr.storage import DirectoryStore

import parcels
from parcels._core.particle import ParticleClass
from parcels._core.utils.time import timedelta_to_float
from parcels._reprs import particlefile_repr

if TYPE_CHECKING:
    from parcels._core.particle import Variable
    from parcels._core.particleset import ParticleSet
    from parcels._core.utils.time import TimeInterval

__all__ = ["ParticleFile"]

_DATATYPES_TO_FILL_VALUES = {
    np.dtype(np.float16): np.nan,
    np.dtype(np.float32): np.nan,
    np.dtype(np.float64): np.nan,
    np.dtype(np.bool_): np.iinfo(np.int8).max,
    np.dtype(np.int8): np.iinfo(np.int8).max,
    np.dtype(np.int16): np.iinfo(np.int16).max,
    np.dtype(np.int32): np.iinfo(np.int32).max,
    np.dtype(np.int64): np.iinfo(np.int64).min,
    np.dtype(np.uint8): np.iinfo(np.uint8).max,
    np.dtype(np.uint16): np.iinfo(np.uint16).max,
    np.dtype(np.uint32): np.iinfo(np.uint32).max,
    np.dtype(np.uint64): np.iinfo(np.uint64).max,
}


class ParticleFile:
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file. This can also be a Zarr store object.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a numpy.timedelta64, a datimetime.timedelta object or a positive float (in seconds).
    chunks :
        Tuple (trajs, obs) to control the size of chunks in the zarr output.
    create_new_zarrfile : bool
        Whether to create a new file. Default is True

    Returns
    -------
    ParticleFile
        ParticleFile object that can be used to write particle data to file
    """

    def __init__(self, store, outputdt, chunks=None, create_new_zarrfile=True):
        if not isinstance(outputdt, (np.timedelta64, timedelta, float)):
            raise ValueError(
                f"Expected outputdt to be a np.timedelta64, datetime.timedelta or float (in seconds), got {type(outputdt)}"
            )

        outputdt = timedelta_to_float(outputdt)

        if outputdt <= 0:
            raise ValueError(f"outputdt must be positive/non-zero. Got {outputdt=!r}")

        self._outputdt = outputdt

        _assert_valid_chunks_tuple(chunks)
        self._chunks = chunks
        self._pids_written = {}
        self.metadata = {}
        self._initialized = not create_new_zarrfile

        if not isinstance(store, zarr.storage.Store):
            store = _get_store_from_pathlike(store)

        self._store = store

        # TODO v4: Enable once updating to zarr v3
        # if store.read_only:
        #     raise ValueError(f"Store {store} is read-only. Please provide a writable store.")

        # TODO v4: Add check that if create_new_zarrfile is False, the store already exists

    def __repr__(self) -> str:
        return particlefile_repr(self)

    def set_metadata(self, parcels_grid_mesh: Literal["spherical", "flat"]):
        self.metadata.update(
            {
                "feature_type": "trajectory",
                "Conventions": "CF-1.6/CF-1.7",
                "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                "parcels_version": parcels.__version__,
                "parcels_grid_mesh": parcels_grid_mesh,
            }
        )

    @property
    def outputdt(self):
        return self._outputdt

    @property
    def chunks(self):
        return self._chunks

    @property
    def store(self):
        return self._store

    @property
    def create_new_zarrfile(self):
        return not self._initialized

    def _extend_trajectories(self, Z, dtype):  # noqa: N803
        extra_trajs = len(self._pids_written) - Z.shape[0]
        if len(Z.shape) == 2:
            a = np.full((extra_trajs, Z.shape[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
        else:
            a = np.full((extra_trajs,), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
        Z.append(a, axis=0)
        zarr.consolidate_metadata(self.store)

    def _extend_observations(self, Z, dtype):  # noqa: N803
        a = np.full((Z.shape[0], self.chunks[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
        obs = zarr.group(store=self.store, overwrite=False)["obs"]
        if len(obs) == Z.shape[1]:
            obs.append(np.arange(self.chunks[1]) + obs[-1] + 1)
        Z.append(a, axis=1)
        zarr.consolidate_metadata(self.store)

    def write(self, pset: ParticleSet, time, indices=None):
        """Write all data from one time step to the zarr file,
        before the particle locations are updated.

        Parameters
        ----------
        pset :
            ParticleSet object to write
        time :
            Time at which to write ParticleSet (same time object as fieldset)
        """
        pclass = pset._ptype
        time_interval = pset.fieldset.time_interval
        particle_data = pset._data

        self._write_particle_data(
            particle_data=particle_data, pclass=pclass, time_interval=time_interval, time=time, indices=indices
        )

    def _write_particle_data(self, *, particle_data, pclass, time_interval, time, indices=None):
        if isinstance(time, (np.timedelta64, np.datetime64)):
            time = timedelta_to_float(time - time_interval.left)
        nparticles = len(particle_data["trajectory"])
        vars_to_write = _get_vars_to_write(pclass)

        if indices is None:
            indices_to_write = _to_write_particles(particle_data, time)
        else:
            indices_to_write = indices

        if len(indices_to_write) == 0:
            return

        pids = particle_data["trajectory"][indices_to_write]
        start = len(self._pids_written)
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = start + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)

        once_ids = np.where(particle_data["obs_written"][indices_to_write] == 0)[0]
        ids_once = ids[once_ids]
        indices_once = indices_to_write[once_ids]
        obs_indices = particle_data["obs_written"][
            indices_to_write
        ]  # always 0 on initial write; updated by caller after both branches

        if not self._initialized:
            self._initial_write(
                ids=ids,
                ids_once=ids_once,
                indices_to_write=indices_to_write,
                indices_once=indices_once,
                vars_to_write=vars_to_write,
                pids=pids,
                nparticles=nparticles,
                particle_data=particle_data,
                time_interval=time_interval,
            )
        else:
            self._append_write(
                ids=ids,
                obs_indices=obs_indices,
                ids_once=ids_once,
                indices_once=indices_once,
                indices_to_write=indices_to_write,
                vars_to_write=vars_to_write,
                particle_data=particle_data,
            )

        particle_data["obs_written"][indices_to_write] = obs_indices + 1

    def _initial_write(
        self,
        *,
        ids,
        ids_once,
        indices_to_write,
        indices_once,
        vars_to_write,
        pids,
        nparticles,
        particle_data,
        time_interval,
    ):
        if self.chunks is None:
            self._chunks = (nparticles, 1)
        n_unique = len(self._pids_written)
        if (n_unique > len(ids)) or (n_unique > self.chunks[0]):
            arrsize = (n_unique, self.chunks[1])
        else:
            arrsize = (len(ids), self.chunks[1])

        ds = xr.Dataset(
            attrs=self.metadata,
            coords={"trajectory": ("trajectory", pids), "obs": ("obs", np.arange(arrsize[1], dtype=np.int32))},
        )
        attrs = _create_variables_attribute_dict(vars_to_write, time_interval)
        for var in vars_to_write:
            if var.name != "trajectory":  # 'trajectory' is written as coordinate
                if var.to_write == "once":
                    data = np.full((arrsize[0],), _DATATYPES_TO_FILL_VALUES[var.dtype], dtype=var.dtype)
                    data[ids_once] = particle_data[var.name][indices_once]
                    dims = ["trajectory"]
                else:
                    data = np.full(arrsize, _DATATYPES_TO_FILL_VALUES[var.dtype], dtype=var.dtype)
                    data[ids, 0] = particle_data[var.name][indices_to_write]
                    dims = ["trajectory", "obs"]
                ds[var.name] = xr.DataArray(data=data, dims=dims, attrs=attrs[var.name])
                ds[var.name].encoding["chunks"] = self.chunks[0] if var.to_write == "once" else self.chunks
        ds.to_zarr(self.store, mode="w")
        self._initialized = True

    def _append_write(
        self, *, ids, obs_indices, ids_once, indices_once, indices_to_write, vars_to_write, particle_data
    ):
        # obs_indices is a snapshot from the caller; caller updates obs_written after this returns
        Z = zarr.group(store=self.store, overwrite=False)
        for var in vars_to_write:
            if len(self._pids_written) > Z[var.name].shape[0]:
                self._extend_trajectories(Z[var.name], dtype=var.dtype)
            if var.to_write == "once":
                if len(ids_once) > 0:
                    Z[var.name].vindex[ids_once] = particle_data[var.name][indices_once]
            else:
                if max(obs_indices) >= Z[var.name].shape[1]:
                    self._extend_observations(Z[var.name], dtype=var.dtype)
                Z[var.name].vindex[ids, obs_indices] = particle_data[var.name][indices_to_write]


def _get_store_from_pathlike(path: Path | str) -> DirectoryStore:
    path = str(Path(path))  # Ensure valid path, and convert to string
    extension = os.path.splitext(path)[1]
    if extension != ".zarr":
        raise ValueError(f"ParticleFile name must end with '.zarr' extension. Got path {path!r}.")

    return DirectoryStore(path)


def _get_vars_to_write(particle: ParticleClass) -> list[Variable]:
    return [v for v in particle.variables if v.to_write is not False]


def _create_variables_attribute_dict(vars_to_write: list[Variable], time_interval: TimeInterval) -> dict:
    """Creates the dictionary with variable attributes.

    Notes
    -----
    For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
    """
    attrs = {}

    for var in vars_to_write:
        fill_value = {"_FillValue": _DATATYPES_TO_FILL_VALUES[var.dtype]}
        attrs[var.name] = {**var.attrs, **fill_value}

    attrs["time"].update(_get_calendar_and_units(time_interval))

    return attrs


def _to_write_particles(particle_data, time):
    """Return the Particles that need to be written at time: if particle.time is between time-dt/2 and time+dt (/2)"""
    return np.where(
        (
            np.less_equal(
                time - np.abs(particle_data["dt"] / 2),
                particle_data["time"],
                where=np.isfinite(particle_data["time"]),
            )
            & np.greater_equal(
                time + np.abs(particle_data["dt"] / 2),
                particle_data["time"],
                where=np.isfinite(particle_data["time"]),
            )  # check time - dt/2 <= particle_data["time"] <= time + dt/2
            | (
                (np.isnan(particle_data["dt"]))
                & np.equal(time, particle_data["time"], where=np.isfinite(particle_data["time"]))
            )  # or dt is NaN and time matches particle_data["time"]
        )
        & (np.isfinite(particle_data["trajectory"]))
        & (np.isfinite(particle_data["time"]))
    )[0]


def _get_calendar_and_units(time_interval: TimeInterval) -> dict[str, str]:
    calendar = None
    units = "seconds"
    if time_interval:
        if isinstance(time_interval.left, (np.datetime64, datetime)):
            calendar = "standard"
        elif isinstance(time_interval.left, cftime.datetime):
            calendar = time_interval.left.calendar

    if calendar is not None:
        units += f" since {time_interval.left}"

    attrs = {"units": units}
    if calendar is not None:
        attrs["calendar"] = calendar

    return attrs


def _assert_valid_chunks_tuple(chunks):
    e = ValueError(f"chunks must be a tuple of integers with length 2, got {chunks=!r} instead.")
    if chunks is None:
        return

    if not isinstance(chunks, tuple):
        raise e
    if len(chunks) != 2:
        raise e
    if not all(isinstance(c, int) for c in chunks):
        raise e
