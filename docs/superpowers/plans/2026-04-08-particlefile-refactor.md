# ParticleFile Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `src/parcels/_core/particlefile.py` to eliminate redundant state, split the monolithic `_write_particle_data` method, and fix two small module-level issues — all without changing public behaviour.

**Architecture:** The single file `particlefile.py` is modified in-place across four incremental tasks, each independently verifiable by running the test suite. No new files are created. Public API is unchanged.

**Tech Stack:** Python, zarr, xarray, numpy. Tests via `pixi run pytest tests/test_particlefile.py`.

---

## File Map

| File                                | Change              |
| ----------------------------------- | ------------------- |
| `src/parcels/_core/particlefile.py` | All changes         |
| `tests/test_particlefile.py`        | No changes required |

---

### Task 1: Establish baseline

**Files:**

- Run: `tests/test_particlefile.py`

- [ ] **Step 1: Run the test suite and confirm it passes**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: all non-xfail tests pass. Note any pre-existing failures so they are not mistaken for regressions.

---

### Task 2: Remove `_maxids` and rename internal flag to `_initialized`

**Files:**

- Modify: `src/parcels/_core/particlefile.py`

The field `_maxids` is always equal to `len(self._pids_written)`. The flag `_create_new_zarrfile` is internally used as a "have we written yet?" boolean — rename it to `_initialized` (its negation) to make that intent clear. The public property `create_new_zarrfile` is kept, returning `not self._initialized`.

- [ ] **Step 1: Update `__init__` — remove `_maxids`, replace `_create_new_zarrfile` with `_initialized`**

In `src/parcels/_core/particlefile.py`, replace:

```python
        self._maxids = 0
        self._pids_written = {}
        self.metadata = {}
        self._create_new_zarrfile = create_new_zarrfile
```

with:

```python
        self._pids_written = {}
        self.metadata = {}
        self._initialized = not create_new_zarrfile
```

- [ ] **Step 2: Update the `create_new_zarrfile` property**

Replace:

```python
    @property
    def create_new_zarrfile(self):
        return self._create_new_zarrfile
```

with:

```python
    @property
    def create_new_zarrfile(self):
        return not self._initialized
```

- [ ] **Step 3: Update `_write_particle_data` — replace `_maxids` uses and `_create_new_zarrfile` flag**

Replace:

```python
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = self._maxids + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)
        self._maxids = len(self._pids_written)
```

with:

```python
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        start = len(self._pids_written)
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = start + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)
```

Replace the two occurrences of `self._maxids` in the `if self.create_new_zarrfile:` branch:

```python
        if self.create_new_zarrfile:
            if self.chunks is None:
                self._chunks = (nparticles, 1)
            if (self._maxids > len(ids)) or (self._maxids > self.chunks[0]):
                arrsize = (self._maxids, self.chunks[1])
            else:
                arrsize = (len(ids), self.chunks[1])
```

with:

```python
        if self.create_new_zarrfile:
            if self.chunks is None:
                self._chunks = (nparticles, 1)
            n_unique = len(self._pids_written)
            if (n_unique > len(ids)) or (n_unique > self.chunks[0]):
                arrsize = (n_unique, self.chunks[1])
            else:
                arrsize = (len(ids), self.chunks[1])
```

Replace:

```python
            obs = np.zeros((self._maxids), dtype=np.int32)
```

with:

```python
            obs = np.zeros((len(self._pids_written),), dtype=np.int32)
```

Replace (at end of `if self.create_new_zarrfile` branch):

```python
            ds.to_zarr(store, mode="w")
            self._create_new_zarrfile = False
```

with:

```python
            ds.to_zarr(store, mode="w")
            self._initialized = True
```

Also replace the `else` branch reference:

```python
        else:
            Z = zarr.group(store=store, overwrite=False)
            obs = particle_data["obs_written"][indices_to_write]
            for var in vars_to_write:
                if self._maxids > Z[var.name].shape[0]:
```

with:

```python
        else:
            Z = zarr.group(store=store, overwrite=False)
            obs = particle_data["obs_written"][indices_to_write]
            for var in vars_to_write:
                if len(self._pids_written) > Z[var.name].shape[0]:
```

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: same results as Task 1 baseline.

- [ ] **Step 5: Commit**

```bash
git add src/parcels/_core/particlefile.py
git commit -m "refactor(particlefile): remove _maxids field, rename _create_new_zarrfile to _initialized

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 3: Fix `_create_variables_attribute_dict` — eliminate duplicate variable filter

**Files:**

- Modify: `src/parcels/_core/particlefile.py`

`_create_variables_attribute_dict` currently replicates the `[var for var in particle.variables if var.to_write is not False]` filter that `_get_vars_to_write` already encapsulates. Change its signature to accept the already-filtered list directly.

- [ ] **Step 1: Replace the function signature and body**

Replace:

```python
def _create_variables_attribute_dict(particle: ParticleClass, time_interval: TimeInterval) -> dict:
    """Creates the dictionary with variable attributes.

    Notes
    -----
    For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
    """
    attrs = {}

    vars = [var for var in particle.variables if var.to_write is not False]
    for var in vars:
        fill_value = {"_FillValue": _DATATYPES_TO_FILL_VALUES[var.dtype]}

        attrs[var.name] = {**var.attrs, **fill_value}

    attrs["time"].update(_get_calendar_and_units(time_interval))

    return attrs
```

with:

```python
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
```

- [ ] **Step 2: Update the call site in `_write_particle_data`**

Replace:

```python
            attrs = _create_variables_attribute_dict(pclass, time_interval)
```

with:

```python
            attrs = _create_variables_attribute_dict(vars_to_write, time_interval)
```

- [ ] **Step 3: Remove the now-unused `ParticleClass` import in the `TYPE_CHECKING` block**

Check whether `ParticleClass` is still used elsewhere in the file:

```bash
grep -n "ParticleClass" src/parcels/_core/particlefile.py
```

If the only remaining use is in `_get_vars_to_write`'s type annotation, keep the import. If it appears nowhere else after the signature change, remove the import line:

```python
from parcels._core.particle import ParticleClass
```

(Note: `ParticleClass` is still used in `_get_vars_to_write(particle: ParticleClass)`, so the import stays.)

- [ ] **Step 4: Run tests**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: same results as baseline.

- [ ] **Step 5: Commit**

```bash
git add src/parcels/_core/particlefile.py
git commit -m "refactor(particlefile): fix _create_variables_attribute_dict to accept vars_to_write directly

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 4: Split `_extend_zarr_dims` into `_extend_trajectories` and `_extend_observations`

**Files:**

- Modify: `src/parcels/_core/particlefile.py`

The current `_extend_zarr_dims(Z, dtype, axis)` method handles two completely different operations based on the `axis` argument. Replace it with two single-purpose methods.

- [ ] **Step 1: Replace `_extend_zarr_dims` with two focused methods**

Replace the entire `_extend_zarr_dims` method:

```python
    def _extend_zarr_dims(self, Z, dtype, axis):  # noqa: N803
        if axis == 1:
            a = np.full((Z.shape[0], self.chunks[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            obs = zarr.group(store=self.store, overwrite=False)["obs"]
            if len(obs) == Z.shape[1]:
                obs.append(np.arange(self.chunks[1]) + obs[-1] + 1)
        else:
            extra_trajs = self._maxids - Z.shape[0]
            if len(Z.shape) == 2:
                a = np.full((extra_trajs, Z.shape[1]), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
            else:
                a = np.full((extra_trajs,), _DATATYPES_TO_FILL_VALUES[dtype], dtype=dtype)
        Z.append(a, axis=axis)
        zarr.consolidate_metadata(self.store)
```

with:

```python
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
```

- [ ] **Step 2: Update the two call sites in `_write_particle_data`**

Replace:

```python
                if self._maxids > Z[var.name].shape[0]:
                    self._extend_zarr_dims(Z[var.name], dtype=var.dtype, axis=0)
```

with:

```python
                if len(self._pids_written) > Z[var.name].shape[0]:
                    self._extend_trajectories(Z[var.name], dtype=var.dtype)
```

Replace:

```python
                    if max(obs) >= Z[var.name].shape[1]:
                        self._extend_zarr_dims(Z[var.name], dtype=var.dtype, axis=1)
```

with:

```python
                    if max(obs) >= Z[var.name].shape[1]:
                        self._extend_observations(Z[var.name], dtype=var.dtype)
```

- [ ] **Step 3: Run tests**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: same results as baseline.

- [ ] **Step 4: Commit**

```bash
git add src/parcels/_core/particlefile.py
git commit -m "refactor(particlefile): split _extend_zarr_dims into _extend_trajectories and _extend_observations

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 5: Split `_write_particle_data` into dispatcher + `_initial_write` + `_append_write`

**Files:**

- Modify: `src/parcels/_core/particlefile.py`

This is the core of the refactor. `_write_particle_data` becomes a thin dispatcher (~20 lines) that handles ID tracking and always-computed `ids_once`/`indices_once`, then delegates to `_initial_write` or `_append_write`. The `obs` naming confusion is resolved by renaming to `obs_indices` throughout.

- [ ] **Step 1: Replace the entire `_write_particle_data` method and add `_initial_write` and `_append_write`**

Replace the entire `_write_particle_data` method (lines 163–241 in the original):

```python
    def _write_particle_data(self, *, particle_data, pclass, time_interval, time, indices=None):
        # if pset._data._ncount == 0:
        #     warnings.warn(
        #         f"ParticleSet is empty on writing as array at time {time:g}",
        #         RuntimeWarning,
        #         stacklevel=2,
        #     )
        #     return
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
        to_add = sorted(set(pids) - set(self._pids_written.keys()))
        start = len(self._pids_written)
        for i, pid in enumerate(to_add):
            self._pids_written[pid] = start + i
        ids = np.array([self._pids_written[p] for p in pids], dtype=int)

        once_ids = np.where(particle_data["obs_written"][indices_to_write] == 0)[0]
        if len(once_ids) > 0:
            ids_once = ids[once_ids]
            indices_to_write_once = indices_to_write[once_ids]

        store = self.store
        if self.create_new_zarrfile:
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
            obs = np.zeros((len(self._pids_written),), dtype=np.int32)
            for var in vars_to_write:
                if var.name not in ["trajectory"]:  # because 'trajectory' is written as coordinate
                    if var.to_write == "once":
                        data = np.full(
                            (arrsize[0],),
                            _DATATYPES_TO_FILL_VALUES[var.dtype],
                            dtype=var.dtype,
                        )
                        data[ids_once] = particle_data[var.name][indices_to_write_once]
                        dims = ["trajectory"]
                    else:
                        data = np.full(arrsize, _DATATYPES_TO_FILL_VALUES[var.dtype], dtype=var.dtype)
                        data[ids, 0] = particle_data[var.name][indices_to_write]
                        dims = ["trajectory", "obs"]
                    ds[var.name] = xr.DataArray(data=data, dims=dims, attrs=attrs[var.name])
                    ds[var.name].encoding["chunks"] = self.chunks[0] if var.to_write == "once" else self.chunks
            ds.to_zarr(store, mode="w")
            self._initialized = True
        else:
            Z = zarr.group(store=store, overwrite=False)
            obs = particle_data["obs_written"][indices_to_write]
            for var in vars_to_write:
                if len(self._pids_written) > Z[var.name].shape[0]:
                    self._extend_trajectories(Z[var.name], dtype=var.dtype)
                if var.to_write == "once":
                    if len(once_ids) > 0:
                        Z[var.name].vindex[ids_once] = particle_data[var.name][indices_to_write_once]
                else:
                    if max(obs) >= Z[var.name].shape[1]:
                        self._extend_observations(Z[var.name], dtype=var.dtype)
                    Z[var.name].vindex[ids, obs] = particle_data[var.name][indices_to_write]

        particle_data["obs_written"][indices_to_write] = obs + 1
```

with:

```python
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
        obs_indices = particle_data["obs_written"][indices_to_write]

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

    def _initial_write(self, *, ids, ids_once, indices_to_write, indices_once, vars_to_write, pids, nparticles, particle_data, time_interval):
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

    def _append_write(self, *, ids, obs_indices, ids_once, indices_once, indices_to_write, vars_to_write, particle_data):
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
```

- [ ] **Step 2: Run tests**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: same results as baseline.

- [ ] **Step 3: Commit**

```bash
git add src/parcels/_core/particlefile.py
git commit -m "refactor(particlefile): split _write_particle_data into dispatcher + _initial_write + _append_write

- ids_once/indices_once are now always computed (empty arrays if no once-vars)
- obs renamed to obs_indices to eliminate naming collision between branches

Co-authored-by: Claude <noreply@anthropic.com>"
```

---

### Task 6: Verify final state

**Files:**

- Run: `tests/test_particlefile.py`

- [ ] **Step 1: Run the full test suite**

```bash
pixi run pytest tests/test_particlefile.py -v --tb=short
```

Expected: same results as Task 1 baseline. All non-xfail tests pass.

- [ ] **Step 2: Confirm no references to removed identifiers remain**

```bash
grep -n "_maxids\|_create_new_zarrfile\|_extend_zarr_dims" src/parcels/_core/particlefile.py
```

Expected: no output.
