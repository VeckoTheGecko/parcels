# ParticleFile Refactor Design

**Date:** 2026-04-08
**File:** `src/parcels/_core/particlefile.py`

## Summary

Simplify and refactor `particlefile.py` by splitting the monolithic `_write_particle_data` method, eliminating redundant state, and fixing two small module-level issues. Public API is unchanged.

---

## Problems Being Fixed

1. **`_write_particle_data` is an 80-line monolith** with two deeply nested code paths (initial write vs. append) that are hard to read independently.
2. **`ids_once` / `indices_to_write_once` have conditional initialization** — set only when `len(once_ids) > 0`, referenced in both branches. Fragile.
3. **`obs` variable is overloaded** — means `np.zeros(...)` in the initial-write branch, and `particle_data["obs_written"][...]` in the append branch. The same name flows into a shared post-if assignment (line 241), making the logic invisible.
4. **`_maxids` is redundant state** — always equals `len(self._pids_written)`.
5. **`_create_variables_attribute_dict` duplicates the variable filter** from `_get_vars_to_write`.
6. **`_extend_zarr_dims` mixes two unrelated axis operations** under one method, with an `if axis == 1` branch that makes each path harder to understand in isolation.
7. **`_create_new_zarrfile` is a mutable flag** that would be clearer as `_initialized: bool`.

---

## Architecture

Public API unchanged: `ParticleFile.__init__`, `.write()`, `.set_metadata()`, all properties.

### New internal shape

```
_write_particle_data(...)         ← dispatcher: ID tracking, ids_once setup, branch
  ├── _initial_write(...)         ← creates xr.Dataset, writes to zarr, sets _initialized=True
  └── _append_write(...)          ← opens zarr group, extends dims, writes data

_extend_trajectories(Z, dtype)    ← was: _extend_zarr_dims(..., axis=0)
_extend_observations(Z, dtype)    ← was: _extend_zarr_dims(..., axis=1)
```

---

## Detailed Changes

### State

| Before                            | After                                                             |
| --------------------------------- | ----------------------------------------------------------------- |
| `self._maxids: int`               | Removed — use `len(self._pids_written)` inline                    |
| `self._create_new_zarrfile: bool` | Renamed to `self._initialized: bool = not create_new_zarrfile`    |
| Property `create_new_zarrfile`    | Returns `not self._initialized` (same semantics, tests unchanged) |

### `_write_particle_data` (dispatcher, ~20 lines)

Responsibilities:

- ID tracking: update `_pids_written`, compute `ids`
- Unconditionally compute `ids_once` and `indices_to_write_once` (empty arrays when no once-vars, not conditionally defined)
- Read `obs_indices = particle_data["obs_written"][indices_to_write]`
- Branch to `_initial_write` or `_append_write`
- Update `particle_data["obs_written"][indices_to_write] = obs_indices + 1` after branch

### `_initial_write(...)`

Parameters: `ids, obs_indices, ids_once, indices_once, vars_to_write, pids, nparticles, time_interval`

Responsibilities:

- Compute `arrsize` using `len(self._pids_written)` (not `_maxids`)
- Build `xr.Dataset` and write to zarr with `mode="w"`
- Set `self._initialized = True`

### `_append_write(...)`

Parameters: `ids, obs_indices, ids_once, indices_once, vars_to_write`

Responsibilities:

- Open zarr group
- Loop over vars; call `_extend_trajectories` or `_extend_observations` as needed
- Write data via `vindex`

### `_extend_trajectories(Z, dtype)` (~5 lines)

Extends axis 0 by `len(self._pids_written) - Z.shape[0]` rows. Was the `else` branch of `_extend_zarr_dims`.

### `_extend_observations(Z, dtype)` (~6 lines)

Extends axis 1 by `chunks[1]` columns and appends to the `obs` coordinate. Was the `if axis == 1` branch of `_extend_zarr_dims`.

### Module-level: `_create_variables_attribute_dict`

Replace the inline `[var for var in particle.variables if var.to_write is not False]` with a call to `_get_vars_to_write(particle)`.

---

## What Is Not Changing

- `_get_store_from_pathlike` — no changes
- `_to_write_particles` — no changes
- `_get_calendar_and_units` — no changes
- `_assert_valid_chunks_tuple` — no changes
- `_DATATYPES_TO_FILL_VALUES` — no changes
- All tests should pass without modification

---

## Test Strategy

Run the existing test suite: `pixi run pytest tests/test_particlefile.py`

No new tests required — the refactor is behaviour-preserving. The existing tests cover:

- Initial write and append paths
- `to_write="once"` variables
- Trajectory extension (particle removal and re-addition)
- Observation extension (chunked output)
- All dtypes
- `create_new_zarrfile` property read-only check
