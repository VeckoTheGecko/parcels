#!/usr/bin/env python
# coding: utf-8

# # Particle-Particle interaction
#

# In this notebook, we show an example of the new 'particle-particle-interaction' functionality in Parcels. Note that this functionality is still in development, and the implementation is fairly rudimentary and slow for now. Importantly:
#
# - Particle-particle interaction only works in Scipy mode
# - The type of interactions that are supported is still limited
#
# Interactions are implemented through `InteractionKernels`, which are similar to normal `Kernels`. The `InteractionKernels` are applied between particles that are located closer to each other than a specified `interaction_distance`. In general, the code structure needs three adaptations to apply particle-particle interaction:
#
# 1. The `ParticleSet` requires an `interaction_distance` argument upon creation, to define the `interaction_distance`.
# 2. `ParticleSet.execute()` requires the `pyfunc_inter` argument, which contains the `InteractionKernels` that will be executed, similarly to the `pyfunc` argument for normal `Kernels`.
# 3. `InteractionKernels` have two additional arguments compared to normal `Kernels`:
#
# ```python
# def InteractionKernel(particle, fieldset, time, neighbors, mutator)
# ```
#
# The `neighbors` argument provides a list of the particles that are within a neighborhood (i.e. closer than the `interaction_distance` argument in `ParticleSet` creation).
#
# The `mutator` argument is an initially empty list with all the mutations that need to be performed on particles at the end of running all `InteractionKernels` on all particles.
# This `mutator` argument is required, because otherwise the order at which interactions are applied has implications for the simulation. As a consequence, the simulation will likely be dependent on the order of the particle list if no mutator list is used.
#

# ## Pulling particles
#
# Below is an example of what can be done with particle-particle interaction. We create a square grid of $N\times N$ particles, which are all subject to Brownian Motion (via the built-in `DiffusionUniformKh` Kernel). Furthermore, some of the particles also 'attract' other particles that are within the interaction distance: these attracted particles move with a constant velocity to the attracting particles.
#

# In[1]:


%matplotlib notebook
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

import parcels

# In[2]:


def Pull(particle, fieldset, time, neighbors, mutator):
    """InterActionKernel that "pulls" all neighbor particles
    toward the attracting particle with a constant velocity"""
    distances = []
    na_neighbors = []
    # only execute kernel for particles that are 'attractor'
    if not particle.attractor:
        return StateCode.Success
    for n in neighbors:
        if n.attractor:
            continue
        x_p = np.array([particle.depth, particle.lat, particle.lon])
        x_n = np.array([n.depth, n.lat, n.lon])

        # compute distance between attracted and attracting particle
        distances.append(np.linalg.norm(x_p - x_n))
        na_neighbors.append(n)

    velocity = 0.04  # predefined attracting velocity
    for n in na_neighbors:
        assert n.dt == particle.dt
        dx = np.array(
            [particle.lat - n.lat, particle.lon - n.lon, particle.depth - n.depth]
        )
        dx_norm = np.linalg.norm(dx)

        # calculate vector of position change
        distance = velocity * n.dt
        d_vec = distance * dx / dx_norm

        # define mutation function for mutator
        def f(n, dlat, dlon, ddepth):
            n.lat_nextloop += (
                dlat  # note that we need to change the locations for the next loop
            )
            n.lon_nextloop += dlon
            n.depth_nextloop += ddepth

        # add mutation to the mutator
        mutator[n.id].append((f, d_vec))

# In[3]:


npart = 11

X, Y = np.meshgrid(np.linspace(-1, 1, npart), np.linspace(-1, 1, npart))

# Define a fieldset without flow
fieldset = parcels.FieldSet.from_data(
    {"U": 0, "V": 0}, {"lon": 0, "lat": 0}, mesh="flat"
)
fieldset.add_constant_field("Kh_zonal", 0.0005, mesh="flat")
fieldset.add_constant_field("Kh_meridional", 0.0005, mesh="flat")


# Create custom particle class with extra variable that indicates
# whether the interaction kernel should be executed on this particle.
InteractingParticle = parcels.ScipyParticle.add_variable(
    "attractor", dtype=np.bool_, to_write="once"
)


attractor = [
    True if i in [int(npart * npart / 2 - 3), int(npart * npart / 2 + 3)] else False
    for i in range(npart * npart)
]
pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=InteractingParticle,
    lon=X,
    lat=Y,
    interaction_distance=0.5,  # note the interaction_distance argument here
    attractor=attractor,
)

output_file = pset.ParticleFile(name="InteractingParticles.zarr", outputdt=1)

pset.execute(
    pyfunc=parcels.DiffusionUniformKh,
    pyfunc_inter=Pull,  # note the pyfunc_inter here
    runtime=60,
    dt=1,
    output_file=output_file,
)

# In[4]:


# %%capture
data_xarray = xr.open_zarr("InteractingParticles.zarr")
data_attr = data_xarray.where(data_xarray["attractor"].compute() == 1, drop=True)
data_other = data_xarray.where(data_xarray["attractor"].compute() == 0, drop=True)

timerange = np.arange(
    np.nanmin(data_xarray["time"].values),
    np.nanmax(data_xarray["time"].values),
    np.timedelta64(1, "s"),  # timerange in nanoseconds
)

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
ax = fig.add_subplot()

ax.set_ylabel("Meridional distance [m]")
ax.set_xlabel("Zonal distance [m]")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

time_id = np.where(
    data_other["time"] == timerange[0]
)  # Indices of the data where time = 0
time_id_attr = np.where(
    data_attr["time"] == timerange[0]
)  # Indices of the data where time = 0

scatter = ax.scatter(
    data_other["lon"].values[time_id],
    data_other["lat"].values[time_id],
    c="b",
    s=5,
    zorder=1,
)
scatter_attr = ax.scatter(
    data_attr["lon"].values[time_id_attr],
    data_attr["lat"].values[time_id_attr],
    c="r",
    s=40,
    zorder=2,
)

circs = []
for lon_a, lat_a in zip(
    data_attr["lon"].values[time_id_attr], data_attr["lat"].values[time_id_attr]
):
    circs.append(
        ax.add_patch(
            plt.Circle(
                (lon_a, lat_a), 0.25, facecolor="None", edgecolor="r", linestyle="--"
            )
        )
    )

t = str(timerange[0].astype("timedelta64[s]"))
title = ax.set_title("Particles at t = " + t + " (Red particles are attractors)")


def animate(i):
    t = str(timerange[i].astype("timedelta64[s]"))
    title.set_text("Particles at t = " + t + "\n (Red particles are attractors)")

    time_id = np.where(data_other["time"] == timerange[i])
    time_id_attr = np.where(data_attr["time"] == timerange[i])
    scatter.set_offsets(
        np.c_[data_other["lon"].values[time_id], data_other["lat"].values[time_id]]
    )
    scatter_attr.set_offsets(
        np.c_[
            data_attr["lon"].values[time_id_attr], data_attr["lat"].values[time_id_attr]
        ]
    )
    for c, lon_a, lat_a in zip(
        circs,
        data_attr["lon"].values[time_id_attr],
        data_attr["lat"].values[time_id_attr],
    ):
        c.center = (lon_a, lat_a)
    return (
        scatter,
        scatter_attr,
        circs,
    )


anim = FuncAnimation(fig, animate, frames=len(timerange), interval=200, blit=True)
data_xarray.close()

# In[5]:


HTML(anim.to_jshtml())

# ## Merging particles
#
# Another type of interaction that is supported is the merging of particles. The supported merging functions also comes with limitations (only mutual-nearest particles can be accurately merged), so this is really just a prototype. Nevertheless, the example below shows the possibilities that merging of particles can provide for more complex simulations.
#
# In the example below, we use two build-in Kernels: `NearestNeighborWithinRange` and `MergeWithNearestNeighbor`.
#

# In[6]:


npart = 800

X = np.random.uniform(-1, 1, size=npart)
Y = np.random.uniform(-1, 1, size=npart)

# Define a fieldset without flow
fieldset = parcels.FieldSet.from_data(
    {"U": 0, "V": 0}, {"lon": 0, "lat": 0}, mesh="flat"
)
fieldset.add_constant_field("Kh_zonal", 0.0005, mesh="flat")
fieldset.add_constant_field("Kh_meridional", 0.0005, mesh="flat")


# Create custom InteractionParticle class
# with extra variables nearest_neighbor and mass
MergeParticle = parcels.ScipyInteractionParticle.add_variables(
    [
        parcels.Variable("nearest_neighbor", dtype=np.int64, to_write=False),
        parcels.Variable("mass", initial=1, dtype=np.float32),
    ]
)

pset = parcels.ParticleSet(
    fieldset=fieldset,
    pclass=MergeParticle,
    lon=X,
    lat=Y,
    interaction_distance=0.05,  # note this argument here
)

output_file = pset.ParticleFile(name="MergingParticles.zarr", outputdt=1)

pset.execute(
    pyfunc=parcels.DiffusionUniformKh,
    pyfunc_inter=pset.InteractionKernel(parcels.NearestNeighborWithinRange)
    + parcels.MergeWithNearestNeighbor,  # note the pyfunc_inter here
    runtime=60,
    dt=1,
    output_file=output_file,
)

# In[7]:


# %%capture
data_xarray = xr.open_zarr("MergingParticles.zarr")

timerange = np.arange(
    np.nanmin(data_xarray["time"].values),
    np.nanmax(data_xarray["time"].values),
    np.timedelta64(1, "s"),
)  # timerange in nanoseconds

fig = plt.figure(figsize=(4, 4), constrained_layout=True)
ax = fig.add_subplot()

ax.set_ylabel("Meridional distance [m]")
ax.set_xlabel("Zonal distance [m]")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)

time_id = np.where(
    data_xarray["time"] == timerange[0]
)  # Indices of the data where time = 0

scatter = ax.scatter(
    data_xarray["lon"].values[time_id],
    data_xarray["lat"].values[time_id],
    s=data_xarray["mass"].values[time_id],
    c="b",
    zorder=1,
)

t = str(timerange[0].astype("timedelta64[s]"))
title = ax.set_title("Particles at t = " + t)


def animate(i):
    t = str(timerange[i].astype("timedelta64[s]"))
    title.set_text("Particles at t = " + t)

    time_id = np.where(data_xarray["time"] == timerange[i])
    scatter.set_offsets(
        np.c_[data_xarray["lon"].values[time_id], data_xarray["lat"].values[time_id]]
    )
    scatter.set_sizes(data_xarray["mass"].values[time_id])

    return (scatter,)


anim = FuncAnimation(fig, animate, frames=len(timerange), interval=200, blit=True)
data_xarray.close()

# In[8]:


HTML(anim.to_jshtml())

# ## Interacting with the FieldSet
#
# An important feature of Parcels is to evaluate a `Field` at the `Particle` location using the square bracket notation: `particle.Temperature = fieldset.T[time, depth, lat, lon]`. These types of particle-field interactions are recommended to be implemented in standard `Kernels`, since the `InteractionKernels` do not report the `StateCodes` that are used to flag particles that encounter an error in the particle-field interaction, e.g. `OutOfBoundsError`. Any variable that is needed in the particle-particle interaction can be stored in a `Variable` by sampling the field in a `Kernel` before executing the `InteractionKernel`.
#
