#!/usr/bin/env python

# # Curvilinear grids
#

# Parcels also supports [curvilinear grids](https://www.nemo-ocean.eu/doc/node108.html) such as those used in the [NEMO models](https://www.nemo-ocean.eu/).
#
# We will be using the example data in the `NemoCurvilinear_data/` directory. These fields are a purely zonal flow on an aqua-planet (so zonal-velocity is 1 m/s and meridional-velocity is 0 m/s everywhere, and no land). However, because of the curvilinear grid, the `U` and `V` fields vary north of 20N.
#

# In[1]:


from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import parcels

# We can create a `FieldSet` just like we do for normal grids.
# Note that NEMO is discretised on a C-grid. U and V velocities are not located on the same nodes (see https://www.nemo-ocean.eu/doc/node19.html ).
#
# ```
#  __V1__
# |      |
# U0     U1
# |__V0__|
# ```
#
# To interpolate U, V velocities on the C-grid, Parcels needs to read the f-nodes, which are located on the corners of the cells (for indexing details: https://www.nemo-ocean.eu/doc/img360.png ).
#

# In[2]:


example_dataset_folder = parcels.download_example_dataset("NemoCurvilinear_data")
filenames = {
    "U": {
        "lon": f"{example_dataset_folder}/mesh_mask.nc4",
        "lat": f"{example_dataset_folder}/mesh_mask.nc4",
        "data": f"{example_dataset_folder}/U_purely_zonal-ORCA025_grid_U.nc4",
    },
    "V": {
        "lon": f"{example_dataset_folder}/mesh_mask.nc4",
        "lat": f"{example_dataset_folder}/mesh_mask.nc4",
        "data": f"{example_dataset_folder}/V_purely_zonal-ORCA025_grid_V.nc4",
    },
}
variables = {"U": "U", "V": "V"}

dimensions = {"lon": "glamf", "lat": "gphif", "time": "time_counter"}

fieldset = parcels.FieldSet.from_nemo(
    filenames, variables, dimensions, allow_time_extrapolation=True
)

# And we can plot the `U` field.
#

# In[3]:


plt.pcolormesh(
    fieldset.U.grid.lon,
    fieldset.U.grid.lat,
    fieldset.U.data[0, :, :],
    vmin=0,
    vmax=1,
)
plt.colorbar()
plt.show()

# As you see above, the `U` field indeed is 1 m/s south of 20N, but varies with longitude and latitude north of that. We can confirm by doing a field evaluation at (60N, 50E):

# In[4]:


u, v = fieldset.UV.eval(0, 0, 60, 50, applyConversion=False)
print(f"(u, v) = ({u:.3f}, {v:.3f})")
assert np.isclose(u, 1.0, atol=1e-3)

# Now we can run particles as normal. Parcels will take care to rotate the `U` and `V` fields.

# In[5]:


# Start 20 particles on a meridional line at 180W
npart = 20
lonp = -180 * np.ones(npart)
latp = [i for i in np.linspace(-70, 85, npart)]

pset = parcels.ParticleSet.from_list(fieldset, parcels.JITParticle, lon=lonp, lat=latp)
pfile = parcels.ParticleFile("nemo_particles", pset, outputdt=timedelta(days=1))

pset.execute(
    parcels.AdvectionRK4,
    runtime=timedelta(days=30),
    dt=timedelta(hours=6),
    output_file=pfile,
)

# And then we can plot these trajectories. As expected, all trajectories go exactly zonal and due to the curvature of the earth, ones at higher latitude move more degrees eastward (even though the distance in km is equal for all particles).
#

# In[6]:


ds = xr.open_zarr("nemo_particles.zarr")

plt.plot(ds.lon.T, ds.lat.T, ".-")
plt.show()

# ## Speeding up `ParticleSet` initialisation by efficiently finding particle start-locations on the `Grid`
#
# On a Curvilinear grid, determining the location of each `Particle` on the grid is more complicated and therefore takes longer than on a Rectilinear grid. Since Parcels version 2.2.2, a function is available on the `ParticleSet` class, that speeds up the look-up. After creating the `ParticleSet`, but before running the `ParticleSet.execute()`, simply call the function `ParticleSet.populate_indices()`.
#

# In[7]:


pset = parcels.ParticleSet.from_list(fieldset, parcels.JITParticle, lon=lonp, lat=latp)
pset.populate_indices()
