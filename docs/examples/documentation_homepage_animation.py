#!/usr/bin/env python
# coding: utf-8

# # The homepage animation

# In[1]:


# %matplotlib qt
import copy

import cartopy
import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter, writers

# Load in particle data, define the time range at which to plot (**`plottimes`**) and select the indices of the first timestep in the variable `'b'`.
#

# In[2]:


filename = "medusarun.nc"
pfile = xr.open_dataset(str(filename), decode_cf=True)
lon = np.ma.filled(pfile.variables["lon"], np.nan)
lat = np.ma.filled(pfile.variables["lat"], np.nan)
time = np.ma.filled(pfile.variables["time"], np.nan)

pfile.close()

plottimes = np.arange(time[0, 0], np.nanmax(time), np.timedelta64(10, "D"))
starttime = 20
b = time == plottimes[0 + starttime]

# The particle data in `medusarun.nc` is available at https://surfdrive.surf.nl/files/index.php/s/lwjW5w05jtHuYz9
#

# The animation consists of three figures: the northern hemisphere, the southern hemisphere and the oceanparcels logo. To organize their locations we use [matplotlib.gridspec](https://matplotlib.org/tutorials/intermediate/gridspec.html). The animation spans 12 frames and updates the particle positions based on the timestep in `plottimes`.
#

# In[3]:


fig = plt.figure(figsize=(8, 4))
gs = gridspec.GridSpec(ncols=8, nrows=4, figure=fig)

### Northern Hemisphere
ax1 = fig.add_subplot(
    gs[:, :4],
    projection=ccrs.NearsidePerspective(
        central_latitude=90, central_longitude=-30, satellite_height=15000000
    ),
)
ax1.set_facecolor("#1EB7D0")
ax1.add_feature(cartopy.feature.LAND, zorder=1)
ax1.coastlines()
scat1 = ax1.scatter(
    lon[b],
    lat[b],
    marker=".",
    s=25,
    c="#AB2200",
    edgecolor="white",
    linewidth=0.15,
    transform=ccrs.PlateCarree(),
)

### Southern Hemisphere
ax2 = fig.add_subplot(
    gs[:, 4:],
    projection=ccrs.NearsidePerspective(
        central_latitude=-90, central_longitude=-30, satellite_height=15000000
    ),
)
ax2.set_facecolor("#1EB7D0")
ax2.add_feature(cartopy.feature.LAND, zorder=1)
ax2.coastlines()
scat2 = ax2.scatter(
    lon[b],
    lat[b],
    marker=".",
    s=25,
    c="#AB2200",
    edgecolor="white",
    linewidth=0.15,
    transform=ccrs.PlateCarree(),
)

frames = np.arange(0, 20)


def animate(t):
    b = time == plottimes[t + starttime]
    scat1.set_offsets(np.vstack((lon[b], lat[b])).transpose())
    scat2.set_offsets(np.vstack((lon[b], lat[b])).transpose())
    return scat1, scat2


anim = animation.FuncAnimation(fig, animate, frames=frames, interval=150, blit=True)
anim

# needed for tight_layout to work with cartopy
fig.canvas.draw()
plt.tight_layout()
# writergif = PillowWriter(fps=6)
# anim.save('homepageshort.gif',writer=writergif)

# The resulting animation is then
#
# ![gif](images/homepage.gif)
