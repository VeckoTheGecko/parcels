#!/usr/bin/env python

# # TimeStamps and calendars
#

# In[1]:


import warnings
from glob import glob

import numpy as np

import parcels

# Some NetCDF files, such as for example those from the [World Ocean Atlas](https://www.nodc.noaa.gov/OC5/woa18/), have time calendars that can't be parsed by `xarray`. These result in a `ValueError: unable to decode time units`, for example when the calendar is in 'months since' a particular date.
#
# In these cases, a workaround in Parcels is to use the `timestamps` argument in `Field` (or `FieldSet`) creation. Here, we show how this works for example temperature data from the World Ocean Atlas in the Pacific Ocean
#

# The following cell raises an error, since the calendar of the World Ocean Atlas data is in "months since 1955-01-01 00:00:00"
#

# In[2]:


example_dataset_folder = parcels.download_example_dataset("WOA_data")
tempfield = parcels.Field.from_netcdf(
    glob(f"{example_dataset_folder}/woa18_decav_*_04.nc"),
    "t_an",
    {"lon": "lon", "lat": "lat", "time": "time"},
)

# However, we can create our own numpy array of timestamps associated with each of the 12 snapshots in the netcdf file
#

# In[3]:


timestamps = np.expand_dims(
    np.array([np.datetime64(f"2001-{m:02d}-15") for m in range(1, 13)]), axis=1
)

# And then we can add the `timestamps` as an extra argument
#

# In[4]:


with warnings.catch_warnings():
    warnings.simplefilter("ignore", parcels.FileWarning)
    tempfield = parcels.Field.from_netcdf(
        glob(f"{example_dataset_folder}/woa18_decav_*_04.nc"),
        "t_an",
        {"lon": "lon", "lat": "lat", "time": "time"},
        timestamps=timestamps,
    )

# Note, by the way, that adding the `time_periodic` argument to `Field.from_netcdf()` will also mean that the climatology can be cycled for multiple years.
#

# Furthermore, note that we used `warnings.catch_warnings()` with `warnings.simplefilter("ignore", parcels.FileWarning)` to wrap the `FieldSet.from_nemo()` call above. This is to silence an expected warning because the time dimension in the `coordinates.nc` file can't be decoded by `xarray`.
