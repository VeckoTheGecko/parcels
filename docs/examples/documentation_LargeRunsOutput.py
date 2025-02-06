#!/usr/bin/env python

# # Dealing with large output files
#

# You might imagine that if you followed the instructions [on the making of parallel runs](https://docs.oceanparcels.org/en/latest/examples/documentation_MPI.html) and the loading of the resulting dataset, you could just use the `dataset.to_zarr()` function to save the data to a single `zarr` datastore. This is true for small enough datasets. However, there is a bug in `xarray` which makes this inefficient for large data sets.
#
# At the same time, it will often improve performance if large datasets are saved as a single `zarr` store, chunked appropriately, and the type of the variables in them modified. It is often also useful to add other variables to the dataset. This document describes how to do all this.
#

# ## Why are we doing this? And what chunk sizes should we choose?
#

# If you are running a relatively small case (perhaps 1/10 the size of the memory of your machine), nearly anything you do will work. However, as your problems get larger, it can help to write the data into a single zarr datastore, and to chunk that store appropriately.
#
# To illustrate this, here is the time it takes to retrieve all the results (with `ds['variableName'].values`) of some common data structures with different chunk sizes. (What is a chunk size? More on that below). The data in this example has 39 million trajectories starting over 120 times, and there are 250 observations, resulting in a directory size of 88Gb in double precision and 39 in single. In this table, "trajectory:5e4, obs:10" indicates that each chunk extends over 50,000 trajectories and 10 obs. The chunking in the original data is roughly a few thousand observations and 10 obs.
#
# | File type               | open [s] | read 1 obs, all traj [s] | read 23 obs, all traj [s] | read 8000 contiguous traj, all obs [s] | read traj that start at a given time, all obs [s] |
# | ----------------------- | -------- | ------------------------ | ------------------------- | ------------------------------------- | ------------------------------------------------- |
# | Straigth from parcels   | 2.9      | 8.4                      | 59.9                      | 1.5                                   | 17.4                                              |
# | trajectory:5e4, obs:10  | 0.48     | 2.5                      | 19.5                      | 0.4                                   | 10.33                                             |
# | trajectory:5e4, obs:100 | 0.55     | 20.5                     | 13.8                      | 0.5                                   | 3.88                                              |
# | trajectory:5e5, obs:10  | 0.54     | 2.2                      | 16.3                      | 0.85                                  | 18.5                                              |
# | trajectory:5e5, obs:100 | 0.46     | 19.9                     | 40.0                      | 0.62                                  | 49.36                                             |
#
# You can see several things in this. It is always quicker to open a single file, and for all data access patterns, there is are chunksizes that are more efficient than the default output. Why is this?
#
# The chunksize determines how data is stored on disk. For the default zarr datastore, each chunk of data is stored as a single compressed file. In netCDF, chunking is similar except that the compressed data is stored within a single file. In either case, if you must access any data from within a chunk, you must read the entire chunk from disk.
#
# So when we access one obs dimension and many trajectories, the chunking scheme that is elongated in the trajectory direction is fastest. When we get all the observation for a scattered set of trajectories, the chunking that is elongated in observations is the best. In general, the product of the two chunksizes (the number of data points in a chunk) should be hundreds of thousands to 10s of millions. A suboptimal chunking scheme is usually not tragic, but if you know how you will most often access the data, you can save considerable time.
#

# ## How to save the output of an MPI ocean parcels run to a single zarr dataset
#

# First, we need to import the necessary modules, specify the directory `inputDir` which contains the output of the parcels run (the directory that has proc01, proc02 and so forth), the location of the output zarr file `outputDir` and a dictionary giving the chunk size for the `trajectory` and `obs` coordinates, `chunksize`.
#

# In[ ]:


import time
from glob import glob
from os import path

import xarray as xr
from dask.diagnostics import ProgressBar
from numpy import *
from pylab import *

# first specify the directory in which the MPI code wrote its output
inputDir = (
    "dataPathsTemp/"
    + "theAmericas_wholeGlobe_range100km_depthFrom_1m_to_500m_habitatTree_months01_to_02_fixed_1m/"
    + "2007/tracks.zarr"
)


# specify chunksize and where the output zarr file should go; also set chunksize of output file
chunksize = {"trajectory": 5 * int(1e4), "obs": 10}
outputDir = "/home/pringle/jnkData/singleFile_5e4_X_10_example.zarr"

# Now for large datasets, this code can take a while to run; for 36 million trajectories and 250 observations, it can take an hour and a half. I prefer not to accidentally destroy data that takes more than an hour to create, so I put in a safety check and only let the code run if the output directory does not exist.
#

# In[ ]:


# do not overwrite existing data sets
if path.exists(outputDir):
    print("the ouput path", outputDir, "exists")
    print("please delete if you want to replace it")
    assert False, "stopping execution"

# It will often be useful to change the [dtype](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html) of the output data. Doing so can save a great deal of disk space. For example, the input data for this example is 88Gb in size, but by changing lat, lon and z to single precision, I can make the file about half as big.
#
# This comes at the cost of some accuracy. Float64 has 14 digits of accuracy, float32 has 7. For latitude and longitude, going from float64 to float32 increases the error by the circumference of the Earth divided 1e7, or about 1m. This is good enough for what I am doing. However, a year of time has about 3.15e7 seconds, and we often want to know within a second when a particle is released (to avoid floating point issues when picking out particles that start at a specific time). So the 3.15e7/1e7 error (a few seconds) in the time coordinate could cause problems. So I don't want to reduce the precision of time.
#
# There is one other important issue. Due to a bug in xarray, it is much slower to save datasets with a datetime64 variable in them. So time here will be given as float64. If (as we do below) the attribute data is preserved, it will still appear as a datetime type when the data file is loaded
#
# To change precision, put an entry into the dictionary `varType` whose key is the name of the variable, and whose value is the type you wish the variable to be cast to:
#

# In[14]:


varType = {
    "lat": dtype("float32"),
    "lon": dtype("float32"),
    "time": dtype("float64"),  # to avoid bug in xarray
    "z": dtype("float32"),
}

# Now we need to read in the data as discussed in the section on making an MPI run. However, note that `xr.open_zarr()` is given the `decode_times=False` option, which prevents the time variable from being converted into a datetime64[ns] object. This is necessary due to a bug in xarray. As discussed above, when the data set is read back in, time will again be interpreted as a datetime.
#

# In[15]:


print("opening data from multiple process files")
tic = time.time()
files = glob(path.join(inputDir, "proc*"))
dataIn = xr.concat(
    [xr.open_zarr(f, decode_times=False) for f in files],
    dim="trajectory",
    compat="no_conflicts",
    coords="minimal",
)
print("   done opening in %5.2f" % (time.time() - tic))

# Now we can take advantage of the `.astype` operator to change the type of the variables. This is a lazy operator, and it will only be applied to the data when the data values are requested below, when the data is written to a new zarr store.
#

# In[16]:


for v in varType.keys():
    dataIn[v] = dataIn[v].astype(varType[v])

# The dataset is then rechunked to our desired shape. This does not actually do anything right now, but will when the data is written below. Before doing this, it is useful to remove the per-variable chunking metadata, because of inconsistencies which arise due to (I think) each MPI process output having a different chunking. This is explained in more detail in https://github.com/dcs4cop/xcube/issues/347
#

# In[17]:


print("re-chunking")
tic = time.time()
for v in dataIn.variables:
    if "chunks" in dataIn[v].encoding:
        del dataIn[v].encoding["chunks"]
dataIn = dataIn.chunk(chunksize)
print("   done in", time.time() - tic)

# The dataset `dataIn` is now ready to be written back out with dataIn.to_zarr(). Because this can take a while, it is nice to delay computation and then compute() the resulting object with a progress bar, so we know how long we have to get a cup of coffee or tea.
#

# In[19]:


delayedObj = dataIn.to_zarr(outputDir, compute=False)
with ProgressBar():
    results = delayedObj.compute()

# We can now load the zarr data set we have created, and see what is in it, compared to what was in the input dataset. Note that since we have not used "decode_times=False", the time coordinate appears as a datetime object.
#

# In[20]:


dataOriginal = xr.concat(
    [xr.open_zarr(f) for f in files],
    dim="trajectory",
    compat="no_conflicts",
    coords="minimal",
)
dataProcessed = xr.open_zarr(outputDir)
print("The original data\n", dataOriginal, "\n\nThe new dataSet\n", dataProcessed)
