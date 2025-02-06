#!/usr/bin/env python

# # JIT-vs-Scipy Particles

# This tutorial is meant to highlight the potentially very big difference between the computational time required to run Parcels in **JIT** (Just-In-Time compilation) versus in **Scipy** mode. It also discusses how to more efficiently sample in Scipy mode.
#

# ## Short summary: JIT is faster than scipy
#
# In the code snippet below, we use `AdvectionRK4` to advect 100 particles in the peninsula `FieldSet`. We first do it in JIT mode (by setting `ptype=JITParticle` in the declaration of `pset`) and then we also do it in Scipy mode (by setting `ptype=ScipyParticle` in the declaration of `pset`).
#
# In both cases, we advect the particles for 1 hour, with a timestep of 30 seconds.
#
# To measure the computational time, we use the `timer` module.
#

# In[1]:


from datetime import timedelta

import parcels

parcels.timer.root = parcels.timer.Timer("root")

parcels.timer.fieldset = parcels.timer.Timer(
    "fieldset creation", parent=parcels.timer.root
)

example_dataset_folder = parcels.download_example_dataset("Peninsula_data")
fieldset = parcels.FieldSet.from_parcels(
    f"{example_dataset_folder}/peninsula", allow_time_extrapolation=True
)
parcels.timer.fieldset.stop()

ptype = {"scipy": parcels.ScipyParticle, "jit": parcels.JITParticle}
ptimer = {
    "scipy": parcels.timer.Timer("scipy", parent=parcels.timer.root, start=False),
    "jit": parcels.timer.Timer("jit", parent=parcels.timer.root, start=False),
}

for p in ["scipy", "jit"]:
    pset = parcels.ParticleSet.from_line(
        fieldset=fieldset,
        pclass=ptype[p],
        size=100,
        start=(3e3, 3e3),
        finish=(3e3, 45e3),
    )

    ptimer[p].start()
    pset.execute(
        parcels.AdvectionRK4, runtime=timedelta(hours=1), dt=timedelta(seconds=30)
    )
    ptimer[p].stop()

parcels.timer.root.stop()
parcels.timer.root.print_tree()

# As you can see above, even in this very small example **Scipy mode took more than 2 times as long** (2.1 seconds versus 1.0 seconds) as the JIT mode. For larger examples, this can grow to hundreds of times slower.
#
# This is just an illustrative example, depending on the number of calls to `AdvectionRK4`, the size of the `FieldSet`, the size of the `pset`, the ratio between `dt` and `outputdt` in the `.execute` etc, the difference between JIT and Scipy can vary significantly. However, JIT will almost always be faster!
#
# So why does Parcels support both JIT and Scipy mode then? Because Scipy is easier to debug when writing custom kernels, so can provide faster development of new features.
#

# _As an aside, you may wonder why we use the `time.time` module, and not the `timeit` module, to time the runs above. That's because it affects the AST of the kernels, causing errors in JIT mode._
#

# ## Further digging into Scipy mode: adding `particle` keyword to `Field`-sampling
#
# Sometimes, you'd want to run Parcels in Scipy mode anyways. In that case, there are ways to make Parcels a bit faster.
#
# As background, one of the most computationally expensive operations in Parcels is the [Field Sampling](https://docs.oceanparcels.org/en/latest/examples/tutorial_sampling.html). In the default sampling in Scipy mode, we don't keep track of _where_ in the grid a particle is; which means that for every sampling call, we need to again search for which grid cell a particle is in.
#
# Let's see how this works in the simple Peninsula FieldSet used above. We use a simple Euler-Forward Advection now to make the point. In particular, we use two types of Advection Kernels
#

# In[2]:


def AdvectionEE_depth_lat_lon_time(particle, fieldset, time):
    (u1, v1) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
    particle.lon += u1 * particle.dt
    particle.lat += v1 * particle.dt


def AdvectionEE_depth_lat_lon_time_particle(particle, fieldset, time):
    (u1, v1) = fieldset.UV[
        time,
        particle.depth,
        particle.lat,
        particle.lon,
        particle,  # note the extra particle argument here
    ]
    particle.lon += u1 * particle.dt
    particle.lat += v1 * particle.dt


kernels = {
    "dllt": AdvectionEE_depth_lat_lon_time,
    "dllt_p": AdvectionEE_depth_lat_lon_time_particle,
}

# In[3]:


parcels.timer.root = parcels.timer.Timer("root")
ptimer = {
    "dllt": parcels.timer.Timer("dllt", parent=parcels.timer.root, start=False),
    "dllt_p": parcels.timer.Timer("dllt_p", parent=parcels.timer.root, start=False),
}

for k in ["dllt", "dllt_p"]:
    pset = parcels.ParticleSet.from_line(
        fieldset=fieldset,
        pclass=parcels.ScipyParticle,
        size=100,
        start=(3e3, 3e3),
        finish=(3e3, 45e3),
    )

    ptimer[k].start()
    pset.execute(kernels[k], runtime=timedelta(hours=1), dt=timedelta(seconds=30))
    ptimer[k].stop()

parcels.timer.root.stop()
parcels.timer.root.print_tree()

# You will see that the two kernels don't really differ in speed. That is because the Peninsula FieldSet is a simple _Rectilinear_ Grid, where indexing a particle location to the grid is very fast.
#
# However, the difference is much more pronounced if we use a _Curvilinear_ Grid like in the [NEMO dataset](https://docs.oceanparcels.org/en/latest/examples/tutorial_nemo_curvilinear.html).
#

# In[4]:


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

# In[5]:


parcels.timer.root = parcels.timer.Timer("root")
ptimer = {
    "dllt": parcels.timer.Timer("dllt", parent=parcels.timer.root, start=False),
    "dllt_p": parcels.timer.Timer("dllt_p", parent=parcels.timer.root, start=False),
}

for k in ["dllt", "dllt_p"]:
    pset = parcels.ParticleSet.from_line(
        fieldset=fieldset,
        pclass=parcels.ScipyParticle,
        size=10,
        start=(45, 40),
        finish=(60, 40),
    )

    ptimer[k].start()
    pset.execute(kernels[k], runtime=timedelta(days=10), dt=timedelta(hours=6))
    ptimer[k].stop()

parcels.timer.root.stop()
parcels.timer.root.print_tree()

# Now, the difference is massive, with the `AdvectionEE_depth_lat_lon_time_particle` kernel more than 20 times faster than the kernel without the `particle` argument at the end of the Field sampling operation.
#
# So, if you want to run in Scipy mode, make sure to add `particle` at the end of your Field sampling!
#
