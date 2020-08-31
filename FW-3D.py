import sympy
print(sympy.__file__)
from sympy import Abs
import numpy as np
import argparse


from examples.seismic import Model
from examples.seismic import TimeAxis
from examples.seismic import RickerSource
from examples.seismic import Receiver
from devito import Eq, solve
from devito import Operator
from examples.seismic import plot_shotrecord



parser = argparse.ArgumentParser()
parser.add_argument('--velocity_setup', action="store", dest="vsetup", type=str, default=None, required=True,
                    help=" expects a npz file with shape, spacing, origin and velocity model")
parser.add_argument('--src_loc', action="store", dest="src_loc", type=str, default=None, required=True)
parser.add_argument('--rec_loc_setup', action="store", dest="rec_loc_setup", type=str, default=None, required=True,
                    help=" expects a npz file with receiver locations n_point x 3 matrix")
parser.add_argument('--time', action="store", dest="time", type=float, default=1.0)


# Argument loading
res = parser.parse_args()
REC_LOC_SETUP = res.rec_loc_setup
rsetup = np.load(REC_LOC_SETUP)
REC_LOC = rsetup["rec_loc"]
SRC_LOC = res.src_loc
SRC_LOC_X, SRC_LOC_Y, SRC_LOC_Z = [float(a) for a in SRC_LOC.split(',')]
VSETUP = res.vsetup
setup = np.load(VSETUP)
shape = setup["shape"]
spacing = setup["spacing"]
origin = setup["origin"]
v = setup["velocity_model"]
TIME = res.time

model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbl=10, bcs="damp")

#### SOURCE SETTING ####
t0 = 0.  # Simulation starts a t=0
tn = np.ceil(TIME * 1000)  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=1, time_range=time_range)

src.coordinates.data[0, :] = np.array([SRC_LOC_X, SRC_LOC_Y, SRC_LOC_Z])
#src.show()


#### RECEIVER SETTING ####
rec = Receiver(name='rec', grid=model.grid, npoint=REC_LOC.shape[0], time_range=time_range)
rec.coordinates.data[:,:] = REC_LOC


# ACOUSTIC MODEL

from devito import TimeFunction
u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
pde = model.m * u.dt2 - u.laplace + model.damp * u.dt

stencil = Eq(u.forward, solve(pde, u.forward))
src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
rec_term = rec.interpolate(expr=u.forward)
op = Operator([stencil] + src_term + rec_term, subs=model.spacing_map)
op(time=time_range.num-1, dt=model.critical_dt)
plot_shotrecord(rec.data, model, t0, tn)
