
# Model: div(D*grad(c))-w dc/dz = -dc/dt - standard heat equation with diffusion matrix
from fenics import *
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

dt = 0.005  # timestep for discretisation
t = 0  # starting point of discretisation

# Create mesh and define function space
mesh = BoxMesh(Point(-4, -4, -2), Point(4, 4, 2), 20, 20, 20)
V = FunctionSpace(mesh, 'P', 1)

u_0 = Expression('3*exp(-a*(pow(x[1],2)+pow(x[0],2)+pow(x[2]-1.5,2)))',
                 degree=3, a=1)


# Defining variational problem:

# Defining the dispersion matrix coefficients:
Dx = 3
Dy = 3
Dz = 3

# Defining the dispersion matrix:
D = sym(as_tensor([[Dx,  0,  0],
                   [0,  Dy,  0],
                   [0,   0, Dz]]))

uA = Function(V)  # note: not the trial/test function
uB = interpolate(u_0, V)
v = TestFunction(V)  # the trial/test function mapped to the function space
tF = 1
w = -0.1
F = dot(D*grad(uA), grad(v))*dx + v*(uA-uB)/dt*dx - v*w*grad(uA)[2]*dx

counter = 1
vtkfile = File('fenics_solutions/heat_solution.pvd')

while t <= tF:
    t += dt
    solve (F == 0, uA)
    uA.rename('uA', 'uA')
    vtkfile << uA, counter
    uB.assign(uA)
    counter += 1
# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)

error_max = np.abs(u_e.vector() - uB.vector()).max()
print('error_max = ', error_max)