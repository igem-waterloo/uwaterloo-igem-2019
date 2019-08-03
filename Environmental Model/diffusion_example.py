#Nonlinear diffusion model from documentation
#Here we use the Algebraic Newton Method to "discretize" the nonlinearity out
#MODEL: -div(q(u)grad(u))=f -> Nonlinear poisson equation

# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
import dolfin as df

import numpy as np
dt = 0.001
t=0
# Create mesh and define function space
mesh = BoxMesh(Point(-2,-2,-2),Point(2,2,2),20,20,20)
V = FunctionSpace(mesh, 'P',1)

u_0 = Expression('exp(-10*a*pow(x[2]-2,2))',
                 degree=3, a=5)
def boundary(x, on_boundary):
    return on_boundary and abs(x[0]+2)>1e-14
def boundaryLeft(x,on_boundary):
    return on_boundary and abs(x[0]+2)<=1e-14
def boundaryBottom(x,on_boundary):
    return on_boundary and abs(x[1]-2)<=1e-14

Dx=1
Dy=1
Dz=1
# Define variational problem
D = sym(as_tensor([[Dx,  0,  0],
	           [0,  Dy,  0],
	           [0,   0, Dz]])) 
# dispersion coefficient matrix
uA = Function(V)  # Note: not TrialFunction!
uB = interpolate(u_0,V)
v = TestFunction(V)
tF=1
w=-0.1
F = dot(D*grad(uA), grad(v))*dx + v*(uA-uB)/dt*dx - v*w*grad(uA)[2]*dx
import matplotlib.pyplot as pl
counter = 1
vtkfile = File('diffusion3d/solution.pvd')
while t<=tF:
	t+=dt
	solve(F==0,uA)
	uA.rename('uA','uA')
	vtkfile << uA, counter
	uB.assign(uA)
	counter+=1
# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)

error_max = np.abs(u_e.vector() - uB.vector()).max()
print('error_max = ', error_max)

