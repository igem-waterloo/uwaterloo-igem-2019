#Model: div(D*grad(c))-w dc/dz = -dc/dt
from fenics import *
import dolfin as df
import numpy as np

dt = 0.005
t=0

# Create mesh and define function space
mesh = BoxMesh(Point(-4,-4,-2),Point(4,4,2),20,20,20)
V = FunctionSpace(mesh, 'P',1)

u_0 = Expression('3*exp(-a*(pow(x[1],2)+pow(x[0],2)+pow(x[2]-1.5,2)))',
                 degree=3, a=1)
Dx=3
Dy=3
Dz=3
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

