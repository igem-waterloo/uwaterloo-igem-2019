#Nonlinear diffusion model from documentation
#Here we use the Algebraic Newton Method to "discretize" the nonlinearity out
#MODEL: -div(q(u)grad(u))=f -> Nonlinear poisson equation

# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
import dolfin as df

import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'P',1)

# Define boundary condition
import sympy as sy
x,y=sy.symbols('x[0],x[1]')
u_c=1-0.5*x**2+0.23*x*y**2
u_code=sy.printing.ccode(u_c) 
u_D = Expression(u_code, degree=3)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)
Dx=1
Dy=1
Dz=1
# Define variational problem
D = sym(as_tensor([[Dx, 0],
	           [0, Dy]])) 
# dispersion coefficient matrix
uA = Function(V)  # Note: not TrialFunction!
uB = interpolate(u_D,V)
v = TestFunction(V)
tF=1
dt = 0.1
t=0
F = dot(D*grad(uA), grad(v))*dx - uA**3*v*dx + v*(uA-uB)/dt*dx
import matplotlib.pyplot as pl
while t<=tF:
	t+=dt
	solve(F==0,uA,bc)
	
	vtkfile = File('nonlin_testing/solution_'+str(t)+'.pvd')
	vtkfile << uA
	uB.assign(uA)

	# Plot solution
	plot(uB)
	pl.show()
	pl.savefig('nonlin_poisson'+str(t)+'.png')

# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)

error_max = np.abs(u_e.vector() - uB.vector()).max()
print('error_max = ', error_max)

