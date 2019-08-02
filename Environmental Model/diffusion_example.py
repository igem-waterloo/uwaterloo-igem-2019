#Nonlinear diffusion model from documentation
#Here we use the Algebraic Newton Method to "discretize" the nonlinearity out
#MODEL: -div(q(u)grad(u))=f -> Nonlinear poisson equation

# Warning: from fenics import * will import both `sym` and
# `q` from FEniCS. We therefore import FEniCS first and then
# overwrite these objects.
from fenics import *
import numpy as np
def q(u):
    "Return nonlinear coefficient"
    return 1 + u**3
# Create mesh and define function space
mesh = UnitSquareMesh(10, 10)
V = FunctionSpace(mesh, 'P',1)

# Define boundary condition
import sympy as sym
x,y=sym.symbols('x[0],x[1]')
u_c=1-0.5*x**2+0.23*x*y**2
u_code=sym.printing.ccode(u_c) 
u_D = Expression(u_code, degree=3)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = Function(V)  # Note: not TrialFunction!
v = TestFunction(V)
F = q(u)*dot(grad(u), grad(v))*dx - u**3*v*dx

# Compute solution
solve(F == 0, u, bc)

# Plot solution
import matplotlib.pyplot as pl
plot(u)
pl.show()
pl.savefig('nonlin_poisson.png')

# Compute maximum error at vertices. This computation illustrates
# an alternative to using compute_vertex_values as in poisson.py.
u_e = interpolate(u_D, V)

error_max = np.abs(u_e.vector() - u.vector()).max()
print('error_max = ', error_max)

