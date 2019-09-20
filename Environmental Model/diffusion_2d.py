# IMPORTANT: If you want to run any Fenics simulation in parallel to speed it up, Fenics plays really well with MPI
# Install mpich for Ubuntu (and the python bindings if they don't install automatically) and run the following command:
# mpirun -n #t python3 demo.py
# where #t is the number of threads you want and demo.py is your file.

# Model: div(D*grad(c))-w dc/dz = -dc/dt
from fenics import *
import dolfin as df
import numpy as np
import mshr as ms

dt = 0.02
t = 0


class Nodule(SubDomain):
    def inside(self, x, on_boundary):
        r = [0, 0, 0]
        R = ((x[0]-r[0])**2+(x[1]-r[1])**2+(x[2]-r[2])**2)
        return True if R <= 1 else False


# Define subdomains (root nodules) - prototype class:
subdomain1 = Nodule()  # define a new root nodule
mesh = RectangleMesh(Point(-7, -10), Point(7, 10), 50, 50)

u_0 = Expression('(0.005/pow(25*2*3.141,0.5))*exp(-1*(0.5*pow(x[1]-2,2)+0.1*pow(x[0],2)))',
                 degree=2)

V = FunctionSpace(mesh, 'P', 1)
dofmap = V.dofmap()  # this basically lets us grab the raw data from the mesh about cells

cf = MeshFunction('size_t', mesh, 2)  # define a function with values on the cells of the mesh (the 3 means '3d' blocks)

subdomain1.mark(cf, 1)  # mark the function as 1 inside the subdomain

heaviside = Function(V)  # define a function in V - this will be the heaviside
print(len(heaviside.vector()))
for cell in cells(mesh):  # set the characteristic functions
	if cf[cell] == 1:
		ind=dofmap.cell_dofs(cell.index())
		print(ind)
		heaviside.vector()[ind] = 1
        	# project the characteristic function cf into our space V
       		# ie - this is the heaviside function

# Defining the variational problem:
Dx = Constant(9.2e-7*100)
Dy = Constant(9.2e-7*100)
Dz = Constant(2.1e-6*100)

# Dispersion coefficient matrix:
D = sym(as_tensor([[Dx, 0],
                   [0, Dz]]))

uA = TrialFunction(V)
u = Function(V)  # Note: not TrialFunction!
uB = interpolate(u_0,V)
v = TestFunction(V)

tF = 50
w = Constant(0.004*100)
K = 0.0017*100*heaviside #degradation only happening inside the subdomain
A = inner(D*grad(uA), grad(v))*dt*dx + v*uA*dx - v*w*dt*grad(uA)[2]*dx + dt*K*uA*v*dx
L = v*uB*dx

counter = 0
vtkfile = File('diffusion3d_working/solution.pvd')

while t <= tF:
    t += dt
    solve(A == L, u)
    u.rename('u', 'u')
    if counter % 10 == 0:
        vtkfile << u, counter
        print('\nTimestep: '+str(counter)+'\n')
    uB.assign(u)
    counter += 1
