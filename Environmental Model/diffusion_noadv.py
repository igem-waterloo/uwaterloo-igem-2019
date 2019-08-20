# IMPORTANT: If you want to run any Fenics simulation in parallel to speed it up, Fenics plays really well with MPI
# Install mpich for Ubuntu (and the python bindings if they don't install automatically) and run the following command:
# mpirun -n #t python3 demo.py
# where #t is the number of threads you want and demo.py is your file.
# Model: div(D*grad(c))-w dc/dz = -dc/dt

from fenics import *
import dolfin as df
import numpy as np
import mshr as ms

dt = 0.5
t = 0

class Nodule(SubDomain):
    def inside(self, x, on_boundary):
        r = [0, -3, 0]
        R = ((x[0]-r[0])**2+(x[1]-r[1])**2)
        return True if R <= 1 else False

# Define subdomains (root nodules) - prototype class:
subdomain1 = Nodule()  # define a new root nodule
rect = ms.Rectangle(Point(-7,-10),Point(7,0))
circ = ms.Circle(Point(0,-3),1)
rect.set_subdomain(1,circ)
mesh = ms.generate_mesh(rect,100)
#u_0 = Expression('(0.005/pow(25*2*3.141,0.5))*exp(-1*(0.5*pow(x[1]-2,2)+0.1*pow(x[0],2)))',
#                 degree=2)

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
Dx = Constant(0.0056*100)
Dy = Constant(0.0056*100)
Dz = Constant(0.0056*100)

# Dispersion coefficient matrix:
D = sym(as_tensor([[Dx, 0],
                   [0, Dy]]))

def top_boundary(x,on_boundary):
	return (x[1]>-DOLFIN_EPS) and on_boundary

bc = DirichletBC(V,Constant(0.050),top_boundary)

uA = TrialFunction(V)
u = Function(V)  # Note: not TrialFunction!
uB = interpolate(Constant(0.050),V)
v = TestFunction(V)

tF=1000
w=Constant(0.004*100)
K=0.0017*100*heaviside #degradation only happening inside the subdomain
A = inner(D*grad(uA), grad(v))*dt*dx + v*uA*dx + dt*K*uA*v*dx
L=v*uB*dx

counter = 0
vtkfile = File('diffusion2d_noadv_working/solution.pvd')

while t <= tF:
    t += dt
    solve(A == L, u,bc)
    u.rename('u', 'u')
    if counter % 10 == 0:
        vtkfile << u, counter
        print('\nTimestep: '+str(counter)+'\n')
        print('\nTime: '+str(t)+'\n')
    uB.assign(u)
    counter += 1
