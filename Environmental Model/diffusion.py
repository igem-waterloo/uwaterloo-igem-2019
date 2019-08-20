# IMPORTANT: If you want to run any Fenics simulation in parallel to speed it up, Fenics plays really well with MPI
# Install mpich for Ubuntu (and the python bindings if they don't install automatically) and run the following command:
# mpirun -n #t python3 demo.py
# where #t is the number of threads you want and demo.py is your file.
# Model: div(D*grad(c)) - w dc/dz + M(c,t) = -dc/dt

from fenics import *
import dolfin as df #fenics backend servicing
import numpy as np 
import mshr as ms #mesh generation tool

dt = 0.02 #time increment
t = 0 #initial time

#definition of the nodule
class Nodule(SubDomain):
	def inside(self, x, on_boundary):
		r = [0, -3, 0]
		R = ((x[0]-r[0])**2+(x[1]-r[1])**2)
		r2 = [0, -5, 0]
		R2 = ((x[0]-r2[0])**2+(x[1]-r2[1])**2)
		return True if (R <= 0.5 or R2<=0.5) else False

subdomain1 = Nodule()  # define a new root nodule
rect = ms.Rectangle(Point(-4,-10.0),Point(4,0))
circ = ms.Circle(Point(0,-3),0.5) # top nodule
circ2 = ms.Circle(Point(0,-5),0.5) # bottom nodule
rect.set_subdomain(1,circ) #include the circles in the domain
rect.set_subdomain(2,circ2)
mesh = ms.generate_mesh(rect,40) #generating the mesh

V = FunctionSpace(mesh, 'CG', 1) #solution space
dofmap = V.dofmap()  # this basically lets us grab the raw data from the mesh about cells

cf = MeshFunction('size_t', mesh, 2)  # define a function with values on the cells of the mesh (the 3 means '3d' blocks)

subdomain1.mark(cf, 1)  # mark the function as 1 inside the subdomain

heaviside = Function(V)  # define a function in V - this will be the heaviside

for cell in cells(mesh):  # set the characteristic functions
	if cf[cell] == 1:
		ind=dofmap.cell_dofs(cell.index())
		heaviside.vector()[ind] = 1
        	# project the characteristic function cf into our space V
       		# ie - this is the heaviside function

# dispersion coefficients in cm^2/day
Dx = Constant(1.144137e-01)
Dy = Constant(1.144137e-01)
Dz = Constant(1.262505e-01)

D = sym(as_tensor([[Dx, 0],
                   [0, Dz]]))

# define the soil surface
def top_boundary(x,on_boundary):
	return (x[1]>-DOLFIN_EPS) and on_boundary

# initial condition
bound = Expression("(x[1]>-2)?5e-9:2e-9",degree=1)

# define boundary condition
bc = DirichletBC(V,Constant(10e-9),top_boundary)

# Functions
uA = TrialFunction(V) # function to be solved for
u = Function(V)  # for storing the solution
uB = interpolate(bound,V) # t-dt timestep solution
u_k = interpolate(bound,V) # iterator for nonlinear problem
v = TestFunction(V) # variational component


# Constants
km = Constant(5.8e-9) # mol/cm^3
tF = 100 # final time
w=Constant(0.1217) # advection speed in units cm/day
vmax = 5e-9*heaviside # degradation in mol/cm^3/day

A = inner(D*grad(uA), grad(v))*dt*dx + v*uA*dx - v*w*dt*grad(uA)[1]*dx # primary terms 
L = -vmax*(u_k/(km+u_k))*v*dt*dx + v*uB*dx # constant and nonlinear terms

counter = 0 #for writing the file
vtkfile = File('diffusion2d_working/solution.pvd')

tolerance = 1e-7 # error tolerance
maxit = 50 # maximum picard iterations before failure
while t <= tF:
	# Picard Iteration for NL problem
	ite = 0 # iteration variable
	eps = 1 # initialize error
	print('\nTime: '+str(t))
	while ite < maxit and eps > tolerance:
		solve(A == L, u,bc) # solve the PDE	
		diff = np.abs(u.vector() - u_k.vector())  
		eps = np.linalg.norm(diff, ord=np.Inf) # find pointwise-maximum error in solution
		print('Eps: '+str(eps))
		u_k.assign(u)   # update for next iteration
		ite+=1
	
	u.rename('u', 'u')
	if counter % 20 == 0:
        	vtkfile << u, counter
        	print('\nTimestep: '+str(counter)+'\n')
        	print('\nTime: '+str(t)+'\n')
	uB.assign(u) # increment timewise
	counter += 1
	t+=dt
