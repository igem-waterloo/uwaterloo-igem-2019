# IMPORTANT: If you want to run any Fenics simulation in parallel to speed it up, Fenics plays really well with MPI
# Install mpich for Ubuntu (and the python bindings if they don't install automatically) and run the following command:
# mpirun -n #t python3 demo.py
# where #t is the number of threads you want and demo.py is your file.
# Model: div(D*grad(c)) - w dc/dz + M(c,t) = -dc/dt

from fenics import *
import dolfin as df # fenics backend servicing
import numpy as np 
import mshr as ms # mesh generation tool

dt = 0.1 # time increment
t = 0 # initial time
tF=100

num_steps = int(tF/dt)

# definition of the nodule
class Nodule(SubDomain):
	def inside(self, x, on_boundary):
		r = [0, -3, 0]
		R = ((x[0]-r[0])**2+(x[1]-r[1])**2)**0.5
		r2 = [0, -5, 0]
		R2 = ((x[0]-r2[0])**2+(x[1]-r2[1])**2)**0.5
		return True if (R <= 0.5 or R2<=0.5) else False
class NoduleBoundary(SubDomain):
	def inside(self,x,on_boundary):
		r=[0,-3,0]
		r2=[0,-5,0]
		R = ((x[0]-r[0])**2+(x[1]-r[1])**2)**0.5
		R2 = ((x[0]-r2[0])**2+(x[1]-r2[1])**2)**0.5
		return True if near(R,0.5) or near(R2,0.5) else False

nodule = Nodule()  # define a new root nodule
nodbound = NoduleBoundary()

rect = ms.Rectangle(Point(-4,-10.0),Point(4,0))
circ = ms.Circle(Point(0,-3),0.5) # top nodule
circ2 = ms.Circle(Point(0,-5),0.5) # bottom nodule
rect.set_subdomain(1,circ) #include the circles in the domain
rect.set_subdomain(2,circ2)

#mesh=Mesh('2dmesh.xml')
mesh = ms.generate_mesh(rect,50) #generating the mesh

boundary_marker = MeshFunction("size_t",mesh,0)
nodbound.mark(boundary_marker,1)

P1=FiniteElement('P', triangle, 1)
element=MixedElement([P1,P1,P1])

VelocitySpace = VectorFunctionSpace(mesh, 'P', 2) # velocity field space
PressureSpace = FunctionSpace(mesh, 'P', 1) # pressure space
V = FunctionSpace(mesh, element) # concentration space
dofmap = V.dofmap()  # this basically lets us grab the raw data from the mesh about cells
v1,v2,v3 = TestFunctions(V)

wmax=Constant(0.1217) # advection speed in units cm/day

inflow_profile = Expression(('0','-wmax*4*(4+x[0])*(4-x[0])/64'),degree=2,wmax=wmax)
inflow = 'near(x[1],0)'
outflow = 'near(x[1],-10)'
walls = 'near(x[0],-4)||near(x[0],4)'
nodb = 'near(pow(x[0],2)+pow(x[1]+3,2),pow(0.5,2),0.05)||near(pow(x[0],2)+pow(x[1]+5,2),pow(0.5,2),0.05)'
inodb = 'pow(x[0],2)+pow(x[1]+3,2)<=0.5||pow(x[0],2)+pow(x[1]+5,2)<=0.5'

bcu_inflow = DirichletBC(VelocitySpace,inflow_profile,inflow)
bcp_outflow = DirichletBC(PressureSpace,Constant(0),outflow)
bcu_walls = DirichletBC(VelocitySpace,Constant((0,0)),walls)
bcu_nodule = DirichletBC(VelocitySpace,Constant((0,0)),nodb)
bcp_inod = DirichletBC(PressureSpace,Constant(0),inodb)
bcu_inod = DirichletBC(VelocitySpace,Constant((0,0)),inodb)

bcu = [bcu_inflow,bcu_walls,bcu_nodule,bcu_inod]
bcp = [bcp_outflow]
# this code adapted from fenics docs
mu = 0.001 # dynamic viscosity
rho = 1 # density
# Define trial and test functions
u_vel = TrialFunction(VelocitySpace)
v_vel = TestFunction(VelocitySpace)
p_pre = TrialFunction(PressureSpace)
q_pre = TestFunction(PressureSpace)
# Define functions for solutions at previous and current time steps
vel_u_n = Function(VelocitySpace)
vel_u_ = Function(VelocitySpace)
pre_p_n = Function(PressureSpace)
pre_p_ = Function(PressureSpace)
# Define expressions used in variational forms
U_vel = 0.5*(vel_u_n + u_vel)
normal_vec = FacetNormal(mesh)
f_src = Constant((0, 0))
k_const = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

nodule_marker = MeshFunction("size_t",mesh,2)
nodule.mark(nodule_marker,1)
ds = Measure('ds',domain=mesh,subdomain_data=boundary_marker)
#dy = Measure('dx',domain=mesh,subdomain_data=nodule_marker)

# Define symmetric gradient
def epsilon(u):
	return sym(nabla_grad(u))
# Define stress tensor
def sigma(u, p):
	return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u_vel - vel_u_n) / k_const, v_vel)*dx \
	+ rho*dot(dot(vel_u_n, nabla_grad(vel_u_n)), v_vel)*dx \
	+ inner(sigma(U_vel, pre_p_n), epsilon(v_vel))*dx \
	+ dot(pre_p_n*normal_vec, v_vel)*ds(0) - dot(mu*nabla_grad(U_vel)*normal_vec, v_vel)*ds(0) \
	- dot(f_src, v_vel)*dx
av1 = lhs(F1)
Lv1 = rhs(F1)
# Define variational problem for step 2
av2 = dot(nabla_grad(p_pre), nabla_grad(q_pre))*dx
Lv2 = dot(nabla_grad(pre_p_n), nabla_grad(q_pre))*dx - (1/dt)*div(vel_u_)*q_pre*dx
# Define variational problem for step 3
av3 = dot(u_vel, v_vel)*dx
Lv3 = dot(vel_u_, v_vel)*dx - dt*dot(nabla_grad(pre_p_ - pre_p_n), v_vel)*dx
# Assemble matrices
Av1 = assemble(av1)
Av2 = assemble(av2)
Av3 = assemble(av3)

[bc.apply(Av1) for bc in bcu]
[bc.apply(Av2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile('diffusion_velocityfield/velocity.xdmf')
xdmffile_p = XDMFFile('diffusion_velocityfield/pressure.xdmf')
# Create time series (for use in reaction)
timeseries_u = TimeSeries('diffusion_velocityfield/velocity_series')
timeseries_p = TimeSeries('diffusion_velocityfield/pressure_series')

for n in range(num_steps):
	# Update current time
	t += dt
	# Step 1: Tentative velocity step
	bv1 = assemble(Lv1)
	[bc.apply(bv1) for bc in bcu]
	solve(Av1, vel_u_.vector(), bv1, 'bicgstab', 'hypre_amg')
	# Step 2: Pressure correction step
	bv2 = assemble(Lv2)
	[bc.apply(bv2) for bc in bcp]
	solve(Av2, pre_p_.vector(), bv2, 'bicgstab', 'hypre_amg')
	# Step 3: Velocity correction step
	bv3 = assemble(Lv3)
	solve(Av3, vel_u_.vector(), bv3, 'cg', 'sor')
	
	# Save solution to file (XDMF/HDF5)
	xdmffile_u.write(vel_u_, t)
	xdmffile_p.write(pre_p_, t)
	# Save nodal values to file
	timeseries_u.store(vel_u_.vector(), t)
	timeseries_p.store(pre_p_.vector(), t)
	# Update previous solution
	vel_u_n.assign(vel_u_)
	pre_p_n.assign(pre_p_)
	
	print('vel max', vel_u_.vector().get_local().max())

t=0 # reset time to zero

cf = MeshFunction('size_t', mesh, 2)  # define a function with values on the cells of the mesh (the 3 means '3d' blocks)

nodule.mark(cf,1)  # mark the function as 1 inside the subdomain
dx = Measure('dx')[cf]

# dispersion coefficients in cm^2/day
Dx = Constant(1.144137e-01)
Dy = Constant(1.144137e-01)
Dz = Constant(1.262505e-01)

# dispersion in roots/apoplastic system
Dr = Constant(0.137) # cm^2/day

D = sym(as_tensor([[Dx, 0],
                   [0, Dz]]))

# define the soil surface
def top_boundary(x,on_boundary):
	return (x[1]>-DOLFIN_EPS) and on_boundary

# define boundary condition
bexp = Expression(("10e-9*exp(-t/2)","0","0"),degree=2,t=t)
bc = DirichletBC(V,bexp,top_boundary)

# define initial condition
class initcond(UserExpression):
	def eval(self, value, x):
		value[1] = 0.0
		value[2] = 0.0
		if x[1] > -2:
			value[0]=10e-9
		else:
			value[0]=1e-10
	def value_shape(self):
		return (3,)
f = initcond()

# Functions
uA = TrialFunction(V) # function to be solved for
u = Function(V)  # for storing the solution
uB = interpolate(f,V) # t-dt timestep solution
_u_k = interpolate(f,V) # iterator for nonlinear problem
u_k,u_k2,u_k3 = split(_u_k)
uA1,uA2,uA3=split(uA)
uB1,uB2,uB3=split(uB)
w = Function(VelocitySpace)

# Constants
km = Constant(5.8e-9) # mol/cm^3
vmax = 5e-9 # degradation in mol/cm^3/day
k_deg = 2.177e-9 # 1/day
ks = 1.4e-6 # 1/day

A1 = inner(D*grad(uA1),grad(v1))*dt*dx(0) + inner(Dr*grad(uA1),grad(v1))*dt*dx(1)\
	+ (uA1*ks*v1 + v1*uA1 + v1*dt*inner(w,grad(uA1)))*dx('everywhere') # primary terms 
L1 = -vmax*(u_k/(km+u_k))*v1*dt*dx(1) + v1*uB1*dx('everywhere') # constant and nonlinear terms

A2 = inner(D*grad(uA2),grad(v2))*dt*dx(0)+inner(Dr*grad(uA2), grad(v2))*dt*dx(1) \
	+ (-uA2*ks*v2 + v2*uA2 + v2*dt*inner(w,grad(uA2)))*dx('everywhere') + uA2*k_deg*dt*v2*dx(1)
L2 = vmax*(u_k/(km+u_k))*v2*dt*dx(1) + v2*uB2*dx('everywhere')

A3 = inner(D*grad(uA3),grad(v3))*dt*dx(0)+inner(Dr*grad(uA3), grad(v3))*dt*dx(1)\
	+ (v3*uA3 + v3*dt*inner(w,grad(uA3)))*dx('everywhere') - uA2*k_deg*dt*v3*dx(1)
L3 = v3*uB3*dx('everywhere')

A=A1+A2+A3
L=L1+L2+L3

counter = 0 #for writing the file

vtkfile1 = File('diffusion2d_tracking/solution1.pvd')
vtkfile2 = File('diffusion2d_tracking/solution2.pvd')
vtkfile3 = File('diffusion2d_tracking/solution3.pvd')

tolerance = 1e-7 # error tolerance
maxit = 50 # maximum picard iterations before failure


print(num_steps)
for i in range(1,num_steps):
	bexp.t=t
	# Picard Iteration for NL problem
	ite = 0 # iteration variable
	eps = 1 # initialize error
	print('\nTime: '+str(t))
	timeseries_u.retrieve(w.vector(), t)

	while ite < maxit and eps > tolerance:
		solve(A == L, u,bc) # solve the PDE	
		diff = np.abs(u.vector() - _u_k.vector())  
		eps = np.linalg.norm(diff, ord=np.Inf) # find pointwise-maximum error in solution
		print('Eps: '+str(eps))
		_u_k.assign(u)   # update for next iteration
		ite+=1
	
	u.rename('u', 'u')
	u1,u2,u3 = u.split()
	
	vtkfile1 << (u1, i)
	vtkfile2 << (u2, i)
	vtkfile3 << (u3, i)
	
	uB.assign(u) # increment timewise
	print(t)
	t+=dt
