#Model: div(D*grad(c))-w dc/dz = -dc/dt
from fenics import *
import dolfin as df
import numpy as np
import matplotlib.pyplot as pl
import mshr as ms

dt = 0.05
t=0

class Nodule(SubDomain):
	def inside(self,x,on_boundary):
		r=[0,0,0]
		R = ((x[0]-r[0])**2+(x[1]-r[1])**2+(x[2]-r[2])**2)**0.5
		return True if R <= 1 else False

#Define subdomains (root nodules) - prototype class
subdomain1=Nodule() #define a new root nodule

# Create mesh and define function space
##mesh = BoxMesh(Point(-10,-10,-10),Point(10,10,2),30,30,30)
#we need to read mesh from xml file
mesh=Mesh('geometry.xml')
V = FunctionSpace(mesh, 'P',1)
dofmap = V.dofmap() #this basically lets us grab the raw data from the mesh about cells

u_0 = Expression('10*exp(-a*(pow(x[1],2)+pow(x[0],2)+pow(x[2]-1.5,2)))',
                 degree=3, a=0.1)
cf=MeshFunction('size_t',mesh,3) 
#define a function with values on the cells of the mesh (the 3 means '3d' blocks)

subdomain1.mark(cf,1) #mark the function as 1 inside the subdomain
heaviside = Function(V) #define a function in V - this will be the heaviside
for cell in cells(mesh): # set the characteristic functions
    if cf[cell] == 1:
        heaviside.vector()[dofmap.cell_dofs(cell.index())] = 1
	#project the characteristic function cf into our space V 
	# ie - this is the heaviside function	
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

tF=10
w=1
K=10*heaviside #degradation only happening inside the subdomain
F = dot(D*grad(uA), grad(v))*dx + v*(uA-uB)/dt*dx - v*w*grad(uA)[2]*dx + K*uA*v*dx
counter = 1
vtkfile = File('diffusion3d/solution.pvd')

while t<=tF:
	t+=dt
	solve(F==0,uA)
	uA.rename('uA','uA')
	vtkfile << uA, counter
	uB.assign(uA)
	counter+=1


