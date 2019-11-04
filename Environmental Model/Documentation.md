# Environmental Model Docs

## Classes

- Nodule (inherits: Dolfin.SubDomain)
  - Subdomain object describing a union of disks in the 2d domain
- NoduleBoundary (inherits: Dolfin.SubDomain)
  - Subdomain object describing a union of circles in the 2d domain (codimension 1)
- initcond (inherits: UserExpression)
  - descibes initial conditions

## Vector Space Objects

- VelocitySpace (type: FeNiCS.VectorFunctionSpace)
  - Vector space object containing vector-valued functions
  - Velocity functions are elements of this space
  - Lagrange elements (order 2)
- PressureSpace (type: FeNiCS.FunctionSpace)
  - Vector space object containing real-valued functions
  - Pressure functions are elements of this space
  - Lagrange elements (order 1)
- V (type: FeNiCS.FunctionSpace)
  - Main vector space for problem. Contains real-valued functions
  - Describes concentration functions
  - Mixed elements (three Order 1 Lagrange elements, each describing the concentration of one molecule)

## Dirichlet Boundary Conditions
### Velocity Space
- inflow 
  - pressure boundary condition at top of column
- outflow
  - pressure boundary condition at bottom of column
- walls
  - velocity boundary condition on walls -> assumes no horizontal throughput
- nodb, inodb
  - pressure and velocity conditions inside nodules
  - assume zero pressure and velocity inside the nodule due to obstructing cell walls

```
bcu_inflow = DirichletBC(VelocitySpace,inflow_profile,inflow)
bcp_outflow = DirichletBC(PressureSpace,Constant(0),outflow)
bcu_walls = DirichletBC(VelocitySpace,Constant((0,0)),walls)
bcu_nodule = DirichletBC(VelocitySpace,Constant((0,0)),nodb)
bcp_inod = DirichletBC(PressureSpace,Constant(0),inodb)
bcu_inod = DirichletBC(VelocitySpace,Constant((0,0)),inodb)
```

## Navier-Stokes Solver
- Test/Trial Functions
  - u_vel, p_pre
    - trial functions for iteration method
  - v_vel, q_pre
    - test functions for use in the weak-form formulation of the PDE
  - vel_u_n, vel_u_
    - functions describing intermediate solutions during iteration
  - pre_p_n, pre_p_
    - functions describing intermediate solutions during iteration
  - U_n
    - midpoint function between intermediates required for Chorins method
- Constants
  - rho = density of water
  - mu = dynamic viscosity of water
  - f_src = source function
  - k_const = time-interval length
- Mesh Information
  - normal_vec 
    - normal vector to each face element of the mesh
  - nodule_marker
    - array marking which face elements are contained in a nodule
  - ds
    - array containing differentials for the edge elements on the mesh
- Functions
  - epsilon(u)
    - computes the symmetrized Jacobian for the velocity vector
  - sigma(u,p)
    - computes the stress energy tensor
## File output:

```
# Create XDMF files for visualization output
xdmffile_u = XDMFFile('diffusion_velocityfield/velocity.xdmf')
xdmffile_p = XDMFFile('diffusion_velocityfield/pressure.xdmf')
# Create time series (for use in reaction)
timeseries_u = TimeSeries('diffusion_velocityfield/velocity_series')
timeseries_p = TimeSeries('diffusion_velocityfield/pressure_series')
```

## Diffusion-Advection-Reaction solver
- Constants
  - D (type: Dolfin.tensor)
    - Contains diffusion coefficients for the soil
  - Dr (type: Dolfin.constant)
    - Diffusion coefficient inside the apoplastic pathway
  - tolerance = 1e-7
    - error tolerance
  - maxit = 50
    - maximum picard iterations before failure
  - km = Constant(5.8e-6) # mol
    - LibA approximate Km
  - vmax = 4e-7 # degradation in mol/day
    - LibA approximate Vmax
  - k_deg = 9.65 # 1/day
    - NAT1 effective rate constant
  - ks = 1.4e-6 # 1/day
    - Background degradation rate constant
- Mesh Information
  - cf
    - marker function, marks nodule region
  - dx
    - measure array with respect to marker function
- Boundary Conditions
  - top_boundary
    - function checking if an element exists on the soil surface
- Solver
  - uA
    - trial function for concentration
  - u
    - solution function
  
File Locations:
```
vtkfile1 = File('diffusion2d_tracking/solution1.pvd')
vtkfile2 = File('diffusion2d_tracking/solution2.pvd')
vtkfile3 = File('diffusion2d_tracking/solution3.pvd')
```

