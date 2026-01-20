#file for functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import dolfin as dlf


from dolfin import (
    Constant, Function, FunctionSpace, VectorFunctionSpace, 
    TestFunctions, TrialFunctions, Expression, interpolate, assign, exp
)
import ufl
from ufl import *  # Import ALL UFL operators explicitly
from ufl import dot, div, grad, nabla_grad, sqrt, inner, derivative, Measure
from ufl import exp

def mesh_set_up(Lx, Ly, Nx, Ny, showplot, U_inflow):
    mesh_dolfin = dlf.RectangleMesh(dlf.Point(0.0, 0.0),
                                    dlf.Point(Lx, Ly),
                                    Nx, Ny)

    V_elem = dlf.VectorElement("CG", mesh_dolfin.ufl_cell(), 2)
    Q_elem = dlf.FiniteElement("CG", mesh_dolfin.ufl_cell(), 1)
    mixed_elem = dlf.MixedElement([V_elem, Q_elem])
    W = dlf.FunctionSpace(mesh_dolfin, mixed_elem)

    # Unknown and test functions
    w = dlf.Function(W, name="State")
    u, eta = dlf.split(w)
    v, q   = dlf.TestFunctions(W)   
    print("Mesh and mixed space initialised.")

    V_sub = W.sub(0).collapse()
    u_init = interpolate(Constant((U_inflow, 0.0)), V_sub)
    assign(w.sub(0), u_init)


    if showplot:
        plt.figure(figsize=(6,5))
        dlf.plot(mesh_dolfin)
        plt.title("Computational mesh")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.show()

    return mesh_dolfin, W, w, u, eta, v, q

def setup_boundary_markers_and_bcs(mesh, W, Lx, Ly, U_inflow, noslip):

    # --- Define boundary subdomains ---
    class InletBoundary(dlf.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dlf.near(x[0], 0.0)

    class OutflowBoundary(dlf.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and dlf.near(x[0], Lx)

    class WallBoundary(dlf.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (dlf.near(x[1], 0.0) or dlf.near(x[1], Ly))

    # --- Create and mark boundary facets ---
    boundary_markers = dlf.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    inlet  = InletBoundary();  inlet.mark(boundary_markers, 1)
    outlet = OutflowBoundary(); outlet.mark(boundary_markers, 2)
    walls  = WallBoundary();   walls.mark(boundary_markers, 3)

    inflow_expr = dlf.Constant((U_inflow, 0.0))
    bc_inflow = dlf.DirichletBC(W.sub(0), inflow_expr, boundary_markers, 1)
    print(f"Boundary markers created and BCs applied:")
    print(f"   - Inlet  (ID=1): inflow velocity = {U_inflow:.2f} m/s")
    print( "   - Outlet (ID=2): open boundary (no Dirichlet BC)")

    #if noslip:
    #    noslip  = dlf.Constant((0.0, 0.0))
    #    bc_wall = dlf.DirichletBC(W.sub(0), noslip, boundary_markers, 3)
    #    bcs = [bc_inflow, bc_wall]
    #    print( "   - Walls  (ID=3): no‑slip (Dirichlet BC)\n")

    #else:
    bcs = [bc_inflow]
    print( "   - Walls  (ID=3): free‑slip (no Dirichlet BC)\n")
    

    return boundary_markers, bcs

def place_turbines_random(mesh, Lx, Ly, n_turbines, min_spacing, D, distance_from_inlet,
distance_to_side, distance_to_outlet, seed=None, margin=None, max_attempts=100):

    # Enforce placement rectangle
    xmin = distance_from_inlet 
    xmax = Lx - distance_to_outlet 
    ymin = distance_to_side 
    ymax = Ly - distance_to_side 

    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Domain too small for requested placement bounds and margin")

    positions = []
    attempts = 0

    while len(positions) < n_turbines and attempts < max_attempts:
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        new_pos = np.array([x, y])

        if all(np.linalg.norm(new_pos - np.array(pos)) >= min_spacing for pos in positions):
            positions.append(new_pos)

        attempts += 1
    print(f"Managed to place {len(positions)} turbines within {attempts} attempts.")

    if len(positions) < n_turbines:
        raise RuntimeError("Failed to place all turbines with the given constraints.")
    else:
        print("Turbines placed successfully.")

    return np.array(positions)

def show_turbine_positions_plot(showplot, initial_positions, mesh, D, Lx, Ly):

    if showplot:
        plt.figure()
        plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells(),  alpha=0.1)
        for pos in initial_positions:
            circle = plt.Circle((pos[0], pos[1]), D/2.0, color='r', alpha=0.5)
            plt.gca().add_artist(circle)
        plt.xlim(0, Lx)
        plt.ylim(0, Ly)
        plt.show()
    else: 
        print("Turbine figure display not requested.")

    return 


def show_turbine_positions(initial_positions, showcoordinates):
    
    if showcoordinates:
        print("The initial turbine positions are:")
        for i, pos in enumerate(initial_positions):
            print(f" Turbine {i+1}: x = {pos[0]:.2f} m, y = {pos[1]:.2f} m")
    else:
        print("Turbine coordinates display not requested.")
    return



def solve_tidal_flow_velocities(turbine_positions, w, W, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma, u, eta, v, q):
    
    n = dlf.FacetNormal(mesh)
    H = depth + eta            # total water depth
    f_u = dlf.Constant((0.0, 0.0))   # no internal body forcing

    # --- Turbine-induced momentum sink coefficient field ------------------
    x, y = dlf.SpatialCoordinate(mesh)
    f_u = Constant((0, 0))

    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += (
            0.5 * C_T * A_T / (2.0 * np.pi * sigma**2)
            * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))
        )

    # --- Nonlinear residual form F ----------------------------------------
    F = (
        dlf.inner(nu * grad(u), grad(v)) * dlf.dx                            # viscosity
        + dlf.inner(dot(u, nabla_grad(u)), v) * dlf.dx                      # advection
        - g * dlf.div(H * v) * eta * dlf.dx                                  # free-surface coupling
        + (cb / H) * dlf.inner(u * sqrt(dot(u, u)), v) * dlf.dx           # bottom friction
    )

    # Turbine momentum sink using spatially varying field
#    F += (Ct_field / H) * dlf.inner(u * sqrt(dot(u, u)), v) * dlf.dx

    # Continuity and body force term
    F += H * dlf.div(u) * q * dlf.dx - dlf.inner(f_u, v) * dlf.dx

    # --- Solve nonlinear problem with Newton's method ---------------------
    dlf.solve(
        F == 0,
        w,
        bcs,
        solver_parameters={
            "newton_solver": {
                "linear_solver": "mumps",
                "absolute_tolerance": 1e-8,
                "relative_tolerance": 1e-7,
                "maximum_iterations": 30,
                "relaxation_parameter": 1.0,
            }
        },
    )

    velocity = w.sub(0, deepcopy=True)
    #turbine_powers, _ = compute_turbine_power(
        #velocity, turbine_positions, rho, C_T, A_T
    #)
    #total_power = float(np.sum(turbine_powers))
    #print(f"The total power is {total_power/1e3:.1f} kW")

    return velocity


import dolfin as dlf
import numpy as np
from ufl import grad, div, dot, nabla_grad, sqrt, exp

def solve_tidal_flow_velocities2(
    turbine_positions,
    w, W, mesh, bcs,
    depth, nu, cb, g, C_T, A_T, sigma
):

    u, eta = dlf.split(w)
    v, q   = dlf.TestFunctions(W)
    
    n = dlf.FacetNormal(mesh)
    H = depth + eta                     # total water depth
    eps_H = 1e-6                        # small floor to avoid division by 0
    Hsafe = H + eps_H
    f_u = dlf.Constant((0.0, 0.0))      # no internal body forcing

    # --- Turbine-induced momentum sink coefficient field ---
    x, y = dlf.SpatialCoordinate(mesh)

    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += (
            0.5 * C_T * A_T / (2.0 * np.pi * sigma**2)
            * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))
        )

    # --- Nonlinear residual form F ---
    F = (dlf.inner(nu * dlf.grad(u), dlf.grad(v)) * dlf.dx
    + dlf.inner(dot(u, nabla_grad(u)), v) * dlf.dx
    - g * dlf.div(H * v) * eta * dlf.dx
    + (cb / H) * dlf.inner(u * sqrt(dot(u, u)), v) * dlf.dx
    + H * dlf.div(u) * q * dlf.dx - dlf.inner(f_u, v) * dlf.dx)


    # Turbine momentum sink using spatially varying field
    F += (Ct_field / Hsafe) * dlf.inner(u * sqrt(dot(u, u)), v) * dlf.dx

    # Continuity and body force term
    F += H * div(u) * q * dlf.dx - dlf.inner(f_u, v) * dlf.dx
        # Jacobian of F with respect to w
   # J_F = dlf.derivative(F, w)

    #problem = dlf.NonlinearVariationalProblem(F, w, bcs, J_F)
    #solver  = dlf.NonlinearVariationalSolver(problem)

    dlf.solve(
        F == 0,
        w,
        bcs,
        solver_parameters={
            "newton_solver": {
                "linear_solver": "mumps",
                "absolute_tolerance": 1e-8,
                "relative_tolerance": 1e-7,
                "maximum_iterations": 30,
                "relaxation_parameter": 1.0,
            }
        },
    )
    # --- Solve nonlinear problem with Newton's method ---

    velocity = w.sub(0, deepcopy=True)
    return velocity


def solve_tidal_flow_velocities3(turbine_positions, w, W, mesh, bcs, depth, nu, cb, g, C_T, A_T, sigma):
    u, eta = split(w)
    v, q = TestFunctions(W)
    
    n = FacetNormal(mesh)
    H = depth + eta
    f_u = Constant((0.0, 0.0))

    # Turbine field
    x, y = SpatialCoordinate(mesh)
    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += (0.5 * C_T * A_T / (2.0 * np.pi * sigma**2)
                    * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2)))

    # Residual F - ALL BARE UFL OPERATORS
    F = (inner(nu * grad(u), grad(v)) * dx
        + inner(dot(u, nabla_grad(u)), v) * dx
        - g * div(H * v) * eta * dx
        + (cb / H) * inner(u * sqrt(dot(u, u)), v) * dx
        + H * div(u) * q * dx - inner(f_u, v) * dx)

    solve(F == 0, w, bcs, 
          solver_parameters={"newton_solver": {
              "linear_solver": "mumps",
              "absolute_tolerance": 1e-8, 
              "relative_tolerance": 1e-7,
              "maximum_iterations": 30}})

    return w.sub(0, deepcopy=True)
