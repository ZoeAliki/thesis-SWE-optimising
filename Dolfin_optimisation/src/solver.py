#solver

import matplotlib.pyplot as plt
from dolfin import (
    RectangleMesh, Point,
    VectorElement, FiniteElement, MixedElement,
    FunctionSpace, Function, split, TestFunctions,
    Constant, assign, plot, interpolate,
)
from dolfin_adjoint import *
import numpy as np



def setup_swe_problem(Lx, Ly, Nx, Ny, U_inflow, showplot):
    # --- Domain and mesh setup ---
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # --- Mixed Taylor–Hood function space ---
    P2 = VectorElement("P", mesh.ufl_cell(), 2)   # Quadratic velocity
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)   # Linear free-surface
    mixed_element = MixedElement([P2, P1])
    W = FunctionSpace(mesh, mixed_element)

    # --- Define trial functions and test functions ---
    w = Function(W)           # Combined [u, eta]
    u, eta = split(w)
    v, q = TestFunctions(W)

    # --- Initialize uniform inflow velocity ---
    V_sub = W.sub(0).collapse()
    u_init = interpolate(Constant((U_inflow, 0.0)), V_sub)
    assign(w.sub(0), u_init)

    # --- Define boundary markers ---
    inflow = 'near(x[0], 0.0)'
    outflow = f'near(x[0], {Lx})'
    walls = f'near(x[1], 0.0) || near(x[1], {Ly})'

    # --- Mesh visualization (optional) ---
    print(f"Success! Initialized u_init with U_inflow = {U_inflow} m/s "
          f"on a {Nx}x{Ny} mesh.")

    if not showplot:
        return mesh, W, w, u, eta, v, q, inflow, outflow, walls
        
    plt.figure(figsize=(6, 5))
    plot(mesh)
    plt.title("Computational mesh verification")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.show()

    return mesh, W, w, u, eta, v, q, inflow, outflow, walls

#for safety, may be necessARY LAter
#H_safe = conditional(gt(H, 1e-3), H, 1e-3)

def solve_tidal_flow_velocities(turbine_positions, w, W, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma):
# Trial and test functions 
    (u_, eta_) = split(w)      # trial: velocity, free-surface
    (v_, q_) = TestFunctions(W)

    n = FacetNormal(mesh)
    H = depth + eta_     # for full nonlinear free-surface coupling
    f_u = Constant((0, 0)) #no internal forcing term - coriolis/windstress/pressure gradient - do look into tidal driving

# --- Turbine-induced momentum sink coefficient field ----------------------
    x, y = SpatialCoordinate(mesh)

# Build Gaussian field for all turbines

    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += 0.5 * C_T * A_T / (2.0 * np.pi * sigma**2) * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))

# Define the full nonlinear residual form F
    F = (inner(nu * grad(u_), grad(v_)) * dx #viscosity
         + inner(dot(u_, nabla_grad(u_)), v_) * dx #advection
         - g * div(H * v_) * eta_ * dx
     #- g * H * eta_ * div(v_) * dx #pressure gradient for incompr LOOK AT THIS LATER, TWICE THE ELEVATION TAKEN INTO ACCOUNT??
         + (cb/H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx) #bottom friction

# Turbine momentum sink using spatially varying field
    F += (Ct_field / H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx
    F += H * div(u_) * q_ * dx - inner(f_u, v_) * dx  # Full residual (no separate L) mass conservation/continuity

# Solve nonlinear problem with Newton's method
    solve(F == 0, w, bcs,
          solver_parameters={"newton_solver": {
              "linear_solver": "mumps",  
              "absolute_tolerance": 1e-8,
              "relative_tolerance": 1e-7,
              "maximum_iterations": 20,
              "relaxation_parameter": 1.0
          }})

#compute power 
    velocity = w.sub(0, deepcopy=True)
    turbine_powers, _ = compute_turbine_power(velocity, turbine_positions, rho, C_T, A_T)
    total_power = np.sum(turbine_powers)
    print(f"The total power is {total_power/1e3:.1f} kW")

    return total_power, velocity

def solve_tidal_flow_velocities_adjoint(turbine_positions, w, W, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma):
# Trial and test functions 
    (u_, eta_) = split(w)      # trial: velocity, free-surface
    (v_, q_) = TestFunctions(W)

    n = FacetNormal(mesh)
    H = depth + eta_     # for full nonlinear free-surface coupling
    f_u = Constant((0, 0)) #no internal forcing term - coriolis/windstress/pressure gradient - do look into tidal driving

# --- Turbine-induced momentum sink coefficient field ----------------------
    x, y = SpatialCoordinate(mesh)

# Build Gaussian field for all turbines

    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += 0.5 * C_T * A_T / (2.0 * np.pi * sigma**2) * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))

# Define the full nonlinear residual form F
    F = (inner(nu * grad(u_), grad(v_)) * dx #viscosity
         + inner(dot(u_, nabla_grad(u_)), v_) * dx #advection
         - g * div(H * v_) * eta_ * dx
     #- g * H * eta_ * div(v_) * dx #pressure gradient for incompr LOOK AT THIS LATER, TWICE THE ELEVATION TAKEN INTO ACCOUNT??
         + (cb/H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx) #bottom friction

# Turbine momentum sink using spatially varying field
    F += (Ct_field / H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx
    F += H * div(u_) * q_ * dx - inner(f_u, v_) * dx  # Full residual (no separate L) mass conservation/continuity

    Cd = Function(V_ctrl, name="Cd")
    Cd.assign(Constant(0.0025))

    J_form = 0.5 * rho * inner(u_, u_)**1.5 * dx  # power‑like field
    J = assemble(J_form)


    J_F = derivative(F, w)
    problem = NonlinearVariationalProblem(F, w, bcs, J=J_F)
    solver = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["linear_solver"] = "mumps"
    solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-8
    solver.parameters["newton_solver"]["relative_tolerance"] = 1e-7
    solver.parameters["newton_solver"]["maximum_iterations"] = 30
    solver.solve()


#compute power 
    velocity = w.sub(0, deepcopy=True)
    turbine_powers, _ = compute_turbine_power(velocity, turbine_positions, rho, C_T, A_T)
    total_power = np.sum(turbine_powers)
    print(f"The total power is {total_power/1e3:.1f} kW")

    return total_power, velocity

#def objective_function1(turbine_positions, velocity_function, rho, C_T, A_T):
#    turbine_power, velocity = solve_tidal_flow_velocities(turbine_positions, w, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma)
#    return turbine_power, velocity

def setup_boundary_markers_and_bcs(mesh, W, Lx, U_inflow):
    """
    Define subdomain classes, mark boundaries with MeshFunction, and create DirichletBCs.
    
    Returns
    -------
    boundary_markers : MeshFunction
    bcs : list of DirichletBC
    """
    from dolfin import SubDomain, MeshFunction, DirichletBC, Constant, near
    
    # --- Boundary definition and marking ---
    class InletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class OutflowBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx)

    # Create MeshFunction for boundary markers
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    # Mark boundaries
    inlet = InletBoundary()
    inlet.mark(boundary_markers, 1)
    
    outflow = OutflowBoundary()
    outflow.mark(boundary_markers, 2)

    # --- Inflow velocity BC ---
    inflow_expr = Constant((U_inflow, 0.0))
    bc_inflow = DirichletBC(W.sub(0), inflow_expr, boundary_markers, 1)

    bcs = [bc_inflow]

    print(f"✅ Boundary markers created and BCs applied:")
    print(f"   - Inlet (ID=1): {U_inflow} m/s")
    print(f"   - Outflow (ID=2): marked for future use")
    
    return boundary_markers, bcs

import numpy as np
from copy import deepcopy

from dolfin import (
    Constant, SpatialCoordinate, FacetNormal,
    TestFunctions, split, grad, nabla_grad, div, inner, dot, sqrt,
    exp, dx, solve,
)

# You probably already import W, depth, nu, cb, g somewhere else in your module
# or pass them in as arguments (see below).


def solve_tidal_flow_velocities2(
    turbine_positions,
    w,
    W,
    mesh,
    bcs,
    rho,
    depth,
    nu,
    cb,
    g,
    C_T,
    A_T,
    sigma,
):
    
    # Trial and test functions
    u_, eta_ = split(w)          # unknowns: velocity, free-surface
    v_, q_ = TestFunctions(W)    # test functions

    n = FacetNormal(mesh)
    H = depth + eta_             # total water depth
    f_u = Constant((0.0, 0.0))   # no internal body forcing

    # --- Turbine-induced momentum sink coefficient field ------------------
    x, y = SpatialCoordinate(mesh)

    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += (
            0.5 * C_T * A_T / (2.0 * np.pi * sigma**2)
            * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))
        )

    # --- Nonlinear residual form F ----------------------------------------
    F = (
        inner(nu * grad(u_), grad(v_)) * dx                            # viscosity
        + inner(dot(u_, nabla_grad(u_)), v_) * dx                      # advection
        - g * div(H * v_) * eta_ * dx                                  # free-surface coupling
        + (cb / H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx           # bottom friction
    )

    # Turbine momentum sink using spatially varying field
    F += (Ct_field / H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx

    # Continuity and body force term
    F += H * div(u_) * q_ * dx - inner(f_u, v_) * dx

    # --- Solve nonlinear problem with Newton's method ---------------------
    solve(
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

    # --- Compute turbine power --------------------------------------------
    from src.turbines import compute_turbine_power  # or import at top of file

    velocity = w.sub(0, deepcopy=True)
    turbine_powers, _ = compute_turbine_power(
        velocity, turbine_positions, rho, C_T, A_T
    )
    total_power = float(np.sum(turbine_powers))
    print(f"The total power is {total_power/1e3:.1f} kW")

    return total_power, velocity


def objective_function1(
    turbine_positions_flat,
    w,
    W,
    mesh,
    bcs,
    rho,
    depth,
    nu,
    cb,
    g,
    C_T,
    A_T,
    sigma,
):
    """
    Objective wrapper for optimization: take a flat parameter vector of turbine
    positions, reshape to (x, y) list, call the PDE solver, and return total power.

    Parameters
    ----------
    turbine_positions_flat : 1D array-like, length 2*n_turbines
        [x0, y0, x1, y1, ..., x_{N-1}, y_{N-1}]
    (other parameters as in solve_tidal_flow_velocities)

    Returns
    -------
    total_power : float
    velocity : dolfin.Function
    """
    # Convert flat vector -> list of (x, y) pairs
    params = np.asarray(turbine_positions_flat, dtype=float)
    assert params.size % 2 == 0, "turbine_positions_flat must have even length"
    n_turbines = params.size // 2
    turbine_positions = [
        (params[2 * i], params[2 * i + 1]) for i in range(n_turbines)
    ]

    total_power, velocity = solve_tidal_flow_velocities2(
        turbine_positions,
        w,
        W,
        mesh,
        bcs,
        rho,
        depth,
        nu,
        cb,
        g,
        C_T,
        A_T,
        sigma,
    )
    return total_power, velocity
