#solver

import matplotlib.pyplot as plt
from dolfin import (
    RectangleMesh, Point,
    VectorElement, FiniteElement, MixedElement,
    FunctionSpace, Function, split, TestFunctions,
    Constant, assign, plot, interpolate,
)

def setup_swe_problem(Lx, Ly, Nx, Ny, U_inflow):
    """
    Create mesh, mixed Taylor–Hood function space, initial condition, and
    boundary expressions for the shallow-water problem.

    Returns
    -------
    mesh : dolfin.Mesh
    W    : dolfin.FunctionSpace
    w    : dolfin.Function   (mixed [u, eta])
    u, eta : UFL components of w
    v, q   : test functions
    inflow, outflow, walls : str
        Boundary condition expression strings.
    """
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
              "linear_solver": "mumps",  # or "petsc" with good preconditioner
              "absolute_tolerance": 1e-8,
              "relative_tolerance": 1e-7,
              "maximum_iterations": 30,
              "relaxation_parameter": 1.0
          }})

#compute power 
    velocity = w.sub(0, deepcopy=True)
    turbine_powers, _ = compute_turbine_power(velocity, turbine_positions, rho, C_T, A_T)
    total_power = np.sum(turbine_powers)
    print(f"The total power is {total_power/1e3:.1f} kW")

    return total_power, velocity

#def objective_function1(turbine_positions, velocity_function, rho, C_T, A_T):
#    turbine_power, velocity = solve_tidal_flow_velocities(turbine_positions, w, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma)
#    return turbine_power, velocity
