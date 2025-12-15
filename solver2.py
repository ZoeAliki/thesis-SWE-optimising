from dolfin import *
import numpy as np
from dolfin import exp

def setup_tidal_solver(mesh, W, depth, nu, cb, g):
    """Prepare symbolic forms and functions reused for optimisation iterations."""

    w = Function(W)           # unknowns: [u, eta]
    u_, eta_ = split(w)
    v_, q_ = TestFunctions(W)

    H = depth + eta_
    f_u = Constant((0.0, 0.0))

    # persistent expressions for dynamic updates
    Ct_field = Function(W.sub(1).collapse())  # dummy Function to hold turbine field coefficient
    x, y = SpatialCoordinate(mesh)

    # --- Static part of residual form (independent of turbines) ---
    F_static = (
        inner(nu * grad(u_), grad(v_)) * dx
        + inner(dot(u_, nabla_grad(u_)), v_) * dx
        - g * div(H * v_) * eta_ * dx
        + (cb / H) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx
        + H * div(u_) * q_ * dx
        - inner(f_u, v_) * dx
    )

    # Return everything needed for later reuse
    return dict(F_static=F_static, w=w, u_=u_, eta_=eta_, v_=v_, q_=q_, H=H, Ct_field=Ct_field)


def update_turbine_field(Ct_field, turbine_positions, C_T, A_T, sigma):
    """Update the Gaussian turbine sink field in-place."""
    mesh = Ct_field.function_space().mesh()
    V = Ct_field.function_space()
    Ct_expr = 0
    x, y = SpatialCoordinate(mesh)
    for (x_i, y_i) in turbine_positions:
        Ct_expr += (
            0.5 * C_T * A_T / (2.0 * np.pi * sigma**2)
            * exp(-((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2))
        )
    #Ct_field.interpolate(Ct_expr)   # ✅ fast in-place update
    Ct_field.assign(project(Ct_expr, Ct_field.function_space()))


from dolfin import *
import numpy as np
from copy import deepcopy

def solve_tidal_flow_velocities_fast(
    turbine_positions, ctx, mesh, bcs, rho, C_T, A_T, sigma, U_inflow=1.0
):
    """Reuse pre-assembled forms and data from `setup_tidal_solver`,
       with safe depth and stable solver behaviour."""

    # --- Reuse cached objects ---
    w        = ctx["w"]
    F_static = ctx["F_static"]
    H_expr   = ctx["H"]       # original symbolic H = depth + eta_
    v_       = ctx["v_"]
    Ct_field = ctx["Ct_field"]

    # --- Introduce safe depth floor to prevent singularities ---
    from ufl import conditional, gt
    H_safe = conditional(gt(H_expr, 1e-3), H_expr, 1e-3)

    # --- Update turbine sink field only (Gaussian drag) ---
    update_turbine_field(Ct_field, turbine_positions, C_T, A_T, sigma)

    # --- Residual form (combine static and dynamic parts) ---
    u_ = split(w)[0]
    F = F_static + (Ct_field / H_safe) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx

    # --- Explicit Jacobian for stability ---
    J = derivative(F, w)

    # --- Optional: sensible initial condition for first solve ---
    # If solver is cold-started, initialise u = inflow profile
    if np.allclose(w.vector().norm("l2"), 0.0):
        V_sub = w.function_space().sub(0).collapse()
        u_init = interpolate(Constant((U_inflow, 0.0)), V_sub)
        assign(w.sub(0), u_init)
        print("Initial guess reset to inflow velocity field.")

    # --- Solve with damping to reduce divergence risks ---
    solver_params = {
        "newton_solver": {
            "linear_solver": "mumps",
            "absolute_tolerance": 1e-8,
            "relative_tolerance": 1e-7,
            "maximum_iterations": 20,
            "relaxation_parameter": 0.7,  # < 1 to damp Newton steps
        }
    }

    solve(F == 0, w, bcs, J=J, solver_parameters=solver_params)

    # --- Compute and return total power ---
    from src.turbines import compute_turbine_power
    velocity = w.sub(0, deepcopy=True)
    turbine_powers, _ = compute_turbine_power(velocity, turbine_positions, rho, C_T, A_T)
    total_power = float(np.sum(turbine_powers))

    print(f"✅ Power for current iteration: {total_power/1e3:.1f} kW")
    return total_power, velocity


# Inside the objective function
def objective_function(x, solver_ctx, mesh, bcs, rho, C_T, A_T, sigma, D):

    n_turbines = len(x) // 2
    turbine_positions = [(x[2*i], x[2*i+1]) for i in range(n_turbines)]
    power, _ = solve_tidal_flow_velocities_fast(turbine_positions, solver_ctx, mesh, bcs, rho, C_T, A_T, sigma)
    

    # spacing penalty
    penalty = 0.0
    for i in range(n_turbines):
        for j in range(i+1, n_turbines):
            dist = np.linalg.norm(np.array(turbine_positions[i]) - np.array(turbine_positions[j]))
            if dist < 5*D:
                penalty += (5*D - dist)**2

    return -power + 1e6 * penalty

