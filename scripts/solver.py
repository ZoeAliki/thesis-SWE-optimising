from fenics import *
from ufl import exp, conditional, gt
import numpy as np

def solve_tidal_flow_velocities(turbine_positions, w, mesh, bcs,
                                rho, depth, nu, cb, g, C_T, A_T, sigma):
    W = w.function_space()
    (u_, eta_) = split(w)
    (v_, q_) = TestFunctions(W)
    n = FacetNormal(mesh)
    H = depth + eta_
    H_safe = conditional(gt(H, 1e-3), H, 1e-3)

    x, y = SpatialCoordinate(mesh)
    Ct_field = 0
    for (x_i, y_i) in turbine_positions:
        Ct_field += 0.5 * C_T * A_T / (2.0 * np.pi * sigma**2) * exp(
            -((x - x_i)**2 + (y - y_i)**2) / (2.0 * sigma**2)
        )

    F = (inner(nu * grad(u_), grad(v_)) * dx
         + inner(dot(u_, nabla_grad(u_)), v_) * dx
         - g * div(H * v_) * eta_ * dx
         + (cb / H_safe) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx)
    F += (Ct_field / H_safe) * inner(u_ * sqrt(dot(u_, u_)), v_) * dx
    F += H_safe * div(u_) * q_ * dx

    solve(F == 0, w, bcs, solver_parameters={
        "newton_solver": {"linear_solver": "mumps"}
    })

    velocity = w.sub(0, deepcopy=True)
    return velocity
