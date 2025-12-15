from scipy.optimize import differential_evolution

import numpy as np

def objective_function(x, w, W, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma, D):

    from solver import (
    setup_swe_problem, setup_boundary_markers_and_bcs, solve_tidal_flow_velocities
)
    n_turbines = len(x) // 2
    turbine_positions = [(x[2 * i], x[2 * i + 1]) for i in range(n_turbines)]
    
    # Compute total power from your existing solver
    total_power, _ = solve_tidal_flow_velocities(
        turbine_positions,
        w, W, mesh, bcs, rho, depth, nu, cb, g, C_T, A_T, sigma
    )
    
    # --- Enforce minimum spacing of turbines ---
    penalty = 0.0
    for i in range(n_turbines):
        for j in range(i + 1, n_turbines):
            dist = np.linalg.norm(np.array(turbine_positions[i]) - np.array(turbine_positions[j]))
            if dist < 5 * D:  # 5 rotor diameters
                penalty += (5 * D - dist) ** 2
    
    # Return negative total power because SciPy minimizes
    return -total_power + 1e6 * penalty


