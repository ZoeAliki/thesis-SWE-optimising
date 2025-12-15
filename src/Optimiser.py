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


import numpy as np
import matplotlib.pyplot as plt

class OptimisationMonitor:
    def __init__(self, n_turbines, mesh, Lx, Ly, plot_interval=1):
        self.iter_count = 0
        self.history = []
        self.best_val = np.inf
        self.n_turbines = n_turbines
        self.mesh = mesh
        self.Lx = Lx
        self.Ly = Ly
        self.plot_interval = plot_interval  # every N iterations

    def __call__(self, xk, convergence):
        """
        Called automatically by SciPy after each iteration.

        Parameters
        ----------
        xk : ndarray
            Current best parameter vector (flattened positions).
        convergence : float
            Current convergence metric.
        """
        self.iter_count += 1
        n_t = self.n_turbines
        positions = xk.reshape(n_t, 2)

        # Store convergence data
        self.history.append((self.iter_count, -self.best_val))  # negative since we minimize -power

        print(f"\n🔁 Iteration {self.iter_count}: convergence = {convergence:.3e}")

        # Optional plotting every few iterations
        if self.iter_count % self.plot_interval == 0:
            plt.figure(figsize=(5, 5))
            plt.triplot(self.mesh)
            plt.scatter(positions[:, 0], positions[:, 1], c='r', label="Turbines")
            plt.xlim(0, self.Lx)
            plt.ylim(0, self.Ly)
            plt.legend()
            plt.title(f"Turbine layout iteration {self.iter_count}")
            plt.show()

        # Optional stop condition if no improvement
        if convergence < 1e-4:
            print("🛑 Early stopping: Convergence below threshold.")
            return True  # stop optimisation
        return False


