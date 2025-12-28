# ------------------------------------------
# set_up_bcs.py — safe version for adjoint use
# ------------------------------------------
import dolfin as dlf

def setup_boundary_markers_and_bcs(mesh, W, Lx, Ly, U_inflow):
    """
    Create boundary markers and apply Dirichlet BCs (inlet & no‑slip walls).
    """

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

    # --- Define velocity BC (inlet only) ---
    inflow_expr = dlf.Constant((U_inflow, 0.0))
    bc_inflow = dlf.DirichletBC(W.sub(0), inflow_expr, boundary_markers, 1)

    bcs = [bc_inflow]

    print(f"✅ Boundary markers created and BCs applied:")
    print(f"   - Inlet  (ID=1): inflow velocity = {U_inflow:.2f} m/s")
    print( "   - Outlet (ID=2): open boundary (no Dirichlet condition)")
    print( "   - Walls  (ID=3): free‑slip (no Dirichlet BC)\n")

    return boundary_markers, bcs


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def place_turbines_random2(Lx, Ly, n_turbines, min_spacing, D, type, seed=None, margin=None, max_attempts=20):
    if seed is not None:
        np.random.seed(seed)

    if margin is None:
        margin = max(min_spacing / 2, D)

    positions = []
    attempts = 0


    while len(positions) < n_turbines and attempts < max_attempts:
        x = np.random.uniform(margin, Lx - margin)
        y = np.random.uniform(margin, Ly - margin)
        new_pos = np.array([x, y])

        if all(np.linalg.norm(new_pos - np.array(pos)) >= min_spacing for pos in positions):
            positions.append(new_pos)

        attempts += 1

    if len(positions) < n_turbines:
        raise RuntimeError("Failed to place all turbines with the given constraints.")

    return np.array(positions)