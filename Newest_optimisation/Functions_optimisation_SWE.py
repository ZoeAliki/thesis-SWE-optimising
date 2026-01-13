#file for functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import dolfin as dlf


def mesh_set_up(Lx, Ly, Nx, Ny, showplot):
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

    if showplot:
        plt.figure(figsize=(6,5))
        dlf.plot(mesh_dolfin)
        plt.title("Computational mesh")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.show()

    return mesh_dolfin, W, w, u, eta, v, q

def setup_boundary_markers_and_bcs(mesh, W, Lx, Ly, U_inflow):

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

    print(f"Boundary markers created and BCs applied:")
    print(f"   - Inlet  (ID=1): inflow velocity = {U_inflow:.2f} m/s")
    print( "   - Outlet (ID=2): open boundary (no Dirichlet BC)")
    print( "   - Walls  (ID=3): free‑slip (no Dirichlet BC)\n")

    return boundary_markers, bcs

def place_turbines_random(mesh, Lx, Ly, n_turbines, min_spacing, D, seed=None, margin=None, max_attempts=100):

    if seed is not None:
        np.random.seed(seed)

    # Optional additional inner margin (defaults to 0)
    if margin is None:
        margin = 0.0

    # Enforce placement rectangle: x in [5*D, Lx-10*D], y in [2*D, Ly-2*D]
    xmin = 5.0 * D + margin
    xmax = Lx - 10.0 * D - margin
    ymin = 2.0 * D + margin
    ymax = Ly - 2.0 * D - margin

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