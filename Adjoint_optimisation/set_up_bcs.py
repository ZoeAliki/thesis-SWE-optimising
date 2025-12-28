# ------------------------------------------
# set_up_bcs.py — safe version for adjoint use
# ------------------------------------------
import dolfin as dlf

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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def place_turbines_random2(mesh, Lx, Ly, n_turbines, min_spacing, D, type, seed=None, margin=None, max_attempts=20):
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
    print("Managed to place ", len(positions), " turbines within ", attempts, " attempts.")

    if len(positions) < n_turbines:
        raise RuntimeError("Failed to place all turbines with the given constraints.")
    
    initial_positions = np.array(positions)
    
    
        #plotting initial turbine positions
    plt.figure()    
    plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells())
    for pos in initial_positions:
        circle = plt.Circle((pos[0], pos[1]), D, color='r', alpha=0.5)

        plt.gca().add_artist(circle)            
    plt.xlim(0, Lx)
    plt.ylim(0, Ly) 
    plt.title('Initial Turbine Positions')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    return np.array(positions)

def place_turbines_rectangular(mesh, Lx, Ly, xn, yn, n_turbines, min_spacing, D, type, seed=None, margin=5*D):

    # Calculate number of turbines in x and y directions
    x = np.linspace(margin, Lx - margin, xn)
    y = np.linspace(margin, Ly - margin, yn)
    xv, yv = np.meshgrid(x, y)
    positions = np.column_stack([xv.ravel(), yv.ravel()])[:n_turbines]


    initial_positions = np.array(positions)

    
        #plotting initial turbine positions
    plt.figure()    
    plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells())
    for pos in initial_positions:
        circle = plt.Circle((pos[0], pos[1]), D, color='r', alpha=0.5)

        plt.gca().add_artist(circle)            
    plt.xlim(0, Lx)
    plt.ylim(0, Ly) 
    plt.title('Initial Turbine Positions (Rectangular Grid)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


    return np.array(positions)

def place_turbines_rectangular2(mesh, Lx, Ly, xn, yn, n_turbines, min_spacing, D, type, seed=None, margin=5*D):
    # Calculate number of turbines in x and y directions
    x = np.linspace(margin, Lx - margin, xn)
    y = np.linspace(margin, Ly - margin, yn)
    xv, yv = np.meshgrid(x, y)
    positions = np.column_stack([xv.ravel(), yv.ravel()])[:n_turbines]

    initial_positions = positions.copy()  # Use copy() to avoid modifying original

    # Plotting initial turbine positions
    plt.figure()
    plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells())
    for pos in initial_positions:
        circle = plt.Circle((pos[0], pos[1]), D/2, color='r', alpha=0.5)  # D/2 = radius
        plt.gca().add_artist(circle)
    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.title('Initial Turbine Positions (Rectangular Grid)')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

    return positions
