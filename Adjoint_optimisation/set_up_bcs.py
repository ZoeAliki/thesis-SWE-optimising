# ------------------------------------------
# set_up_bcs.py — safe version for adjoint use
# ------------------------------------------
import dolfin as dlf
import numpy as np
import matplotlib.pyplot as plt


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

def place_turbines_random2(mesh, Lx, Ly, n_turbines, min_spacing, D, plot=True, show = True, seed=None, margin=None, max_attempts=100):

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
    
    else: 
        print("Turbines placed successfully.")


    initial_positions = np.array(positions)


    if show:
        print("The initial turbine positions are:")
        for i, pos in enumerate(initial_positions):
            print(f" Turbine {i+1}: x = {pos[0]:.2f} m, y = {pos[1]:.2f} m")
    else:
        print("Turbine coordinates display not requested.")

    if plot:
        plt.figure()    
        plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells(),  alpha=0.1)
        for pos in initial_positions:
            circle = plt.Circle((pos[0], pos[1]), D/2, color='r', alpha=0.5)
        plt.gca().add_artist(circle)            
        plt.xlim(0, Lx)
        plt.ylim(0, Ly) 
        plt.title('Initial Turbine Positions')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        plt.close()

    return np.array(positions)



import numpy as np

def place_turbines_rectangular_equal_spacing(mesh, Lx, Ly, n_turbines,
                                             min_spacing, D,  plot=True,
                                             nx=None, ny=None,
                                             margin=None):
    if margin is None:
        margin = max(min_spacing / 2.0, D)

    # Normalize plot flag

    avail_x = Lx - 2.0 * margin
    avail_y = Ly - 2.0 * margin
    if avail_x <= 0 or avail_y <= 0:
        raise ValueError("Domain too small for margins")

    # initial target spacing
    S = np.sqrt((avail_x * avail_y) / float(n_turbines))

    # initial grid counts based on S
    nx = nx or max(1, int(np.floor(avail_x / S)) + 1)
    ny = ny or max(1, int(np.floor(avail_y / S)) + 1)

    # ensure enough cells
    while nx * ny < n_turbines:
        # add to the longer-direction grid first
        if (avail_x / nx) >= (avail_y / ny):
            nx += 1
        else:
            ny += 1

    # spacing from counts (handle nx/ny==1 case)
    Sx = avail_x / (nx - 1) if nx > 1 else avail_x
    Sy = avail_y / (ny - 1) if ny > 1 else avail_y
    S = min(Sx, Sy)

    # recompute counts to fit spacing S exactly
    nx = max(1, int(np.floor(avail_x / S)) + 1)
    ny = max(1, int(np.floor(avail_y / S)) + 1)

    # center the grid
    offset_x = margin + 0.5 * (avail_x - (nx - 1) * S)
    offset_y = margin + 0.5 * (avail_y - (ny - 1) * S)
    xs = offset_x + np.arange(nx) * S
    ys = offset_y + np.arange(ny) * S

    xv, yv = np.meshgrid(xs, ys)
    positions = np.column_stack([xv.ravel(), yv.ravel()])

    # greedy filter to respect min_spacing
    filtered = []
    for p in positions:
        p = np.asarray(p)
        if all(np.linalg.norm(p - np.asarray(q)) >= min_spacing for q in filtered):
            filtered.append(p)
        if len(filtered) >= n_turbines:
            break
    positions = np.array(filtered)[:n_turbines]

    # optional quick plot for verification
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells(), alpha=0.1)

        for pos in positions:
            circ = plt.Circle((pos[0], pos[1]), D/2.0, color='r', alpha=0.5)
            plt.gca().add_artist(circ)
        plt.xlim(0, Lx); plt.ylim(0, Ly); plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Rectangular — equal spacing S={:.2f}'.format(S))
        plt.show()

    return positions