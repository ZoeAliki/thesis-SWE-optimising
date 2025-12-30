# ------------------------------------------
# set_up_bcs.py — safe version for adjoint use
# ------------------------------------------
import dolfin as dlf
import numpy as np
import matplotlib.pyplot as plt

# Helper to coerce common 'truthy' representations to strict Python booleans
def _coerce_bool(x):
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "y", "yes")
    return bool(x)


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

def place_turbines_random2(mesh, Lx, Ly, n_turbines, min_spacing, D, seed=None, margin=None, max_attempts=100):

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

    show = False

    if show:
        print("The initial turbine positions are:")
        for i, pos in enumerate(initial_positions):
            print(f" Turbine {i+1}: x = {pos[0]:.2f} m, y = {pos[1]:.2f} m")
    else:
        print("Turbine coordinates display not requested.")

    initial_positions2 = np.array(positions)

    plot = True

    if plot:
        plt.figure()    
        plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells(),  alpha=0.1)
        for pos in initial_positions2:
            circle = plt.Circle((pos[0], pos[1]), D*2, color='r', alpha=0.5)

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


# -----------------------------------------------------------------------------
# Additional helpers: rectangular grid at minimum spacing and field builder
# -----------------------------------------------------------------------------
def place_turbines_rectangular_min_spacing(mesh, Lx, Ly, min_spacing, D, margin=None, max_turbines=None, plot=True):
    """Place turbines on a rectangular grid with spacing as close as possible to `min_spacing`.

    Returns: positions (N x 2 numpy array)
    """
    plot = _coerce_bool(plot)
    if margin is None:
        margin = max(min_spacing / 2.0, D)

    avail_x = Lx - 2.0 * margin
    avail_y = Ly - 2.0 * margin
    if avail_x <= 0 or avail_y <= 0:
        raise ValueError("Domain too small for margins")

    nx = max(1, int(np.floor(avail_x / min_spacing)) + 1)
    ny = max(1, int(np.floor(avail_y / min_spacing)) + 1)

    # Recompute spacing so grid fits within available area and center it
    Sx = avail_x / (nx - 1) if nx > 1 else avail_x
    Sy = avail_y / (ny - 1) if ny > 1 else avail_y
    S = min(Sx, Sy)

    nx = max(1, int(np.floor(avail_x / S)) + 1)
    ny = max(1, int(np.floor(avail_y / S)) + 1)

    offset_x = margin + 0.5 * (avail_x - (nx - 1) * S)
    offset_y = margin + 0.5 * (avail_y - (ny - 1) * S)
    xs = offset_x + np.arange(nx) * S
    ys = offset_y + np.arange(ny) * S

    xv, yv = np.meshgrid(xs, ys)
    positions = np.column_stack([xv.ravel(), yv.ravel()])

    if max_turbines is not None:
        positions = positions[:int(max_turbines)]

    # Optionally plot
    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.triplot(mesh.coordinates()[:,0], mesh.coordinates()[:,1], mesh.cells(), alpha=0.1)
        for pos in positions:
            circ = plt.Circle((pos[0], pos[1]), D/2.0, color='r', alpha=0.5)
            plt.gca().add_artist(circ)
        plt.xlim(0, Lx); plt.ylim(0, Ly); plt.gca().set_aspect('equal', adjustable='box')
        plt.title('Rectangular grid @ min spacing S={:.2f}'.format(S))
        plt.show()

    return positions


def build_ct_field_from_positions(positions, mesh, function_space, D=1.0, sigma=None, Ct_value=1.0, normalize=False):
    """Build a turbine coefficient field (Function) on `function_space` from `positions`.

    The field is a sum of Gaussian bumps centered at turbine positions. sigma defaults to D/2.
    """
    import dolfin as dlf
    if sigma is None:
        sigma = D / 2.0

    x, y = dlf.SpatialCoordinate(mesh)
    expr = 0
    for xi, yi in positions:
        expr = expr + Ct_value * dlf.exp(-((x - float(xi))**2 + (y - float(yi))**2) / (2.0 * float(sigma**2)))

    try:
        ct_field = dlf.project(expr, function_space)
    except Exception:
        # Fallback: create Function and interpolate
        ct_field = dlf.Function(function_space)
        ct_field.interpolate(expr)

    if normalize:
        vals = ct_field.vector().get_local()
        m = vals.max() if vals.size else 1.0
        if m > 0:
            ct_field.vector().set_local(vals / m)
            ct_field.vector().apply('insert')

    return ct_field


def compute_blockage(positions, Lx, Ly, h, D, h_turbine=10, nbins=10):
    bin_edges = np.linspace(0, Lx, nbins+1)
    blockage_profile = np.zeros(nbins)

    for i in range(nbins):
        # Turbines in x-bin [bin_edges[i], bin_edges[i+1]]
        in_bin = (positions[:,0] >= bin_edges[i]) & (positions[:,0] < bin_edges[i+1])
        N_bin = np.sum(in_bin)
    
        blockage_profile[i] = (N_bin * D * h_turbine) / (Ly * h)

    mean_blockage = np.mean(blockage_profile)
    max_blockage = np.max(blockage_profile)

    return blockage_profile, mean_blockage, max_blockage


