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
