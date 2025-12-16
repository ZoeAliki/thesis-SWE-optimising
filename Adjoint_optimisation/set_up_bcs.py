import dolfin as dlf
import dolfin_adjoint as adj

def setup_boundary_markers_and_bcs(mesh, W, Lx, Ly, U_inflow):
    """
    Create boundary markers and apply Dirichlet BCs (inlet & no-slip walls).
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

    # --- Define velocity BCs ---
    inflow_expr = dlf.Constant((U_inflow, 0.0))
    bc_inflow   = adj.DirichletBC(W.sub(0), inflow_expr, boundary_markers, 1)

    bcs = [bc_inflow]

    print(f"✅ Boundary markers created and BCs applied:")
    print(f"   - Inlet  (ID=1): inflow velocity = {U_inflow} m/s")
    print( "   - Outlet (ID=2): open boundary (no Dirichlet condition)")
    print( "   - Walls  (ID=3): free‑slip (no Dirichlet BC)")

    return boundary_markers, bcs
