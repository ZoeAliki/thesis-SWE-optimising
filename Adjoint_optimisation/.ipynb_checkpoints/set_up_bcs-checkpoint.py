#Boundary conditions
from dolfin import SubDomain, MeshFunction, DirichletBC, Constant, near


def setup_boundary_markers_and_bcs(mesh, W, Lx, Ly, U_inflow):
    """
    Create boundary markers and apply Dirichlet BCs (inlet & no-slip walls).
    """

    # --- Define boundary subdomains ---
    class InletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)

    class OutflowBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], Lx)

    class WallBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[1], 0.0) or near(x[1], Ly))

    # --- Create and mark boundary facets ---
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
    inlet  = InletBoundary(); inlet.mark(boundary_markers, 1)
    outlet = OutflowBoundary(); outlet.mark(boundary_markers, 2)
    walls  = WallBoundary();   walls.mark(boundary_markers, 3)

    # --- Define velocity BCs ---
    inflow_expr = Constant((U_inflow, 0.0)) #only velocity in x direction, none in y
    #wall_expr   = Constant((0.0, 0.0))  # no-slip / free-slip variant if needed

    bc_inflow = DirichletBC(W.sub(0), inflow_expr, boundary_markers, 1)
    #bc_walls  = DirichletBC(W.sub(0), wall_expr,   boundary_markers, 3)

    # Outflow left open (natural)
    bcs = [bc_inflow]

    print(f"✅ Boundary markers created and BCs applied:")
    print(f"   - Inlet  (ID=1): inflow velocity = {U_inflow} m/s")
    print( "   - Outlet (ID=2): open boundary (no Dirichlet condition)")
    #print( "   - Walls  (ID=3): no-slip velocity = (0, 0)")
    print( "   - Walls  (ID=3): free‑slip (no Dirichlet BC)")

    return boundary_markers, bcs
