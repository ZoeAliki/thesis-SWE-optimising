#solver

import matplotlib.pyplot as plt
from dolfin import (
    RectangleMesh, Point,
    VectorElement, FiniteElement, MixedElement,
    FunctionSpace, Function, split, TestFunctions,
    Constant, assign, plot, interpolate,
)

def setup_swe_problem(Lx, Ly, Nx, Ny, U_inflow):
    """
    Create mesh, mixed Taylor–Hood function space, initial condition, and
    boundary expressions for the shallow-water problem.

    Returns
    -------
    mesh : dolfin.Mesh
    W    : dolfin.FunctionSpace
    w    : dolfin.Function   (mixed [u, eta])
    u, eta : UFL components of w
    v, q   : test functions
    inflow, outflow, walls : str
        Boundary condition expression strings.
    """
    # --- Domain and mesh setup ---
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # --- Mixed Taylor–Hood function space ---
    P2 = VectorElement("P", mesh.ufl_cell(), 2)   # Quadratic velocity
    P1 = FiniteElement("P", mesh.ufl_cell(), 1)   # Linear free-surface
    mixed_element = MixedElement([P2, P1])
    W = FunctionSpace(mesh, mixed_element)

    # --- Define trial functions and test functions ---
    w = Function(W)           # Combined [u, eta]
    u, eta = split(w)
    v, q = TestFunctions(W)

    # --- Initialize uniform inflow velocity ---
    V_sub = W.sub(0).collapse()
    u_init = interpolate(Constant((U_inflow, 0.0)), V_sub)
    assign(w.sub(0), u_init)

    # --- Define boundary markers ---
    inflow = 'near(x[0], 0.0)'
    outflow = f'near(x[0], {Lx})'
    walls = f'near(x[1], 0.0) || near(x[1], {Ly})'

    # --- Mesh visualization (optional) ---
    print(f"Success! Initialized u_init with U_inflow = {U_inflow} m/s "
          f"on a {Nx}x{Ny} mesh.")
    plt.figure(figsize=(6, 5))
    plot(mesh)
    plt.title("Computational mesh verification")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.show()

    return mesh, W, w, u, eta, v, q, inflow, outflow, walls
