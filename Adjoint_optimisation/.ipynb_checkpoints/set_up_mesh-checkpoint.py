# Mesh initialization
import matplotlib.pyplot as plt
import dolfin as dlf          # base FEniCS for geometry and element definitions
import dolfin_adjoint as adj  # adjoint-aware layer for everything runtime-related

def mesh_set_up(Lx, Ly, Nx, Ny, showplot):
    # --- Build rectangular domain mesh (plain dolfin) ---
    mesh = dlf.RectangleMesh(dlf.Point(0.0, 0.0), dlf.Point(Lx, Ly), Nx, Ny)

    # --- Mixed Taylor–Hood function space (velocity P2, height P1) ---
    P2 = dlf.VectorElement("CG", mesh.ufl_cell(), 2)   # Quadratic velocity
    P1 = dlf.FiniteElement("CG", mesh.ufl_cell(), 1)   # Linear surface height
    mixed_element = dlf.MixedElement([P2, P1])
    W = dlf.FunctionSpace(mesh, mixed_element)         # ⬅ adjoint-aware FunctionSpace

    # --- State and test functions ---
    w = adj.Function(W, name="State")
    u, eta = dlf.split(w)
    v, q = dlf.TestFunctions(W)

    print("❤️ Mesh and mixed function space successfully initialised.")

    if not showplot:
        return mesh, W, w, u, eta, v, q

    # --- Optional mesh plot ---
    plt.figure(figsize=(6, 5))
    dlf.plot(mesh)
    plt.title("Computational mesh verification")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    plt.show()

    return mesh, W, w, u, eta, v, q


