#Mesh initialising 
from dolfin import *
def mesh_set_up(Lx, Ly, Nx, Ny):
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), Nx, Ny)

    # --- Mixed Taylor–Hood function space ---
    P2 = VectorElement("CG", mesh.ufl_cell(), 2)   # Quadratic velocity
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)   # Linear surface height
    mixed_element = MixedElement([P2, P1])
    W = FunctionSpace(mesh, mixed_element)

    # --- State and test functions ---
    w = Function(W, name="State")
    u, eta = split(w)
    v, q = TestFunctions(W)

    print("❤️ Mesh and mixed function space succesfully initialised.")
    return mesh, W, w, u, eta, v, q



