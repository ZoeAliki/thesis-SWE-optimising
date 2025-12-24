import matplotlib.pyplot as plt
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
    v, q   = dlf.TestFunctions(W)   # ✅ returns vector and scalar arguments

    print("W type:", type(W))
    print("v type:", type(v))
    print("q type:", type(q))
    print("❤️ Mesh and mixed space initialised correctly (FEniCS 2019)")

    if showplot:
        plt.figure(figsize=(6,5))
        dlf.plot(mesh_dolfin)
        plt.title("Computational mesh verification")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.tight_layout()
        plt.show()

    return mesh_dolfin, W, w, u, eta, v, q
