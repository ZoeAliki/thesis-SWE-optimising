# plots.py

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def plot_velocity_field(w, turbine_positions, sigma):
    """
    Plot the velocity field and turbine positions for a solved FEniCS mixed Function w.
    """
    velocity = w.sub(0, deepcopy=True)
    mesh = velocity.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    ux = velocity.sub(0).compute_vertex_values(mesh)
    uy = velocity.sub(1).compute_vertex_values(mesh)
    U = (ux**2 + uy**2)**0.5

    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    plt.figure(figsize=(8, 6))
    c = plt.tricontourf(triang, U, levels=50, cmap="viridis")
    plt.colorbar(c, label="|u| [m/s]")

    for (x_i, y_i) in turbine_positions:
        plt.plot(x_i, y_i, "wo", markersize=6, markeredgecolor="k", zorder=5)
        circle = plt.Circle(
            (x_i, y_i), 2 * sigma,
            color="w", linestyle="--", fill=False, linewidth=1, zorder=4
        )
        plt.gca().add_artist(circle)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Velocity field with turbine locations")
    plt.tight_layout()
    plt.show()


def compute_power_field_plot(C_T, rho, A_T, w, turbine_positions, sigma):
    """
    Compute and plot power density field given mixed Function w and turbine data.
    """
    velocity = w.sub(0, deepcopy=True)
    mesh = velocity.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    ux = velocity.sub(0).compute_vertex_values(mesh)
    uy = velocity.sub(1).compute_vertex_values(mesh)
    U = (ux**2 + uy**2)**0.5

    power_density = 0.5 * rho * C_T * A_T * U**3

    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    plt.figure(figsize=(8, 6))
    c = plt.tricontourf(triang, power_density, levels=50, cmap="plasma")
    plt.colorbar(c, label="Power density [W/m²]")

    # draw mesh as wireframe
    plt.triplot(triang, color="gray", linewidth=0.1, alpha=0.3)

    # turbines
    for (x_i, y_i) in turbine_positions:
        plt.plot(x_i, y_i, "wo", markersize=6, markeredgecolor="k", zorder=5)
        circle = plt.Circle(
            (x_i, y_i), 2 * sigma,
            color="w", linestyle="--", fill=False, linewidth=1, zorder=4
        )
        plt.gca().add_artist(circle)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Local power density and turbine locations")
    plt.tight_layout()
    plt.show()
