import numpy as np
from dolfin import * #imports all fuctions within dolfin
import ufl
from ufl import exp
import matplotlib.pyplot as plt
import random
from matplotlib.tri import Triangulation
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.ticker import FormatStrFormatter
from copy import deepcopy


def plot_velocity_field(w, turbine_positions, sigma, show_plot=True):
    if not show_plot:
        return
        
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
    cb = plt.colorbar(c, label="|u| [m/s]")

    # Force 2 decimal places on colorbar ticks
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Fix y-axis ticks EXACTLY - force specific locations
    ax = plt.gca()
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_yticklabels(['0.00', '100.00', '200.00', '300.00', '400.00'])

    # Turbines
    for (x_i, y_i) in turbine_positions:
        plt.plot(x_i, y_i, "wo", markersize=6, markeredgecolor="k", zorder=5)
        circle = plt.Circle((x_i, y_i), 2 * sigma, color="w", linestyle="--", 
                           fill=False, linewidth=1, zorder=4)
        plt.gca().add_artist(circle)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Velocity field with turbine locations")
    plt.tight_layout()
    plt.show()


def compute_power_field_plot(C_T, rho, A_T, w, turbine_positions, sigma, show_plot=True):
    if not show_plot:
        return
        
    velocity = w.sub(0, deepcopy=True)
    mesh = velocity.function_space().mesh()
    coords = mesh.coordinates()
    cells = mesh.cells()

    ux = velocity.sub(0).compute_vertex_values(mesh)
    uy = velocity.sub(1).compute_vertex_values(mesh)
    U = (ux**2 + uy**2)**0.5

    # Scale power density by 1e-6 for MW/m² display
    power_density_raw = 0.5 * rho * C_T * A_T * U**3
    power_density = power_density_raw / 1e6  # Convert to MW/m²

    triang = Triangulation(coords[:, 0], coords[:, 1], cells)

    plt.figure(figsize=(8, 6))
    c = plt.tricontourf(triang, power_density, levels=50, cmap="plasma")
    cb = plt.colorbar(c, label="Power density [MW/m²]")

    # Force 2 decimal places on colorbar ticks
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Fix y-axis ticks
    ax = plt.gca()
    ax.set_yticks([0, 100, 200, 300, 400])
    ax.set_yticklabels(['0.00', '100.00', '200.00', '300.00', '400.00'])

    # Mesh and turbines
    plt.triplot(triang, color="gray", linewidth=0.1, alpha=0.3)
    for (x_i, y_i) in turbine_positions:
        plt.plot(x_i, y_i, "wo", markersize=6, markeredgecolor="k", zorder=5)
        circle = plt.Circle((x_i, y_i), 2 * sigma, color="w", linestyle="--", 
                           fill=False, linewidth=1, zorder=4)
        plt.gca().add_artist(circle)

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Local power density and turbine locations")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
from dolfin import plot

def compare_layouts(mesh, Lx, Ly, initial_positions, optimised_positions, sigma):
    """
    Plot initial and optimised turbine layouts on the same figure.
    """
    plt.figure(figsize=(7, 5))
    #plot(mesh)  # underlying mesh
    plt.scatter(
        [p[0] for p in initial_positions],
        [p[1] for p in initial_positions],
        c='gray', marker='o', s=60,
        label='Initial layout'
    )
    plt.scatter(
        [p[0] for p in optimised_positions],
        [p[1] for p in optimised_positions],
        c='red', marker='x', s=80,
        label='Optimised layout'
    )

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Turbine layouts: initial vs optimised')
    plt.legend(loc='best')
    #plt.xlim(0, Lx)
    #plt.ylim(0, Ly)
    plt.axis('equal')
    #plt.tight_layout()
    plt.show()
