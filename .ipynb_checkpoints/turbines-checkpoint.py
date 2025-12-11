import numpy as np
import random
import matplotlib.pyplot as plt
from dolfin import plot as fe_plot  # avoid name clash with plt.plot

# --- Turbine layout ---
#turbine_positions = [(60.0, 30.0), (60.0, 50.0)]
#n_turbines = len(turbine_positions)
def place_turbines_random(Lx, Ly, n_turbines, sigma, min_spacing):
    
    turbine_positions = []
    max_att = 500
    att = 0
    
    while len(turbine_positions) < n_turbines and att < max_att:
        x_rand = random.uniform(0.2 * Lx, 0.9 * Lx)  # avoid being too close to inlet/outlet
        y_rand = random.uniform(0.1 * Ly, 0.9 * Ly)
        att +=1
    
        if all(np.hypot(x_rand - x_i, y_rand - y_i) > min_spacing for (x_i, y_i) in turbine_positions):
            turbine_positions.append((x_rand, y_rand))
    
    if len(turbine_positions) < n_turbines:
        print(f"⚠️ Only placed {len(turbine_positions)} turbines after {att} attempts.")
        print("Try reducing min_spacing or the number of turbines.")
    else:
        print(f"✅ Successfully placed {n_turbines} turbines after {att} attempts.")
    
    print(f"Initialized {n_turbines} turbines successfully.")
    
    #Show the initial placement of the turbines
    return turbine_positions



def plot_turbine_layout(mesh, turbine_positions, Lx, Ly, sigma):
    plt.figure(figsize=(7, 5))
    fe_plot(mesh, linewidth=0.2, color="lightgray")

    for (x_i, y_i) in turbine_positions:
        plt.plot(x_i, y_i, 'ro', markersize=6)
        circle = plt.Circle(
            (x_i, y_i), 2*sigma,
            color='r', fill=False, linestyle='--', linewidth=1
        )
        plt.gca().add_artist(circle)

    plt.xlim(0, Lx)
    plt.ylim(0, Ly)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"Random placement of {len(turbine_positions)} turbines")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_turbine_power(velocity_function, turbine_positions, rho, C_T, A_T):
    powers, velocities = [], []

    for (x_i, y_i) in turbine_positions:
        u_local = velocity_function((x_i, y_i))
        speed = np.sqrt(u_local[0]**2 + u_local[1]**2)
        P_i = 0.5 * rho * C_T * A_T * speed**3
        powers.append(P_i)
        velocities.append(speed)

    return np.array(powers), np.array(velocities)

def summarize_turbine_power(powers, velocities, rho, C_T, A_T, U_inflow, n_turbines):
    C_P = C_T * (1 - 0.5 * C_T)
    P_theoretical = 0.5 * rho * C_P * A_T * U_inflow**3
    P_thrust_based = 0.5 * rho * C_T * A_T * U_inflow**3
    Max_park_power = P_thrust_based * n_turbines

    total_power = np.sum(powers)

    print(f"Maximum theoretical power per turbine (Betz-adjusted): {P_theoretical/1e3:.2f} kW")
    print(f"Momentum-sink (raw thrust) power per turbine:          {P_thrust_based/1e3:.2f} kW")
    print(f"Max total power: {Max_park_power/1e6:.2f} MW\n")

    print("Turbine performance summary:")
    print("-" * 55)
    for i, (P, v) in enumerate(zip(powers, velocities), 1):
        print(f"Turbine {i:2d}: Velocity = {v:.2f} m/s | Power = {P/1e3:.2f} kW")
    print("-" * 55)
    print(f"Total extracted power: {total_power/1e6:.2f} MW")
    print(f"Fraction of park capacity: {total_power / Max_park_power * 100:.2f}%\n")

    # Optionally return numeric results
    return total_power, Max_park_power, P_theoretical, P_thrust_based    
