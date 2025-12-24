import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle



def place_turbines_random(Lx, Ly, n_turbines, min_spacing, D, type, seed=None, margin=None, max_attempts=20):
    if seed is not None:
        np.random.seed(seed)

    if margin is None:
        margin = max(min_spacing / 2, D)

    positions = []
    attempts = 0


    while len(positions) < n_turbines and attempts < max_attempts:
        x = np.random.uniform(margin, Lx - margin)
        y = np.random.uniform(margin, Ly - margin)
        new_pos = np.array([x, y])

        if all(np.linalg.norm(new_pos - np.array(pos)) >= min_spacing for pos in positions):
            positions.append(new_pos)

        attempts += 1

    if len(positions) < n_turbines:
        raise RuntimeError("Failed to place all turbines with the given constraints.")

    return np.array(positions)