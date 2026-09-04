import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree


# ============================================================
# 1. LOAD THE DATASET
# ============================================================

data = pd.read_csv("xy_data.csv")

points = data[["x", "y"]].to_numpy()


# ============================================================
# 2. DEFINE THE PARAMETRIC CURVE
# ============================================================

def generate_curve(theta, M, X, n_points=5000):
    """
    Generate uniformly sampled points on the parametric curve.

    theta : angle in radians
    M     : exponential coefficient
    X     : x-axis translation
    """

    t = np.linspace(6, 60, n_points)

    oscillation = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x = (
        t * np.cos(theta)
        - oscillation * np.sin(theta)
        + X
    )

    y = (
        42
        + t * np.sin(theta)
        + oscillation * np.cos(theta)
    )

    return np.column_stack((x, y))


# ============================================================
# 3. DEFINE THE LOSS FUNCTION
# ============================================================

def loss_function(parameters):

    theta, M, X = parameters

    # Generate predicted curve
    predicted_curve = generate_curve(theta, M, X)

    # Build KDTree for efficient nearest-point search
    tree = cKDTree(predicted_curve)

    # Find nearest predicted curve point for every dataset point
    distances, indices = tree.query(points)

    nearest_points = predicted_curve[indices]

    # Calculate L1 distance
    l1_distance = np.abs(points - nearest_points).sum(axis=1)

    # Mean L1 distance
    return np.mean(l1_distance)


# ============================================================
# 4. PARAMETER BOUNDS
# ============================================================

theta_min = 0
theta_max = np.deg2rad(50)

bounds = [
    (theta_min, theta_max),  # theta
    (-0.05, 0.05),           # M
    (0, 100)                 # X
]


# ============================================================
# 5. OPTIMIZATION
# ============================================================

result = differential_evolution(
    loss_function,
    bounds=bounds,
    strategy="best1bin",
    maxiter=500,
    popsize=20,
    tol=1e-10,
    seed=42
)


# ============================================================
# 6. EXTRACT RESULTS
# ============================================================

theta, M, X = result.x

theta_degrees = np.rad2deg(theta)

print("\n" + "=" * 50)
print("OPTIMIZED PARAMETERS")
print("=" * 50)

print(f"Theta (radians) : {theta:.6f}")
print(f"Theta (degrees) : {theta_degrees:.6f}")
print(f"M               : {M:.6f}")
print(f"X               : {X:.6f}")
print(f"Mean L1 Loss    : {result.fun:.8f}")

print("=" * 50)


# ============================================================
# 7. GENERATE FINAL CURVE
# ============================================================

curve = generate_curve(theta, M, X)


# ============================================================
# 8. VISUALIZATION
# ============================================================

plt.figure(figsize=(10, 7))

# Given dataset
plt.scatter(
    data["x"],
    data["y"],
    s=8,
    alpha=0.6,
    label="Given Dataset"
)

# Predicted curve
plt.plot(
    curve[:, 0],
    curve[:, 1],
    linewidth=2,
    label="Optimized Parametric Curve"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title("Given Data vs Optimized Parametric Curve")

plt.legend()
plt.grid(True)

plt.savefig(
    "curve_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 9. FINAL EQUATION
# ============================================================

print("\nFINAL PARAMETRIC EQUATION:")

print(
    f"\nx = t*cos({theta:.6f}) "
    f"- e^({M:.6f}|t|)*sin(0.3t)*sin({theta:.6f}) "
    f"+ {X:.6f}"
)

print(
    f"\ny = 42 + t*sin({theta:.6f}) "
    f"+ e^({M:.6f}|t|)*sin(0.3t)*cos({theta:.6f})"
)