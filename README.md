# Parametric Curve Optimization

## Research and Development / AI Assignment

This project estimates the unknown parameters of a parametric curve using numerical optimization.

---

## Problem Statement

The parametric curve is defined as:

x = t*cos(θ) - e^(M|t|)*sin(0.3t)*sin(θ) + X

y = 42 + t*sin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)

The unknown parameters are:

- θ
- M
- X

The parameter ranges are:

- 0° < θ < 50°
- -0.05 < M < 0.05
- 0 < X < 100
- 6 < t < 60

---

## Approach

The dataset contains points lying on the parametric curve.

The parameter estimation process follows these steps:

1. Load the given `(x, y)` dataset.
2. Generate uniformly sampled points on the parametric curve.
3. Find the nearest generated curve point for every dataset point.
4. Calculate the L1 distance.
5. Minimize the mean L1 distance.
6. Use Differential Evolution for global optimization.
7. Fine-tune the result using local optimization.
8. Visualize the optimized curve against the given dataset.

---

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Matplotlib
- Google Colab

---

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
