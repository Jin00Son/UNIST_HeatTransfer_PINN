#!/usr/bin/env python3
"""FDM reference solution for the 2D jet-impingement heat-transfer problem."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# Dimensional constants
L = 20e-3
D_JET = 2e-3
X_JET = 0.5
K_S = 50.0
QDOT = 1.0e7
T_B = 100.0
T_INF = 25.0
H_SIDE = 25.0

# Nondimensional groups for theta = (T - T_inf) / (T_b - T_inf)
Q_STAR = QDOT * L**2 / (K_S * (T_B - T_INF))
BI_SIDE = H_SIDE * L / K_S

# Empirical top-wall jet impingement h profile
R_OVER_D = np.array(
    [0.00, 0.25, 0.50, 0.75, 1.05, 1.25, 1.50, 1.75, 1.94, 2.25, 2.50,
     2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.05]
)
H_TOP = np.array(
    [2070, 1950, 1710, 1490, 1370, 1410, 1500, 1570, 1600, 1540, 1450,
     1360, 1260, 1160, 1060, 970, 890, 815, 750, 700, 670, 665],
    dtype=float,
)
BI_OF_R_OVER_D = PchipInterpolator(R_OVER_D, H_TOP * L / K_S, extrapolate=True)


def top_bi(x_star):
    r_over_d = np.abs(np.asarray(x_star) - X_JET) * L / D_JET
    r_over_d = np.clip(r_over_d, R_OVER_D.min(), R_OVER_D.max())
    return np.maximum(BI_OF_R_OVER_D(r_over_d), 1e-12)


def node(i, j, n):
    return j * n + i


def solve_fdm(n):
    """Second-order interior FDM with second-order one-sided Robin BCs."""
    if n < 5:
        raise ValueError("Use n >= 5 for second-order boundary differences.")

    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    d = xs[1] - xs[0]
    A = lil_matrix((n * n, n * n))
    b = np.zeros(n * n)

    for j in range(n):
        for i, x in enumerate(xs):
            r = node(i, j, n)

            if j == 0:  # bottom: theta = 1
                A[r, r] = 1.0
                b[r] = 1.0
            elif j == n - 1:  # top: -theta_y = Bi_top(x) theta
                A[r, r] = 3.0 + 2.0 * d * float(top_bi(x))
                A[r, node(i, j - 1, n)] = -4.0
                A[r, node(i, j - 2, n)] = 1.0
            elif i == 0:  # left: theta_x = Bi_side theta
                A[r, r] = -3.0 - 2.0 * d * BI_SIDE
                A[r, node(i + 1, j, n)] = 4.0
                A[r, node(i + 2, j, n)] = -1.0
            elif i == n - 1:  # right: -theta_x = Bi_side theta
                A[r, r] = 3.0 + 2.0 * d * BI_SIDE
                A[r, node(i - 1, j, n)] = -4.0
                A[r, node(i - 2, j, n)] = 1.0
            else:  # interior: theta_xx + theta_yy + Q* = 0
                A[r, r] = 4.0
                A[r, node(i - 1, j, n)] = -1.0
                A[r, node(i + 1, j, n)] = -1.0
                A[r, node(i, j - 1, n)] = -1.0
                A[r, node(i, j + 1, n)] = -1.0
                b[r] = Q_STAR * d**2

    theta = spsolve(A.tocsr(), b).reshape(n, n)
    return xs, ys, theta


def sample_bilinear(xs, ys, field, points):
    values = []
    for x, y in np.asarray(points):
        i = np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2)
        j = np.clip(np.searchsorted(ys, y) - 1, 0, len(ys) - 2)
        tx = (x - xs[i]) / (xs[i + 1] - xs[i])
        ty = (y - ys[j]) / (ys[j + 1] - ys[j])
        values.append(
            (1 - tx) * (1 - ty) * field[j, i]
            + tx * (1 - ty) * field[j, i + 1]
            + (1 - tx) * ty * field[j + 1, i]
            + tx * ty * field[j + 1, i + 1]
        )
    return np.asarray(values).reshape(-1, 1)


def sensor_points():
    sx = [0.20, 0.40, 0.60, 0.80]
    sy = [0.55, 0.70, 0.85, 0.95]
    return np.array([[x, y] for y in sy for x in sx], dtype=float)


def residuals(xs, theta):
    d = xs[1] - xs[0]
    interior = (
        (theta[1:-1, 2:] - 2 * theta[1:-1, 1:-1] + theta[1:-1, :-2]) / d**2
        + (theta[2:, 1:-1] - 2 * theta[1:-1, 1:-1] + theta[:-2, 1:-1]) / d**2
        + Q_STAR
    )
    top = -(3 * theta[-1] - 4 * theta[-2] + theta[-3]) / (2 * d) - top_bi(xs) * theta[-1]
    left = (-3 * theta[1:-1, 0] + 4 * theta[1:-1, 1] - theta[1:-1, 2]) / (2 * d) - BI_SIDE * theta[1:-1, 0]
    right = -(3 * theta[1:-1, -1] - 4 * theta[1:-1, -2] + theta[1:-1, -3]) / (2 * d) - BI_SIDE * theta[1:-1, -1]
    return {
        "pde": float(np.abs(interior).max()),
        "top_robin": float(np.abs(top).max()),
        "left_robin": float(np.abs(left).max()),
        "right_robin": float(np.abs(right).max()),
        "min_theta": float(theta.min()),
        "max_theta": float(theta.max()),
    }


def convergence_report(grids, sensors):
    solutions = {}
    for n in grids:
        print(f"Solving n={n} ({n*n:,} unknowns)")
        solutions[n] = solve_fdm(n)

    xx, yy = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    field_points = np.column_stack([xx.ravel(), yy.ravel()])
    rows = []

    for nc, nf in zip(grids[:-1], grids[1:]):
        xs_c, ys_c, th_c = solutions[nc]
        xs_f, ys_f, th_f = solutions[nf]
        s_c = sample_bilinear(xs_c, ys_c, th_c, sensors)
        s_f = sample_bilinear(xs_f, ys_f, th_f, sensors)
        f_c = sample_bilinear(xs_c, ys_c, th_c, field_points).ravel()
        f_f = sample_bilinear(xs_f, ys_f, th_f, field_points).ravel()
        rows.append(
            {
                "coarse_n": nc,
                "fine_n": nf,
                "max_sensor_abs_diff_theta": float(np.abs(s_f - s_c).max()),
                "max_sensor_rel_diff": float(np.abs(s_f - s_c).max() / max(np.abs(s_f).max(), 1e-12)),
                "field_rel_l2_diff": float(np.linalg.norm(f_f - f_c) / np.linalg.norm(f_f)),
            }
        )
    return solutions, rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_sensor_csv(path, sensors, theta_sensor):
    rows = []
    for (x, y), theta in zip(sensors, theta_sensor.ravel()):
        rows.append(
            {
                "x_star": x,
                "y_star": y,
                "theta": theta,
                "temperature_C": T_INF + theta * (T_B - T_INF),
            }
        )
    write_csv(path, rows)


def save_plots(outdir, xs, ys, theta):
    temp = T_INF + theta * (T_B - T_INF)
    plt.figure(figsize=(6.2, 5.0))
    im = plt.contourf(xs * L * 1e3, ys * L * 1e3, temp, levels=50, cmap="inferno")
    plt.colorbar(im, label="Temperature [degC]")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.gca().set_aspect("equal")
    plt.tight_layout()
    plt.savefig(outdir / "fdm_reference_temperature.png", dpi=220)
    plt.close()

    x = np.linspace(0, 1, 501)
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(x, top_bi(x) * K_S / L, "k", lw=2)
    plt.xlabel("x*")
    plt.ylabel("h(x) [W/(m^2 K)]")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "top_h_profile.png", dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grids", type=int, nargs="+", default=[121, 241, 481])
    parser.add_argument("--outdir", type=Path, default=Path(__file__).with_name("fdm_reference_output"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    grids = sorted(set(args.grids))
    sensors = sensor_points()

    print(f"Q* = {Q_STAR:.8f}, Bi_side = {BI_SIDE:.8f}")
    solutions, conv = convergence_report(grids, sensors)

    n_ref = grids[-1]
    xs, ys, theta = solutions[n_ref]
    theta_sensor = sample_bilinear(xs, ys, theta, sensors)
    temp = T_INF + theta * (T_B - T_INF)
    diag = residuals(xs, theta)

    np.savez_compressed(
        outdir / "fdm_reference_jet_impingement.npz",
        xs=xs,
        ys=ys,
        theta=theta,
        temperature_C=temp,
        sensor_xy=sensors,
        theta_sensor=theta_sensor,
        Q_STAR=Q_STAR,
        BI_SIDE=BI_SIDE,
        L=L,
        D_jet=D_JET,
        k_s=K_S,
        qdot=QDOT,
        T_b=T_B,
        T_inf=T_INF,
        h_side=H_SIDE,
        R_over_D_data=R_OVER_D,
        h_data=H_TOP,
    )
    write_csv(outdir / "fdm_grid_convergence.csv", conv)
    write_csv(outdir / "fdm_diagnostics.csv", [diag])
    write_sensor_csv(outdir / "fdm_sensor_clean.csv", sensors, theta_sensor)
    save_plots(outdir, xs, ys, theta)

    print("\nGrid convergence")
    for r in conv:
        print(
            f"{r['coarse_n']:4d} -> {r['fine_n']:4d} | "
            f"sensor abs = {r['max_sensor_abs_diff_theta']:.3e}, "
            f"field L2 = {r['field_rel_l2_diff']:.3e}"
        )

    print("\nFinest-grid residuals")
    for k, v in diag.items():
        print(f"{k}: {v:.3e}")
    print(f"\nSaved n={n_ref} reference to {outdir}")


if __name__ == "__main__":
    main()
