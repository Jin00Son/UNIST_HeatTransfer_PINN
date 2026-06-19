#!/usr/bin/env python3
"""
Phase 3 final submission code: inverse PINN for top-wall CHTC/Bi recovery.
"""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import PchipInterpolator
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from torch.optim.lr_scheduler import CosineAnnealingLR


# =============================================================================
# Step 0. User settings
# =============================================================================

SEED = 2026
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

SUBMISSION_DIR = Path(__file__).resolve().parent
OUTDIR = SUBMISSION_DIR / "Phase3_Final_Output"
FDM_N = 481

RUN_NAME = "phase3_final_results"
SENSOR_X = [0.02, 0.12, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.88, 0.98]
SENSOR_Y = [0.98, 1.00]

N_DOMAIN = 3000
N_BOUNDARY = 400
HIDDEN_LAYERS = 4
NEURONS = 64

AUX_BI_HIDDEN_WIDTH = 16
AUX_BI_HIDDEN_LAYERS = 2
H_MIN_GUESS = 500.0
H_MAX_GUESS = 2000.0

W_PDE = 1.0
W_BOTTOM = 10.0
W_TOP = 50.0
W_SIDE = 10.0
W_DATA = 100.0
W_THETA_Y_DATA = 10.0

N_ADAM = 15000
PRINT_EVERY = 500
PINN_LR = 1.0e-3
BI_LR = 1.0e-3
ADAM_ETA_MIN = 1.0e-5
RUN_LBFGS = True
LBFGS_STEPS = 600


# =============================================================================
# Step 1. Physical constants and empirical CHTC data
# =============================================================================

L = 20e-3
D_JET = 2e-3
X_JET = 0.5
K_S = 50.0
QDOT = 1.0e7
T_B = 100.0
T_INF = 25.0
H_SIDE = 25.0

Q_STAR = QDOT * L**2 / (K_S * (T_B - T_INF))
BI_SIDE = H_SIDE * L / K_S
R_OVER_D_DATA = np.array([0.00, 0.25, 0.50, 0.75, 1.05, 1.25, 1.50, 1.75, 1.94, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00, 5.05], dtype=float)
H_DATA = np.array([2070, 1950, 1710, 1490, 1370, 1410, 1500, 1570, 1600, 1540, 1450, 1360, 1260, 1160, 1060, 970, 890, 815, 750, 700, 670, 665], dtype=float)
BI_INTERP = PchipInterpolator(R_OVER_D_DATA, H_DATA * L / K_S, extrapolate=True)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_bi_numpy(x_star: np.ndarray) -> np.ndarray:
    r_over_d = np.abs(np.asarray(x_star) - X_JET) * L / D_JET
    return BI_INTERP(r_over_d)


def to_temperature(theta: np.ndarray) -> np.ndarray:
    return T_INF + theta * (T_B - T_INF)


# =============================================================================
# Step 2. FDM reference and sensor data
# =============================================================================


def node(i: int, j: int, n: int) -> int:
    return j * n + i


def solve_fdm_reference(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve theta_xx + theta_yy + Q* = 0 with the same BCs used by the PINN."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    d = xs[1] - xs[0]
    A = lil_matrix((n * n, n * n))
    b = np.zeros(n * n)

    for j in range(n):
        for i, x in enumerate(xs):
            row = node(i, j, n)

            if j == 0:
                A[row, row] = 1.0
                b[row] = 1.0
            elif j == n - 1:
                A[row, row] = 3.0 + 2.0 * d * float(true_bi_numpy(x))
                A[row, node(i, j - 1, n)] = -4.0
                A[row, node(i, j - 2, n)] = 1.0
            elif i == 0:
                A[row, row] = -3.0 - 2.0 * d * BI_SIDE
                A[row, node(i + 1, j, n)] = 4.0
                A[row, node(i + 2, j, n)] = -1.0
            elif i == n - 1:
                A[row, row] = 3.0 + 2.0 * d * BI_SIDE
                A[row, node(i - 1, j, n)] = -4.0
                A[row, node(i - 2, j, n)] = 1.0
            else:
                A[row, row] = 4.0
                A[row, node(i - 1, j, n)] = -1.0
                A[row, node(i + 1, j, n)] = -1.0
                A[row, node(i, j - 1, n)] = -1.0
                A[row, node(i, j + 1, n)] = -1.0
                b[row] = Q_STAR * d**2

    theta = spsolve(A.tocsr(), b).reshape(n, n)
    return xs, ys, theta


def build_reference() -> dict:
    xs, ys, theta = solve_fdm_reference(FDM_N)
    return {"source": f"FDM solved inside Phase3 script, n={FDM_N}", "xs": xs, "ys": ys, "theta": theta}


def make_sensor_points(sensor_x, sensor_y) -> np.ndarray:
    return np.array([[x, y] for y in sensor_y for x in sensor_x], dtype=float)


def sample_bilinear(xs: np.ndarray, ys: np.ndarray, field: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = []
    for x, y in points:
        i = np.searchsorted(xs, x) - 1
        j = np.searchsorted(ys, y) - 1
        tx = (x - xs[i]) / (xs[i + 1] - xs[i])
        ty = (y - ys[j]) / (ys[j + 1] - ys[j])
        values.append(
            (1 - tx) * (1 - ty) * field[j, i]
            + tx * (1 - ty) * field[j, i + 1]
            + (1 - tx) * ty * field[j + 1, i]
            + tx * ty * field[j + 1, i + 1]
        )
    return np.asarray(values).reshape(-1, 1)


def build_theta_y_data(points: np.ndarray, theta_values: np.ndarray):
    """Build theta_y labels from the final 2-row sensor layout."""
    n_x = len(points) // 2
    lower = points[:n_x]
    upper = points[n_x:]
    dy = upper[:, 1:2] - lower[:, 1:2]
    theta_y = (theta_values[n_x:] - theta_values[:n_x]) / dy
    return {
        "lower": lower,
        "upper": upper,
        "theta_y": theta_y,
    }


# =============================================================================
# Step 3. Models and physics losses
# =============================================================================


class PINN2D(nn.Module):
    def __init__(self, hidden_layers: int, neurons: int):
        super().__init__()
        layers = [nn.Linear(2, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), nn.Tanh()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, xy):
        return self.net(xy)


class AuxiliaryBiNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(1, AUX_BI_HIDDEN_WIDTH), nn.Tanh()]
        for _ in range(AUX_BI_HIDDEN_LAYERS - 1):
            layers += [nn.Linear(AUX_BI_HIDDEN_WIDTH, AUX_BI_HIDDEN_WIDTH), nn.Tanh()]
        layers.append(nn.Linear(AUX_BI_HIDDEN_WIDTH, 1))
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        init_bi = 0.5 * (H_MIN_GUESS + H_MAX_GUESS) * L / K_S
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, np.log(np.expm1(init_bi)))

    def forward(self, x):
        return self.softplus(self.net(x))


def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def mse(x):
    return torch.mean(x**2)


def sample_lhs(n: int, seed: int):
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, n + 1)
    x = rng.uniform(edges[:-1], edges[1:])
    y = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(x)
    rng.shuffle(y)
    return torch.tensor(np.c_[x, y], dtype=torch.float64, device=DEVICE).requires_grad_(True)


def edge_points(edge: str, n: int):
    s = torch.linspace(0.0, 1.0, n, dtype=torch.float64, device=DEVICE).reshape(-1, 1)
    z, o = torch.zeros_like(s), torch.ones_like(s)
    if edge == "bottom":
        return torch.cat([s, z], dim=1)
    if edge == "top":
        return torch.cat([s, o], dim=1)
    if edge == "left":
        return torch.cat([z, s], dim=1)
    return torch.cat([o, s], dim=1)


def total_loss(pinn, bi_model, data):
    xy_f = data["xy_f"]
    theta = pinn(xy_f)
    g = grad(theta, xy_f)
    pde = mse(grad(g[:, 0:1], xy_f)[:, 0:1] + grad(g[:, 1:2], xy_f)[:, 1:2] + Q_STAR)

    bottom = mse(pinn(data["bottom"]) - 1.0)

    top_xy = data["top"].clone().detach().requires_grad_(True)
    theta_top = pinn(top_xy)
    top = mse(-grad(theta_top, top_xy)[:, 1:2] - bi_model(top_xy[:, 0:1]) * theta_top)

    left_xy = data["left"].clone().detach().requires_grad_(True)
    right_xy = data["right"].clone().detach().requires_grad_(True)
    theta_l, theta_r = pinn(left_xy), pinn(right_xy)
    side = mse(grad(theta_l, left_xy)[:, 0:1] - BI_SIDE * theta_l) + mse(-grad(theta_r, right_xy)[:, 0:1] - BI_SIDE * theta_r)

    sensor = mse(pinn(data["sensor_xy_t"]) - data["theta_sensor_t"])
    theta_y_pred = (pinn(data["upper_t"]) - pinn(data["lower_t"])) / (data["upper_t"][:, 1:2] - data["lower_t"][:, 1:2])
    theta_y_data = mse(theta_y_pred - data["theta_y_t"])

    loss = W_PDE * pde + W_BOTTOM * bottom + W_TOP * top + W_SIDE * side + W_DATA * sensor + W_THETA_Y_DATA * theta_y_data
    return loss, {"pde": pde, "bottom": bottom, "top": top, "side": side, "data": sensor, "theta_y_data": theta_y_data}


# =============================================================================
# Step 4. Training and evaluation
# =============================================================================


def prepare_case(ref: dict) -> dict:
    sensors = make_sensor_points(SENSOR_X, SENSOR_Y)
    theta_clean = sample_bilinear(ref["xs"], ref["ys"], ref["theta"], sensors)
    dy_data = build_theta_y_data(sensors, theta_clean)

    case = {
        "sensor_xy": sensors,
        "theta_clean": theta_clean,
        "dy_data": dy_data,
        "xy_f": sample_lhs(N_DOMAIN, SEED),
        "bottom": edge_points("bottom", N_BOUNDARY),
        "top": edge_points("top", N_BOUNDARY),
        "left": edge_points("left", N_BOUNDARY),
        "right": edge_points("right", N_BOUNDARY),
    }
    case.update(
        {
            "sensor_xy_t": torch.tensor(sensors, dtype=torch.float64, device=DEVICE),
            "theta_sensor_t": torch.tensor(theta_clean, dtype=torch.float64, device=DEVICE),
            "lower_t": torch.tensor(dy_data["lower"], dtype=torch.float64, device=DEVICE),
            "upper_t": torch.tensor(dy_data["upper"], dtype=torch.float64, device=DEVICE),
            "theta_y_t": torch.tensor(dy_data["theta_y"], dtype=torch.float64, device=DEVICE),
        }
    )
    return case


def train(case: dict):
    pinn = PINN2D(HIDDEN_LAYERS, NEURONS).to(DEVICE)
    bi_model = AuxiliaryBiNN().to(DEVICE)
    optimizer = torch.optim.Adam([{"params": pinn.parameters(), "lr": PINN_LR}, {"params": bi_model.parameters(), "lr": BI_LR}])
    scheduler = CosineAnnealingLR(optimizer, T_max=N_ADAM, eta_min=ADAM_ETA_MIN)
    history = []
    t0 = time.perf_counter()

    print(f"device: {DEVICE}")
    print(f"sensors: {len(case['sensor_xy'])}, theta_y labels: {len(case['dy_data']['theta_y'])}, method: 2-row finite difference")
    for epoch in range(1, N_ADAM + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = total_loss(pinn, bi_model, case)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch == 1 or epoch % PRINT_EVERY == 0:
            row = {"epoch": epoch, "total": float(loss.detach().cpu())}
            row.update({k: float(v.detach().cpu()) for k, v in parts.items()})
            history.append(row)
            print(f"Adam {epoch:6d} | L={row['total']:.3e} Top={row['top']:.2e} Data={row['data']:.2e} dTy={row['theta_y_data']:.2e} | {time.perf_counter() - t0:.1f}s")

    if RUN_LBFGS:
        lbfgs = torch.optim.LBFGS(list(pinn.parameters()) + list(bi_model.parameters()), lr=1.0, max_iter=20, history_size=50, line_search_fn="strong_wolfe")

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss, _ = total_loss(pinn, bi_model, case)
            loss.backward()
            return loss

        for step in range(1, LBFGS_STEPS + 1):
            lbfgs.step(closure)
            if step == 1 or step % 50 == 0:
                loss, parts = total_loss(pinn, bi_model, case)
                row = {"epoch": N_ADAM + step, "total": float(loss.detach().cpu())}
                row.update({k: float(v.detach().cpu()) for k, v in parts.items()})
                history.append(row)
                print(f"L-BFGS {step:4d} | L={row['total']:.3e} Top={row['top']:.2e} dTy={row['theta_y_data']:.2e}")

    return pinn, bi_model, history, time.perf_counter() - t0


def evaluate(pinn, bi_model, ref: dict, case: dict) -> dict:
    xs, ys, theta_ref = ref["xs"], ref["ys"], ref["theta"]
    xx, yy = np.meshgrid(xs, ys)
    with torch.no_grad():
        theta_pred = pinn(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64, device=DEVICE)).cpu().numpy().reshape(len(ys), len(xs))
        theta_sensor_pred = pinn(case["sensor_xy_t"]).cpu().numpy()
        theta_y_pred = ((pinn(case["upper_t"]) - pinn(case["lower_t"])) / (case["upper_t"][:, 1:2] - case["lower_t"][:, 1:2])).cpu().numpy()
        x_plot = np.linspace(0.0, 1.0, 501)[:, None]
        bi_pred = bi_model(torch.tensor(x_plot, dtype=torch.float64, device=DEVICE)).cpu().numpy().ravel()

    bi_true = true_bi_numpy(x_plot.ravel())
    abs_err = np.abs(theta_pred - theta_ref)
    return {
        "theta_pred": theta_pred,
        "theta_sensor_pred": theta_sensor_pred,
        "theta_y_pred": theta_y_pred,
        "abs_err": abs_err,
        "x_plot": x_plot.ravel(),
        "bi_true": bi_true,
        "bi_pred": bi_pred,
        "theta_rel_l2": float(np.linalg.norm(theta_pred.ravel() - theta_ref.ravel()) / np.linalg.norm(theta_ref.ravel())),
        "theta_max_abs": float(abs_err.max()),
        "temp_rmse_C": float(np.sqrt(np.mean((to_temperature(theta_pred) - to_temperature(theta_ref)) ** 2))),
        "temp_max_abs_C": float(np.max(np.abs(to_temperature(theta_pred) - to_temperature(theta_ref)))),
        "bi_rel_l2": float(np.linalg.norm(bi_pred - bi_true) / np.linalg.norm(bi_true)),
    }


# =============================================================================
# Step 5. Output plots and summary
# =============================================================================


def trapz(values: np.ndarray, coordinates: np.ndarray) -> float:
    return float(np.trapz(values, coordinates))


def save_energy_balance_validation(ref: dict, result: dict) -> dict:
    energy_dir = OUTDIR / "energy_balance_validation"
    energy_dir.mkdir(parents=True, exist_ok=True)

    xs, ys = ref["xs"], ref["ys"]
    theta = result["theta_pred"]
    dx = float(xs[1] - xs[0])
    dy = float(ys[1] - ys[0])

    theta_y, theta_x = np.gradient(theta, dy, dx, edge_order=2)
    theta_yy = np.gradient(theta_y, dy, axis=0, edge_order=2)
    theta_xx = np.gradient(theta_x, dx, axis=1, edge_order=2)
    pde_residual = theta_xx + theta_yy + Q_STAR
    interior = np.s_[2:-2, 2:-2]

    bi_x = np.interp(xs, result["x_plot"], result["bi_pred"])
    q_top_cond = -theta_y[-1, :]
    q_bottom_cond = theta_y[0, :]
    q_left_cond = theta_x[:, 0]
    q_right_cond = -theta_x[:, -1]
    q_top_robin = bi_x * theta[-1, :]
    q_left_robin = BI_SIDE * theta[:, 0]
    q_right_robin = BI_SIDE * theta[:, -1]

    q_gen = Q_STAR
    q_top_cond_int = trapz(q_top_cond, xs)
    q_top_robin_int = trapz(q_top_robin, xs)
    q_bottom_cond_int = trapz(q_bottom_cond, xs)
    q_left_cond_int = trapz(q_left_cond, ys)
    q_right_cond_int = trapz(q_right_cond, ys)
    q_left_robin_int = trapz(q_left_robin, ys)
    q_right_robin_int = trapz(q_right_robin, ys)
    q_out_cond = q_top_cond_int + q_bottom_cond_int + q_left_cond_int + q_right_cond_int
    q_out_bc = q_top_robin_int + q_bottom_cond_int + q_left_robin_int + q_right_robin_int

    def imbalance_percent(q_out: float) -> float:
        return abs(q_out - q_gen) / abs(q_gen) * 100.0

    global_rows = [
        ("Q_gen_internal", q_gen, "Integral of uniform internal heat generation over unit domain"),
        ("Q_top_cond", q_top_cond_int, "Top outward conductive flux integral, int -theta_y dx"),
        ("Q_top_robin", q_top_robin_int, "Top Robin flux integral, int Bi_NN theta dx"),
        ("Q_bottom_cond", q_bottom_cond_int, "Bottom outward conductive flux integral, int theta_y dx"),
        ("Q_left_cond", q_left_cond_int, "Left outward conductive flux integral, int theta_x dy"),
        ("Q_right_cond", q_right_cond_int, "Right outward conductive flux integral, int -theta_x dy"),
        ("Q_left_robin", q_left_robin_int, "Left side Robin flux integral, int Bi_side theta dy"),
        ("Q_right_robin", q_right_robin_int, "Right side Robin flux integral, int Bi_side theta dy"),
        ("Q_out_total_conductive_flux", q_out_cond, "Total boundary flux from temperature-gradient fluxes"),
        ("Q_out_total_bc_flux", q_out_bc, "Total boundary flux from Robin BC fluxes"),
        ("global_imbalance_conductive_pct", imbalance_percent(q_out_cond), "Percent imbalance of conductive boundary flux"),
        ("global_imbalance_bc_pct", imbalance_percent(q_out_bc), "Percent imbalance of BC-based boundary flux"),
    ]

    with open(energy_dir / "global_energy_balance.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value_dimensionless", "description"])
        writer.writerows(global_rows)

    metrics = {
        "grid_nx": int(len(xs)),
        "grid_ny": int(len(ys)),
        "dx": dx,
        "dy": dy,
        "local_residual_metric_nx": int(theta[interior].shape[1]),
        "local_residual_metric_ny": int(theta[interior].shape[0]),
        "boundary_integration_points_top_bottom": int(len(xs)),
        "boundary_integration_points_left_right": int(len(ys)),
        "pde_residual_mean_abs": float(np.mean(np.abs(pde_residual[interior]))),
        "pde_residual_rmse": float(np.sqrt(np.mean(pde_residual[interior] ** 2))),
        "pde_residual_max_abs": float(np.max(np.abs(pde_residual[interior]))),
        "pde_residual_rel_rmse_vs_Q": float(np.sqrt(np.mean(pde_residual[interior] ** 2)) / abs(Q_STAR)),
        "Q_gen_internal": q_gen,
        "Q_out_total_conductive_flux": q_out_cond,
        "Q_out_total_bc_flux": q_out_bc,
        "global_imbalance_conductive_pct": imbalance_percent(q_out_cond),
        "global_imbalance_bc_pct": imbalance_percent(q_out_bc),
    }
    with open(energy_dir / "energy_balance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    p98 = float(np.percentile(np.abs(pde_residual[interior]), 98))
    levels = np.linspace(-p98, p98, 41)
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=180)
    cf = ax.contourf(xs, ys, pde_residual, levels=levels, cmap="coolwarm", extend="both")
    ax.set_title(r"Local PDE energy residual: $\theta_{xx}+\theta_{yy}+Q^*$")
    ax.set_xlabel(r"$x^*$")
    ax.set_ylabel(r"$y^*$")
    ax.set_aspect("equal")
    fig.colorbar(cf, ax=ax, label="residual")
    fig.savefig(energy_dir / "local_pde_residual_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    ax.bar(
        ["Generated heat", "Boundary flux\n(gradient)", "Boundary flux\n(Robin BC)"],
        [q_gen, q_out_cond, q_out_bc],
        color=["#111827", "#2563eb", "#7c3aed"],
    )
    ax.axhline(q_gen, color="k", ls="--", lw=1.2, label=r"$Q_{gen}$")
    ax.set_ylabel("integrated dimensionless heat rate")
    ax.set_title("Global energy balance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(energy_dir / "global_energy_balance_bar.png", bbox_inches="tight")
    plt.close(fig)

    with open(energy_dir / "ENERGY_BALANCE_VALIDATION.md", "w") as f:
        f.write("# Energy-Balance Validation\n\n")
        f.write("The final PINN temperature field was checked using the local PDE residual and the global heat balance.\n\n")
        f.write("## Grid Resolution\n\n")
        f.write(f"- Full field grid: {len(xs)} x {len(ys)}\n")
        f.write(f"- Grid spacing: dx = dy = {dx:.6g}\n")
        f.write(f"- Residual metrics: {metrics['local_residual_metric_nx']} x {metrics['local_residual_metric_ny']} interior points\n")
        f.write(f"- Boundary integration: {len(xs)} points on top/bottom and {len(ys)} points on left/right\n\n")
        f.write("## Local PDE Residual\n\n")
        f.write(r"$$R_{PDE}(x,y)=\theta_{xx}+\theta_{yy}+Q^*$$" + "\n\n")
        f.write(f"- mean |R_PDE|: {metrics['pde_residual_mean_abs']:.6e}\n")
        f.write(f"- RMSE R_PDE: {metrics['pde_residual_rmse']:.6e}\n")
        f.write(f"- RMSE / Q*: {100.0 * metrics['pde_residual_rel_rmse_vs_Q']:.4f}%\n\n")
        f.write("![Local PDE residual](local_pde_residual_heatmap.png)\n\n")
        f.write("## Global Energy Balance\n\n")
        f.write(r"$$Q_{gen}=\int_{\Omega}Q^*dA,\qquad Q_{out}=\int_{\partial\Omega}q_{out}ds$$" + "\n\n")
        f.write(f"- Q_gen: {q_gen:.8f}\n")
        f.write(f"- Q_out from conductive gradients: {q_out_cond:.8f}\n")
        f.write(f"- Q_out from Robin BC fluxes: {q_out_bc:.8f}\n")
        f.write(f"- conductive imbalance: {metrics['global_imbalance_conductive_pct']:.5f}%\n")
        f.write(f"- BC-based imbalance: {metrics['global_imbalance_bc_pct']:.5f}%\n\n")
        f.write("![Global energy balance](global_energy_balance_bar.png)\n")

    return metrics


def save_outputs(ref: dict, case: dict, result: dict, pinn, bi_model, history, elapsed: float) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    xs, ys = ref["xs"], ref["ys"]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)
    ax.plot(result["x_plot"], result["bi_true"], "k-", lw=2, label="true Bi")
    ax.plot(result["x_plot"], result["bi_pred"], "r--", lw=2, label="PINN auxiliary NN")
    ax.set_xlabel("x*")
    ax.set_ylabel("Bi_top(x*)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(OUTDIR / "bi_profile.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=180, constrained_layout=True)
    for ax, field, title in zip(axes, [ref["theta"], result["theta_pred"], result["abs_err"]], ["FDM theta", "PINN theta", "absolute error"]):
        im = ax.contourf(xs, ys, field, levels=40, cmap="inferno" if title != "absolute error" else "viridis")
        ax.scatter(case["sensor_xy"][:, 0], case["sensor_xy"][:, 1], s=20, facecolor="none", edgecolor="cyan")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")
        fig.colorbar(im, ax=ax)
    fig.savefig(OUTDIR / "field_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    for key in ["total", "pde", "top", "data", "theta_y_data"]:
        ax.semilogy([row["epoch"] for row in history], [row[key] for row in history], label=key)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(OUTDIR / "loss_history.png", bbox_inches="tight")
    plt.close(fig)

    center_idx = int(np.argmin(np.abs(result["x_plot"] - X_JET)))
    energy_metrics = save_energy_balance_validation(ref, result)
    summary = {
        "run_name": RUN_NAME,
        "reference_source": ref["source"],
        "seed": SEED,
        "sensor_x": SENSOR_X,
        "sensor_y": SENSOR_Y,
        "theta_y_method": "2-row finite difference",
        "weights": {"pde": W_PDE, "bottom": W_BOTTOM, "top": W_TOP, "side": W_SIDE, "data": W_DATA, "theta_y_data": W_THETA_Y_DATA},
        "metrics": {
            "theta_rel_l2": result["theta_rel_l2"],
            "theta_max_abs": result["theta_max_abs"],
            "temp_rmse_C": result["temp_rmse_C"],
            "temp_max_abs_C": result["temp_max_abs_C"],
            "bi_rel_l2": result["bi_rel_l2"],
            "bi_center_true": float(result["bi_true"][center_idx]),
            "bi_center_pred": float(result["bi_pred"][center_idx]),
            "h_center_true": float(result["bi_true"][center_idx] * K_S / L),
            "h_center_pred": float(result["bi_pred"][center_idx] * K_S / L),
            "elapsed_sec": elapsed,
        },
        "energy_balance_metrics": energy_metrics,
    }
    with open(OUTDIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    torch.save({"pinn": pinn.state_dict(), "bi_model": bi_model.state_dict(), "history": history}, OUTDIR / "phase3_auxiliary_nn_pinn.pt")
    np.savez_compressed(OUTDIR / "phase3_auxiliary_nn_fields.npz", xs=xs, ys=ys, theta_ref=ref["theta"], theta_pred=result["theta_pred"], abs_err=result["abs_err"], x_plot=result["x_plot"], bi_true=result["bi_true"], bi_pred=result["bi_pred"], sensor_xy=case["sensor_xy"])


def main() -> None:
    set_all_seeds(SEED)
    ref = build_reference()
    case = prepare_case(ref)
    pinn, bi_model, history, elapsed = train(case)
    result = evaluate(pinn, bi_model, ref, case)
    save_outputs(ref, case, result, pinn, bi_model, history, elapsed)
    print("finished")
    print(f"theta rel L2: {result['theta_rel_l2']:.4e}")
    print(f"Bi rel L2   : {result['bi_rel_l2']:.4e}")
    print(f"summary     : {OUTDIR / 'run_summary.json'}")
    print(f"energy check: {OUTDIR / 'energy_balance_validation'}")


if __name__ == "__main__":
    main()
