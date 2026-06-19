"""
Sequential ablation runner for the PyTorch inverse PINN baseline.

The original notebook is kept intact. This script reuses the same model,
losses, sensor pipeline, and output format, but wraps them as repeatable
experiments so multiple configurations can run unattended.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import time
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR


SEED = 2026
torch.set_default_dtype(torch.float64)


def resolve_device(requested="auto"):
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Check the Jupyter kernel, PyTorch CUDA build, NVIDIA driver, "
                "or CUDA_VISIBLE_DEVICES."
            )
        device = torch.device(requested)
        torch.empty(1, device=device)
        torch.cuda.set_device(device)
        return device
    raise ValueError(f"unknown device request: {requested}")


# Placeholder until main() resolves the requested device. This keeps --help and
# import-time suite inspection from touching CUDA before a real training run.
DEVICE = torch.device("cpu")

PROJECT_DIR = Path(__file__).resolve().parent
REF_PATH = PROJECT_DIR / "fdm_reference_output" / "fdm_reference_jet_impingement.npz"
OUTPUT_ROOT = PROJECT_DIR / "pinn_inverse_output" / "ablation_runs"
TODAY_RESULTS_DIR = PROJECT_DIR / "Submissions" / "Phase3_New_Results_20260609"

data = np.load(REF_PATH)
xs = data["xs"]
ys = data["ys"]
theta_ref = data["theta"]

Q_STAR = float(data["Q_STAR"])
BI_SIDE = float(data["BI_SIDE"])
L = float(data["L"])
D_JET = float(data["D_jet"])
K_S = float(data["k_s"])
T_B = float(data["T_b"])
T_INF = float(data["T_inf"])
R_OVER_D_DATA = data["R_over_D_data"]
H_DATA = data["h_data"]
X_JET = 0.5

USE_NOISY_SENSOR = False
NOISE_LEVEL_THETA = 0.02
NOISE_SEED = SEED + 100

H_MIN_GUESS = 500.0
H_MAX_GUESS = 2000.0
INIT_BI_MIN = H_MIN_GUESS * L / K_S
INIT_BI_MAX = H_MAX_GUESS * L / K_S

GAUSSIAN_PARAM_NAMES = ["theta1", "theta2", "theta3", "theta4", "theta5", "theta6"]

CURRENT_SENSOR_CONFIG = {
    "sensor_x": [0.02, 0.10, 0.18, 0.28, 0.38, 0.44, 0.50, 0.56, 0.62, 0.72, 0.82, 0.90, 0.98],
    "sensor_y": [0.99, 1.00],
}

SHOULDER_HEAVY_SENSOR_X = [0.02, 0.12, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.88, 0.98]

STAGE3_BASELINE_SENSOR_CONFIG = {
    "sensor_x": SHOULDER_HEAVY_SENSOR_X,
    "sensor_y": [0.99, 1.00],
}

FINAL_SENSOR_Y = [0.98, 1.00]

FINAL_SENSOR_CONFIG = {
    "sensor_x": SHOULDER_HEAVY_SENSOR_X,
    "sensor_y": FINAL_SENSOR_Y,
}

FINAL_LOCAL_LINEAR_4ROW_SENSOR_CONFIG = {
    "sensor_x": SHOULDER_HEAVY_SENSOR_X,
    "sensor_y": [0.94, 0.96, 0.98, 1.00],
    "theta_y_method": "local_linear",
}

DEFAULT_CONFIG = {
    "seed": SEED,
    "hidden_layers": 4,
    "neurons": 64,
    "activation": "tanh",
    "sampling": "lhs",
    "boundary_mode": "uniform",
    "n_domain": 3000,
    "n_boundary": 400,
    "sensor_x": CURRENT_SENSOR_CONFIG["sensor_x"],
    "sensor_y": [0.99, 1.00],
    "use_noisy_sensor": USE_NOISY_SENSOR,
    "noise_level_theta": NOISE_LEVEL_THETA,
    "noise_seed": NOISE_SEED,
    "theta_y_method": "fd_pairs",
}

DEFAULT_WEIGHTS = {
    "pde": 1.0,
    "bottom": 10.0,
    "top": 50.0,
    "side": 10.0,
    "data": 100.0,
    "theta_y_data": 1.0,
    "smooth": 0.0,
}

DEFAULT_BI_MODEL_CONFIG = {
    "type": "gaussian",
}

DEFAULT_OPTIMIZER_CONFIG = {
    "n_adam": 15000,
    "print_every": 500,
    "pinn_lr": 1e-3,
    "bi_lr": 1e-3,
    "adam_eta_min": 1e-5,
    "run_lbfgs": True,
    "lbfgs_steps": 600,
    "lbfgs_lr": 0.5,
    "lbfgs_max_iter": 20,
    "lbfgs_history_size": 50,
}

BEST_GAUSSIAN_AND_NODAL_OPTIMIZER_CONFIG = {
    "n_adam": 15000,
    "print_every": 500,
    "pinn_lr": 1e-3,
    "bi_lr": 1e-3,
    "adam_eta_min": 1e-5,
    "run_lbfgs": True,
    "lbfgs_steps": 1500,
    "lbfgs_lr": 0.5,
    "lbfgs_max_iter": 20,
    "lbfgs_history_size": 50,
}

BEST_AUXILIARY_NN_OPTIMIZER_CONFIG = {
    "n_adam": 15000,
    "print_every": 500,
    "pinn_lr": 5e-4,
    "bi_lr": 3e-3,
    "adam_eta_min": 1e-5,
    "run_lbfgs": True,
    "lbfgs_steps": 1500,
    "lbfgs_lr": 0.5,
    "lbfgs_max_iter": 20,
    "lbfgs_history_size": 50,
}

LEGACY_ABLATIONS = [
    {
        "name": "A_current_best_16sensor_theta_y1",
        "topic": "legacy_method",
        "description": "Current best 8x2 edge-anchored sensor placement with theta_y_data=1.",
    },
    {
        "name": "B_no_theta_y_loss",
        "topic": "legacy_method",
        "description": "Remove theta_y sensor loss to isolate ill-conditioning effect.",
        "weights": {"theta_y_data": 0.0},
    },
    {
        "name": "C_theta_y_weight_0p5",
        "topic": "legacy_method",
        "description": "Weaker theta_y constraint.",
        "weights": {"theta_y_data": 0.5},
    },
    {
        "name": "D_theta_y_weight_2p0",
        "topic": "legacy_method",
        "description": "Stronger theta_y constraint.",
        "weights": {"theta_y_data": 2.0},
    },
    {
        "name": "E_sobol_collocation",
        "topic": "legacy_method",
        "description": "Same sensors and weights, Sobol interior collocation.",
        "config": {"sampling": "sobol"},
    },
    {
        "name": "F_noisy_theta_2pct",
        "topic": "legacy_method",
        "description": "Add 2% Gaussian noise to theta sensor observations.",
        "config": {"use_noisy_sensor": True},
    },
    {
        "name": "G_dense48_2row_theta_y1",
        "topic": "legacy_method",
        "description": "48 sensors, two near-top rows, same theta_y_data weight.",
        "config": {
            "sensor_x": np.round(np.linspace(0.02, 0.98, 24), 6).tolist(),
            "sensor_y": [0.99, 1.00],
        },
    },
]

DEPTH_ABLATIONS = [
    {
        "name": "depth_2_hidden_64_tanh",
        "topic": "network_depth",
        "description": "Shallow network: faster, tests under-capacity.",
        "config": {"hidden_layers": 2, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "depth_3_hidden_64_tanh",
        "topic": "network_depth",
        "description": "Moderately shallow network.",
        "config": {"hidden_layers": 3, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "depth_4_hidden_64_tanh_baseline",
        "topic": "network_depth",
        "description": "Current baseline depth.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "depth_5_hidden_64_tanh",
        "topic": "network_depth",
        "description": "Slightly deeper network for richer theta gradients.",
        "config": {"hidden_layers": 5, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "depth_6_hidden_64_tanh",
        "topic": "network_depth",
        "description": "Deepest tested network; checks optimization difficulty vs capacity.",
        "config": {"hidden_layers": 6, "neurons": 64, "activation": "tanh"},
    },
]

NODE_COUNT_ABLATIONS = [
    {
        "name": "nodes_32_depth4_tanh",
        "topic": "node_count",
        "description": "Low-width network to test parameter efficiency.",
        "config": {"hidden_layers": 4, "neurons": 32, "activation": "tanh"},
    },
    {
        "name": "nodes_48_depth4_tanh",
        "topic": "node_count",
        "description": "Moderate-low width.",
        "config": {"hidden_layers": 4, "neurons": 48, "activation": "tanh"},
    },
    {
        "name": "nodes_64_depth4_tanh_baseline",
        "topic": "node_count",
        "description": "Current baseline width.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "nodes_96_depth4_tanh",
        "topic": "node_count",
        "description": "Higher width for better derivative fitting.",
        "config": {"hidden_layers": 4, "neurons": 96, "activation": "tanh"},
    },
    {
        "name": "nodes_128_depth4_tanh",
        "topic": "node_count",
        "description": "Largest width; checks whether capacity improves Bi recovery.",
        "config": {"hidden_layers": 4, "neurons": 128, "activation": "tanh"},
    },
]

ACTIVATION_ABLATIONS = [
    {
        "name": "activation_tanh_baseline",
        "topic": "activation_functions",
        "description": "Baseline smooth activation.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "tanh"},
    },
    {
        "name": "activation_sine",
        "topic": "activation_functions",
        "description": "Periodic smooth activation for high-frequency gradient response.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "sine"},
    },
    {
        "name": "activation_silu",
        "topic": "activation_functions",
        "description": "Smooth non-saturating activation.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "silu"},
    },
    {
        "name": "activation_gelu",
        "topic": "activation_functions",
        "description": "Smooth nonlinearity often stable in deep networks.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "gelu"},
    },
    {
        "name": "activation_softplus",
        "topic": "activation_functions",
        "description": "C-infinity activation; useful for second-derivative PDE residuals.",
        "config": {"hidden_layers": 4, "neurons": 64, "activation": "softplus"},
    },
]

SENSOR_PLACEMENT_ABLATIONS = [
    {
        "name": "sensor_16_edge_anchored_8x2_baseline",
        "topic": "sensor_placement",
        "description": "Current best 8 x 2 edge-anchored near-top sensors.",
        "config": {
            "sensor_x": [0.02, 0.14, 0.28, 0.38, 0.50, 0.62, 0.72, 0.98],
            "sensor_y": [0.99, 1.00],
        },
    },
    {
        "name": "sensor_16_uniform_8x2",
        "topic": "sensor_placement",
        "description": "Uniform x spacing over two near-top rows.",
        "config": {
            "sensor_x": np.round(np.linspace(0.02, 0.98, 8), 6).tolist(),
            "sensor_y": [0.99, 1.00],
        },
    },
    {
        "name": "sensor_16_center_dense_8x2",
        "topic": "sensor_placement",
        "description": "Center-dense x placement to resolve jet peak and secondary peaks.",
        "config": {
            "sensor_x": [0.08, 0.22, 0.34, 0.44, 0.50, 0.56, 0.66, 0.92],
            "sensor_y": [0.99, 1.00],
        },
    },
    {
        "name": "sensor_16_staggered_2row",
        "topic": "sensor_placement",
        "description": "Two staggered rows to avoid duplicating x positions at both y levels.",
        "config": {
            "sensor_x": [
                [0.02, 0.18, 0.34, 0.46, 0.54, 0.66, 0.82, 0.98],
                [0.10, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86, 0.94],
            ],
            "sensor_y": [0.99, 1.00],
        },
    },
    {
        "name": "sensor_48_dense_24x2",
        "topic": "sensor_placement",
        "description": "Dense 48-sensor near-top placement for near-ideal observability.",
        "config": {
            "sensor_x": np.round(np.linspace(0.02, 0.98, 24), 6).tolist(),
            "sensor_y": [0.99, 1.00],
        },
    },
]

DEFAULT_ABLATIONS = (
    DEPTH_ABLATIONS
    + NODE_COUNT_ABLATIONS
    + ACTIVATION_ABLATIONS
    + SENSOR_PLACEMENT_ABLATIONS
)

QUICK_ABLATIONS = [
    DEPTH_ABLATIONS[2],
    NODE_COUNT_ABLATIONS[2],
    ACTIVATION_ABLATIONS[0],
    SENSOR_PLACEMENT_ABLATIONS[0],
]

SMOKE_ABLATIONS = [
    {
        "name": "smoke_current_pipeline",
        "topic": "smoke",
        "description": "Tiny run for syntax and gradient checks.",
        "config": {"n_domain": 32, "n_boundary": 16},
        "optimizer": {"n_adam": 2, "print_every": 1, "run_lbfgs": False},
    }
]

BI_MODEL_BEST_WEIGHT_RETRAIN_ABLATIONS = [
    {
        "name": "method1_gaussian_best_weights_current_sensor",
        "topic": "bi_model_best_weights_current_sensor",
        "description": "Method 1 Gaussian Bi model using its best observed weight scale under the current 26-sensor layout.",
        "config": CURRENT_SENSOR_CONFIG,
        "bi_model": {"type": "gaussian"},
        "weights": {
            "pde": 1.0,
            "bottom": 10.0,
            "top": 50.0,
            "side": 10.0,
            "data": 100.0,
            "theta_y_data": 1.0,
            "smooth": 0.0,
        },
        "optimizer": BEST_GAUSSIAN_AND_NODAL_OPTIMIZER_CONFIG,
    },
    {
        "name": "method2_discrete_nodal_best_weights_current_sensor",
        "topic": "bi_model_best_weights_current_sensor",
        "description": "Method 2 discrete nodal Bi model using its best observed weight scale under the current 26-sensor layout.",
        "config": CURRENT_SENSOR_CONFIG,
        "bi_model": {
            "type": "discrete_nodal",
            "n_nodes": 21,
            "interpolation": "linear",
            "positive_transform": "softplus",
        },
        "weights": {
            "pde": 1.0,
            "bottom": 10.0,
            "top": 50.0,
            "side": 10.0,
            "data": 100.0,
            "theta_y_data": 1.0,
            "smooth": 1e-2,
        },
        "optimizer": BEST_GAUSSIAN_AND_NODAL_OPTIMIZER_CONFIG,
    },
    {
        "name": "method3_auxiliary_nn_best_weights_current_sensor",
        "topic": "bi_model_best_weights_current_sensor",
        "description": "Method 3 auxiliary-NN Bi model using its best observed weight scale under the current 26-sensor layout.",
        "config": CURRENT_SENSOR_CONFIG,
        "bi_model": {
            "type": "auxiliary_nn",
            "positive_transform": "softplus",
            "hidden_width": 32,
            "hidden_layers": 3,
            "activation": "tanh",
            "smooth_n_points": 101,
            "probe_n_points": 21,
        },
        "weights": {
            "pde": 1.0,
            "bottom": 10.0,
            "top": 50.0,
            "side": 10.0,
            "data": 100.0,
            "theta_y_data": 10.0,
            "smooth": 0.0,
        },
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
]


FINAL_AUXILIARY_NN_BI_MODEL_CONFIG = {
    "type": "auxiliary_nn",
    "positive_transform": "softplus",
    "hidden_width": 32,
    "hidden_layers": 3,
    "activation": "tanh",
    "smooth_n_points": 101,
    "probe_n_points": 21,
}

FINAL_AUXILIARY_NN_WEIGHTS = {
    "pde": 1.0,
    "bottom": 10.0,
    "top": 50.0,
    "side": 10.0,
    "data": 100.0,
    "theta_y_data": 10.0,
    "smooth": 0.0,
}

STAGE1_TWO_ROW_SENSOR_ABLATIONS = [
    {
        "name": "stage1_B0_current_26_13x2_aligned",
        "topic": "sensor_stage1_two_row",
        "description": "Stage 1 baseline: current Phase 2 13 x locations over two aligned near-top rows.",
        "config": CURRENT_SENSOR_CONFIG,
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": FINAL_AUXILIARY_NN_WEIGHTS,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
    {
        "name": "stage1_A_top_only_13_13x1_y1p00",
        "topic": "sensor_stage1_two_row",
        "description": "Stage 1 one-row comparison: keep the current x layout but observe only the top row y*=1.00.",
        "config": {
            "sensor_x": CURRENT_SENSOR_CONFIG["sensor_x"],
            "sensor_y": [1.00],
        },
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": FINAL_AUXILIARY_NN_WEIGHTS,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
    {
        "name": "stage1_B_near_top_only_13_13x1_y0p99",
        "topic": "sensor_stage1_two_row",
        "description": "Stage 1 one-row comparison: keep the current x layout but observe only the near-top interior row y*=0.99.",
        "config": {
            "sensor_x": CURRENT_SENSOR_CONFIG["sensor_x"],
            "sensor_y": [0.99],
        },
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": FINAL_AUXILIARY_NN_WEIGHTS,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
    {
        "name": "stage1_C_mid_row_only_13_13x1_y0p995",
        "topic": "sensor_stage1_two_row",
        "description": "Stage 1 one-row midpoint comparison: same x layout and sensor count as one-row cases at y*=0.995.",
        "config": {
            "sensor_x": CURRENT_SENSOR_CONFIG["sensor_x"],
            "sensor_y": [0.995],
        },
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": FINAL_AUXILIARY_NN_WEIGHTS,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
    {
        "name": "stage1_D_staggered_26_13x2_unaligned",
        "topic": "sensor_stage1_two_row",
        "description": "Stage 1 two-row comparison: same 26-sensor count but the two rows use staggered x locations instead of aligned pairs.",
        "config": {
            "sensor_x": [
                [0.02, 0.08, 0.16, 0.26, 0.38, 0.46, 0.50, 0.54, 0.62, 0.74, 0.84, 0.92, 0.98],
                [0.02, 0.10, 0.18, 0.28, 0.40, 0.48, 0.50, 0.52, 0.60, 0.72, 0.82, 0.90, 0.98],
            ],
            "sensor_y": [0.99, 1.00],
        },
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": FINAL_AUXILIARY_NN_WEIGHTS,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    },
]

SENSOR_ABLATION_ROADMAP = {
    "stage1_executable_suite": "sensor_stage1",
    "stage2_executable_suite": "sensor_stage2_fd",
    "stage2_alias_suite": "sensor_stage2",
    "stage3_executable_suite": "sensor_stage3_fd",
    "stage3_alias_suite": "sensor_stage3",
    "stage4_executable_suite": "sensor_stage4_fd",
    "stage4_alias_suite": "sensor_stage4",
    "stage5_executable_suite": "sensor_stage5_noise_fd",
    "stage5_alias_suite": "sensor_stage5_noise",
    "stage5_regularization_executable_suite": "sensor_stage5_regularization_fd",
    "stage5_regularization_alias_suite": "sensor_stage5_regularization",
    "stage1_stage2_executable_suite": "sensor_stage1_stage2",
    "stage1_goal": "Justify whether the final layout needs two vertically separated near-top sensor rows.",
    "stage2_record_only_x_placement": [
        {
            "name": "stage2_B0_current_central_dense_13x2",
            "status": "record_only",
            "goal": "Current Phase 2 x placement: center-including symmetric layout with central density and wider spacing toward outer regions.",
            "config": CURRENT_SENSOR_CONFIG,
        },
        {
            "name": "stage2_uniform_13x2",
            "status": "record_only",
            "goal": "Test whether uniform x spacing is sufficient for both center peak and background Bi recovery.",
            "config": {
                "sensor_x": [0.02, 0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.90, 0.98],
                "sensor_y": [0.99, 1.00],
            },
        },
        {
            "name": "stage2_center_heavy_13x2",
            "status": "record_only",
            "goal": "Increase x density around x*=0.5 to test jet-center peak identifiability.",
            "config": {
                "sensor_x": [0.02, 0.16, 0.30, 0.40, 0.45, 0.48, 0.50, 0.52, 0.55, 0.60, 0.70, 0.84, 0.98],
                "sensor_y": [0.99, 1.00],
            },
        },
        {
            "name": "stage2_shoulder_heavy_13x2",
            "status": "record_only",
            "goal": "Increase coverage near the shoulders around the jet peak while keeping aligned two-row FD pairs.",
            "config": {
                "sensor_x": SHOULDER_HEAVY_SENSOR_X,
                "sensor_y": [0.99, 1.00],
            },
        },
        {
            "name": "stage2_wide_coverage_13x2",
            "status": "record_only",
            "goal": "Balance center sensitivity with broader side-region coverage using aligned finite-difference pairs.",
            "config": {
                "sensor_x": [0.02, 0.08, 0.16, 0.26, 0.38, 0.46, 0.50, 0.54, 0.62, 0.74, 0.84, 0.92, 0.98],
                "sensor_y": [0.99, 1.00],
            },
        },
        {
            "name": "stage2_one_side_biased_13x2",
            "status": "record_only",
            "goal": "Deliberately bias sensors toward one side to show loss of symmetry/background reliability.",
            "config": {
                "sensor_x": [0.02, 0.06, 0.10, 0.16, 0.24, 0.34, 0.44, 0.50, 0.56, 0.62, 0.72, 0.82, 0.90],
                "sensor_y": [0.99, 1.00],
            },
        },
    ],
    "stage3_record_only_y_distance_and_gap": [
        {
            "name": "stage3_top_pair_gap_0p005",
            "status": "record_only",
            "goal": "Very tight two-row pair close to the top boundary using the Stage 2 shoulder-heavy x placement.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.995, 1.00]},
        },
        {
            "name": "stage3_top_pair_gap_0p010_shoulder_baseline",
            "status": "record_only",
            "goal": "Stage 3 baseline: Stage 2 shoulder-heavy x placement with the original near-top vertical pair.",
            "config": STAGE3_BASELINE_SENSOR_CONFIG,
        },
        {
            "name": "stage3_top_pair_gap_0p020",
            "status": "record_only",
            "goal": "Larger vertical gap while preserving a top-wall row.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.98, 1.00]},
        },
        {
            "name": "stage3_top_pair_gap_0p050",
            "status": "record_only",
            "goal": "Test whether the lower row becomes too far from the top Robin boundary.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.95, 1.00]},
        },
        {
            "name": "stage3_shifted_pair_y0p98_0p99",
            "status": "record_only",
            "goal": "Move both rows away from the top while preserving the same vertical gap.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.98, 0.99]},
        },
        {
            "name": "stage3_shifted_pair_y0p95_0p96",
            "status": "record_only",
            "goal": "Identify degradation when both rows are clearly below the top boundary.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.95, 0.96]},
        },
        {
            "name": "stage3_shifted_pair_y0p90_0p91",
            "status": "record_only",
            "goal": "Stress case: far-from-top observations should weaken inverse CHTC identifiability.",
            "config": {"sensor_x": SHOULDER_HEAVY_SENSOR_X, "sensor_y": [0.90, 0.91]},
        },
    ],
    "stage4_record_only_sensor_count": [
        {
            "name": "stage4_10sensors_5x2",
            "status": "record_only",
            "goal": "Very sparse final-layout subset with edge anchors, shoulder anchors, and center.",
            "config": {"sensor_x": [0.02, 0.34, 0.50, 0.66, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_14sensors_7x2",
            "status": "record_only",
            "goal": "Sparse final-layout subset preserving center and broad shoulder coverage.",
            "config": {"sensor_x": [0.02, 0.24, 0.40, 0.50, 0.60, 0.76, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_18sensors_9x2",
            "status": "record_only",
            "goal": "Moderate final-layout subset adding near-center and shoulder points.",
            "config": {"sensor_x": [0.02, 0.24, 0.34, 0.46, 0.50, 0.54, 0.66, 0.76, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_22sensors_11x2",
            "status": "record_only",
            "goal": "Near-baseline final-layout subset with only two shoulder-heavy x locations removed.",
            "config": {"sensor_x": [0.02, 0.12, 0.24, 0.34, 0.46, 0.50, 0.54, 0.66, 0.76, 0.88, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_26sensors_13x2_baseline",
            "status": "record_only",
            "goal": "Selected Stage 2-3 final layout: shoulder-heavy x placement with top-anchored y gap 0.020.",
            "config": FINAL_SENSOR_CONFIG,
        },
        {
            "name": "stage4_34sensors_17x2_dense",
            "status": "record_only",
            "goal": "Dense final-layout reference for checking whether 26 sensors are already near saturation.",
            "config": {
                "sensor_x": [0.02, 0.08, 0.12, 0.20, 0.28, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.72, 0.80, 0.88, 0.92, 0.98],
                "sensor_y": FINAL_SENSOR_Y,
            },
        },
    ],
    "stage4_record_only_nested_sensor_count_update": [
        {
            "name": "stage4_nested_10sensors_5x2",
            "status": "new_run",
            "goal": "Nested sparse subset: keep the future 14-sensor anchors and add only by sensor-count growth.",
            "config": {"sensor_x": [0.02, 0.24, 0.50, 0.76, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_nested_18sensors_9x2",
            "status": "new_run",
            "goal": "Nested 18-sensor subset: preserve the 14-sensor layout and add the first shoulder refinements.",
            "config": {"sensor_x": [0.02, 0.24, 0.34, 0.40, 0.50, 0.60, 0.66, 0.76, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_nested_22sensors_11x2",
            "status": "new_run",
            "goal": "Nested 22-sensor subset: preserve the 18-sensor layout and add near-center refinements.",
            "config": {"sensor_x": [0.02, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.98], "sensor_y": FINAL_SENSOR_Y},
        },
        {
            "name": "stage4_nested_34sensors_17x2_dense",
            "status": "new_run",
            "goal": "Nested dense oversampling check: preserve the selected 26-sensor baseline and add four side/edge refinements.",
            "config": {
                "sensor_x": [0.02, 0.08, 0.12, 0.20, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.80, 0.88, 0.92, 0.98],
                "sensor_y": FINAL_SENSOR_Y,
            },
        },
    ],
    "stage5_record_only_noise_level_sweep": [
        {
            "name": "stage5_noise_0p002_seed2126",
            "status": "record_only",
            "goal": "Sensor-noise robustness test with additive Gaussian noise sigma_theta=0.002 (0.2% of Delta T).",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
        },
        {
            "name": "stage5_noise_0p005_seed2126",
            "status": "record_only",
            "goal": "Sensor-noise robustness test with additive Gaussian noise sigma_theta=0.005 (0.5% of Delta T).",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.005, "noise_seed": 2126},
        },
        {
            "name": "stage5_noise_0p010_seed2126",
            "status": "record_only",
            "goal": "Sensor-noise robustness test with additive Gaussian noise sigma_theta=0.010 (1.0% of Delta T).",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.010, "noise_seed": 2126},
        },
        {
            "name": "stage5_noise_0p020_seed2126",
            "status": "record_only",
            "goal": "Phase 2 proposed thermocouple-accuracy stress test with additive Gaussian noise sigma_theta=0.020 (2.0% of Delta T).",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.020, "noise_seed": 2126},
        },
    ],
    "stage5_record_only_regularization_weighting": [
        {
            "name": "stage5_2_current_noise_0p002_dTy10_smooth0",
            "status": "record_only",
            "goal": "Current noisy-sensor failure case: sigma_theta=0.002 with the original derivative-data weight and no Bi smoothness penalty.",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
            "weights": {"theta_y_data": 10.0, "smooth": 0.0},
        },
        {
            "name": "stage5_2_noise_0p002_dTy1_smooth1em4",
            "status": "record_only",
            "goal": "Most promising noisy-sensor correction: reduce theta_y_data weight and activate weak Bi smoothness.",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
            "weights": {"theta_y_data": 1.0, "smooth": 1e-4},
        },
        {
            "name": "stage5_2_noise_0p002_dTy10_smooth1em4",
            "status": "record_only",
            "goal": "Smoothness-only test: keep the original derivative-data weight but add weak Bi smoothness.",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
            "weights": {"theta_y_data": 10.0, "smooth": 1e-4},
        },
        {
            "name": "stage5_2_noise_0p002_dTy1_smooth0",
            "status": "record_only",
            "goal": "Reduced-derivative-weight test without Bi smoothness regularization.",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
            "weights": {"theta_y_data": 1.0, "smooth": 0.0},
        },
        {
            "name": "stage5_2_noise_0p002_dTy1_smooth1em3",
            "status": "record_only",
            "goal": "Stronger Bi smoothness test under reduced theta_y_data weight.",
            "config": {**FINAL_SENSOR_CONFIG, "use_noisy_sensor": True, "noise_level_theta": 0.002, "noise_seed": 2126},
            "weights": {"theta_y_data": 1.0, "smooth": 1e-3},
        },
    ],
    "promotion_note": "Stage 2 x-placement entries are aligned two-row layouts for sensor-derived finite-difference theta_y loss. Stage 3 fixes the selected shoulder-heavy x placement and ablates only top proximity / vertical gap. Stage 4 fixes the selected x-y layout family and ablates sensor count. Stage 5 fixes the final selected layout and sweeps sensor noise level for robustness. Stage 5-2 fixes sigma_theta=0.002 and ablates noise-aware derivative weighting / Bi smoothness.",
}


def make_final_aux_sensor_ablation(record, topic):
    weights = deepcopy(FINAL_AUXILIARY_NN_WEIGHTS)
    weights.update(record.get("weights", {}))
    return {
        "name": record["name"],
        "topic": topic,
        "description": record["goal"],
        "config": deepcopy(record["config"]),
        "bi_model": FINAL_AUXILIARY_NN_BI_MODEL_CONFIG,
        "weights": weights,
        "optimizer": BEST_AUXILIARY_NN_OPTIMIZER_CONFIG,
    }


STAGE2_X_PLACEMENT_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage2_x_placement")
    for record in SENSOR_ABLATION_ROADMAP["stage2_record_only_x_placement"]
]

STAGE3_Y_PLACEMENT_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage3_y_placement")
    for record in SENSOR_ABLATION_ROADMAP["stage3_record_only_y_distance_and_gap"]
]

STAGE4_SENSOR_COUNT_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage4_sensor_count")
    for record in SENSOR_ABLATION_ROADMAP["stage4_record_only_sensor_count"]
]

STAGE4_NESTED_SENSOR_COUNT_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage4_nested_sensor_count")
    for record in SENSOR_ABLATION_ROADMAP["stage4_record_only_nested_sensor_count_update"]
]

STAGE5_NOISE_LEVEL_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage5_noise_level")
    for record in SENSOR_ABLATION_ROADMAP["stage5_record_only_noise_level_sweep"]
]

STAGE5_NOISE_LEVEL_CONTINUATION_ABLATIONS = [
    spec
    for spec in STAGE5_NOISE_LEVEL_SENSOR_ABLATIONS
    if spec["name"] in {"stage5_noise_0p010_seed2126", "stage5_noise_0p020_seed2126"}
]

STAGE5_REGULARIZATION_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(record, "sensor_stage5_regularization_weighting")
    for record in SENSOR_ABLATION_ROADMAP["stage5_record_only_regularization_weighting"]
]

STAGE5_LOCAL_LINEAR_4ROW_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(
        {
            "name": f"stage5_3_local_linear_4row_noise_{str(noise_level).replace('.', 'p')}",
            "goal": (
                "Noisy-sensor robustness test using 4-row local linear least-squares "
                "theta_y labels instead of two-point finite differences."
            ),
            "config": {
                **FINAL_LOCAL_LINEAR_4ROW_SENSOR_CONFIG,
                "use_noisy_sensor": True,
                "noise_level_theta": noise_level,
                "noise_seed": 2126,
            },
            "weights": {"theta_y_data": 1.0, "smooth": 0.0},
        },
        "sensor_stage5_local_linear_4row",
    )
    for noise_level in [0.002, 0.010, 0.020]
]

STAGE5_LOCAL_LINEAR_4ROW_TUNE_0P002_SENSOR_ABLATIONS = [
    make_final_aux_sensor_ablation(
        {
            "name": f"stage5_4_local_linear_4row_noise_0p002_dTy{str(theta_y_weight).replace('.', 'p')}_smooth0",
            "goal": (
                "Noise sigma_theta=0.002 recovery test using 4-row local-linear theta_y labels "
                "with stronger derivative-data weighting to restore the center/shoulder peaks."
            ),
            "config": {
                **FINAL_LOCAL_LINEAR_4ROW_SENSOR_CONFIG,
                "use_noisy_sensor": True,
                "noise_level_theta": 0.002,
                "noise_seed": 2126,
            },
            "weights": {"theta_y_data": theta_y_weight, "smooth": 0.0},
        },
        "sensor_stage5_local_linear_4row_tune_0p002",
    )
    for theta_y_weight in [2.0, 3.0, 5.0]
]

STAGE5_NOISE_SEED_ABLATIONS = [
    make_final_aux_sensor_ablation(
        {
            "name": f"stage5_noise_seed_noise0p002_seed{noise_seed}_dTy1_smooth0",
            "goal": (
                "Sensor-noise realization sensitivity test with fixed two-row placement, "
                "sigma_theta=0.002, theta_y_data weight 1, and no Bi smoothness."
            ),
            "config": {
                **FINAL_SENSOR_CONFIG,
                "seed": 2026,
                "use_noisy_sensor": True,
                "noise_level_theta": 0.002,
                "noise_seed": noise_seed,
                "theta_y_method": "fd_pairs",
            },
            "weights": {"theta_y_data": 1.0, "smooth": 0.0},
        },
        "sensor_stage5_noise_seed",
    )
    for noise_seed in [2126, 2127, 2128]
]

FINAL_SEED_ROBUSTNESS_COLLOCATION_SEEDS = [2026, 2027, 2028]
FINAL_SEED_ROBUSTNESS_NOISE_SEEDS = [2127, 2128, 2129]

FINAL_SEED_ROBUSTNESS_ABLATIONS = []

for collocation_seed in FINAL_SEED_ROBUSTNESS_COLLOCATION_SEEDS:
    FINAL_SEED_ROBUSTNESS_ABLATIONS.append(
        make_final_aux_sensor_ablation(
            {
                "name": f"final_clean_13x2_colloc_seed{collocation_seed}",
                "goal": (
                    "Final clean selected 26-sensor two-row layout repeated with a different "
                    "collocation/boundary sampling seed."
                ),
                "config": {
                    **FINAL_SENSOR_CONFIG,
                    "seed": collocation_seed,
                    "use_noisy_sensor": False,
                    "theta_y_method": "fd_pairs",
                },
            },
            "final_seed_robustness_clean_13x2",
        )
    )

for collocation_seed in FINAL_SEED_ROBUSTNESS_COLLOCATION_SEEDS:
    FINAL_SEED_ROBUSTNESS_ABLATIONS.append(
        make_final_aux_sensor_ablation(
            {
                "name": f"final_noisy_4row_local_linear_colloc_seed{collocation_seed}_noise_seed2126",
                "goal": (
                    "Final noisy four-row local-linear layout repeated with a different "
                    "collocation/boundary sampling seed and fixed sensor-noise realization."
                ),
                "config": {
                    **FINAL_LOCAL_LINEAR_4ROW_SENSOR_CONFIG,
                    "seed": collocation_seed,
                    "use_noisy_sensor": True,
                    "noise_level_theta": 0.002,
                    "noise_seed": 2126,
                },
                "weights": {"theta_y_data": 2.0, "smooth": 0.0},
            },
            "final_seed_robustness_noisy_4row",
        )
    )

for noise_seed in FINAL_SEED_ROBUSTNESS_NOISE_SEEDS:
    FINAL_SEED_ROBUSTNESS_ABLATIONS.append(
        make_final_aux_sensor_ablation(
            {
                "name": f"final_noisy_4row_local_linear_colloc_seed2026_noise_seed{noise_seed}",
                "goal": (
                    "Final noisy four-row local-linear layout repeated with a different "
                    "sensor-noise realization and fixed collocation/boundary sampling seed."
                ),
                "config": {
                    **FINAL_LOCAL_LINEAR_4ROW_SENSOR_CONFIG,
                    "seed": 2026,
                    "use_noisy_sensor": True,
                    "noise_level_theta": 0.002,
                    "noise_seed": noise_seed,
                },
                "weights": {"theta_y_data": 2.0, "smooth": 0.0},
            },
            "final_seed_robustness_noisy_4row",
        )
    )

STAGE2_REMAINING_SENSOR_ABLATIONS = []

STAGE1_STAGE2_SENSOR_ABLATIONS = (
    STAGE1_TWO_ROW_SENSOR_ABLATIONS + STAGE2_X_PLACEMENT_SENSOR_ABLATIONS
)


def make_capacity_dependency_ablation(method_spec, capacity_name, hidden_layers, neurons):
    spec = deepcopy(method_spec)
    spec["name"] = f"{method_spec['name']}_{capacity_name}"
    spec["topic"] = "bi_model_capacity_dependency_current_sensor"
    spec["description"] = (
        f"{method_spec['description']} PINN capacity ablation: "
        f"{hidden_layers} hidden layers x {neurons} neurons."
    )
    config = deepcopy(spec.get("config", {}))
    config.update(
        {
            "hidden_layers": hidden_layers,
            "neurons": neurons,
            "activation": "tanh",
        }
    )
    spec["config"] = config
    return spec


CAPACITY_DEPENDENCY_SETTINGS = [
    ("small_depth3_width32", 3, 32),
    ("baseline_depth4_width64", 4, 64),
    ("large_depth5_width96", 5, 96),
]


BI_MODEL_CAPACITY_DEPENDENCY_ABLATIONS = [
    make_capacity_dependency_ablation(method_spec, capacity_name, hidden_layers, neurons)
    for method_spec in BI_MODEL_BEST_WEIGHT_RETRAIN_ABLATIONS
    for capacity_name, hidden_layers, neurons in CAPACITY_DEPENDENCY_SETTINGS
]


def deep_update(base, updates):
    out = deepcopy(base)
    for key, value in updates.items():
        out[key] = value
    return out


def slugify(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_sensor_points(sensor_x, sensor_y):
    if isinstance(sensor_x[0], (list, tuple, np.ndarray)):
        if len(sensor_x) != len(sensor_y):
            raise ValueError("row-wise sensor_x must have the same length as sensor_y.")
        return np.array([[x, y] for y, xs_row in zip(sensor_y, sensor_x) for x in xs_row], dtype=float)
    return np.array([[x, y] for y in sensor_y for x in sensor_x], dtype=float)


def sample_bilinear(xs_array, ys_array, field, points):
    values = []
    for x, y in np.asarray(points):
        i = np.clip(np.searchsorted(xs_array, x) - 1, 0, len(xs_array) - 2)
        j = np.clip(np.searchsorted(ys_array, y) - 1, 0, len(ys_array) - 2)
        tx = (x - xs_array[i]) / (xs_array[i + 1] - xs_array[i])
        ty = (y - ys_array[j]) / (ys_array[j + 1] - ys_array[j])
        values.append(
            (1 - tx) * (1 - ty) * field[j, i]
            + tx * (1 - ty) * field[j, i + 1]
            + (1 - tx) * ty * field[j + 1, i]
            + tx * ty * field[j + 1, i + 1]
        )
    return np.asarray(values).reshape(-1, 1)


def sample_theta_y_reference(xs_array, ys_array, field, points):
    dy = float(np.min(np.diff(ys_array)))
    points = np.asarray(points, dtype=float)
    p_plus = points.copy()
    p_minus = points.copy()
    p_plus[:, 1] = np.minimum(points[:, 1] + dy, ys_array.max())
    p_minus[:, 1] = np.maximum(points[:, 1] - dy, ys_array.min())
    theta_plus = sample_bilinear(xs_array, ys_array, field, p_plus)
    theta_minus = sample_bilinear(xs_array, ys_array, field, p_minus)
    denom = (p_plus[:, 1] - p_minus[:, 1]).reshape(-1, 1)
    return (theta_plus - theta_minus) / denom


def build_sensor_theta_y_fd_data(points, theta_values, x_round_decimals=12):
    """Build theta_y labels only from vertically paired sensor temperatures."""
    points = np.asarray(points, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1, 1)

    by_x = {}
    for idx, (x, _y) in enumerate(points):
        by_x.setdefault(round(float(x), x_round_decimals), []).append(idx)

    lower_points = []
    upper_points = []
    pair_points = []
    theta_y_values = []

    for indices in by_x.values():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: points[i, 1])
        for lower_idx, upper_idx in zip(ordered[:-1], ordered[1:]):
            y_lower = float(points[lower_idx, 1])
            y_upper = float(points[upper_idx, 1])
            dy_pair = y_upper - y_lower
            if dy_pair <= 0.0:
                continue
            lower_points.append(points[lower_idx])
            upper_points.append(points[upper_idx])
            pair_points.append([points[lower_idx, 0], 0.5 * (y_lower + y_upper)])
            theta_y_values.append((theta_values[upper_idx, 0] - theta_values[lower_idx, 0]) / dy_pair)

    if not theta_y_values:
        empty_points = np.empty((0, 2), dtype=float)
        empty_values = np.empty((0, 1), dtype=float)
        return empty_points, empty_points.copy(), empty_points.copy(), empty_values

    return (
        np.asarray(lower_points, dtype=float),
        np.asarray(upper_points, dtype=float),
        np.asarray(pair_points, dtype=float),
        np.asarray(theta_y_values, dtype=float).reshape(-1, 1),
    )


def build_sensor_theta_y_local_linear_data(points, theta_values, x_round_decimals=12):
    """Build theta_y labels from a local least-squares linear fit in y."""
    points = np.asarray(points, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1, 1)

    by_x = {}
    for idx, (x, _y) in enumerate(points):
        by_x.setdefault(round(float(x), x_round_decimals), []).append(idx)

    lower_points = []
    upper_points = []
    pair_points = []
    fit_points = []
    fit_weight_rows = []
    theta_y_values = []

    for x_key, indices in by_x.items():
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: points[i, 1])
        group_points = points[ordered]
        y = group_points[:, 1]
        y_top = float(y.max())
        z = y - y_top
        design = np.column_stack([np.ones_like(z), z])
        slope_weights = np.linalg.pinv(design)[1]

        row = np.zeros(len(fit_points) + len(group_points), dtype=float)
        if fit_weight_rows:
            fit_weight_rows = [
                np.pad(existing, (0, len(group_points)), mode="constant")
                for existing in fit_weight_rows
            ]
        row[len(fit_points) : len(fit_points) + len(group_points)] = slope_weights
        fit_weight_rows.append(row)
        fit_points.extend(group_points.tolist())

        lower_points.append(group_points[0])
        upper_points.append(group_points[-1])
        pair_points.append([float(x_key), y_top])
        theta_y_values.append(float(slope_weights @ theta_values[ordered, 0]))

    if not theta_y_values:
        empty_points = np.empty((0, 2), dtype=float)
        return empty_points, empty_points.copy(), empty_points.copy(), np.empty((0, 1), dtype=float), empty_points.copy(), np.empty((0, 0), dtype=float)

    return (
        np.asarray(lower_points, dtype=float),
        np.asarray(upper_points, dtype=float),
        np.asarray(pair_points, dtype=float),
        np.asarray(theta_y_values, dtype=float).reshape(-1, 1),
        np.asarray(fit_points, dtype=float),
        np.vstack(fit_weight_rows).astype(float),
    )


def build_sensor_theta_y_data(points, theta_values, method="fd_pairs"):
    method_key = str(method).lower()
    if method_key in {"fd", "fd_pair", "fd_pairs", "two_row_fd"}:
        lower_xy, upper_xy, pair_xy, theta_y_ref = build_sensor_theta_y_fd_data(points, theta_values)
        return lower_xy, upper_xy, pair_xy, theta_y_ref, np.empty((0, 2), dtype=float), np.empty((0, 0), dtype=float)
    if method_key in {"local_linear", "linear_fit", "least_squares_linear", "ls_linear"}:
        return build_sensor_theta_y_local_linear_data(points, theta_values)
    raise ValueError(f"unknown theta_y_method: {method}")


def sensor_apparent_bi_points(points, theta_values, method="fd_pairs", x_round_decimals=12):
    """Return apparent Bi inferred directly from sensor temperature values."""
    points = np.asarray(points, dtype=float)
    theta_values = np.asarray(theta_values, dtype=float).reshape(-1)
    method_key = str(method).lower()

    by_x = {}
    for idx, (x, _y) in enumerate(points):
        by_x.setdefault(round(float(x), x_round_decimals), []).append(idx)

    rows = []
    for x_key, indices in sorted(by_x.items(), key=lambda item: item[0]):
        if len(indices) < 2:
            continue
        ordered = sorted(indices, key=lambda i: points[i, 1])
        group_points = points[ordered]
        y = group_points[:, 1]
        values = theta_values[ordered]

        if method_key in {"local_linear", "linear_fit", "least_squares_linear", "ls_linear"}:
            y_top = float(y.max())
            z = y - y_top
            design = np.column_stack([np.ones_like(z), z])
            theta_top, theta_y = np.linalg.pinv(design) @ values
        elif method_key in {"fd", "fd_pair", "fd_pairs", "two_row_fd"}:
            y_top = float(y[-1])
            dy_pair = float(y[-1] - y[0])
            if abs(dy_pair) < 1e-12:
                continue
            theta_top = float(values[-1])
            theta_y = float((values[-1] - values[0]) / dy_pair)
        else:
            raise ValueError(f"unknown theta_y_method: {method}")

        rows.append(
            {
                "x": float(x_key),
                "y_top": float(y_top),
                "theta_top": float(theta_top),
                "theta_y": float(theta_y),
                "bi_apparent": float(-theta_y / max(float(theta_top), 1e-12)),
            }
        )
    return rows


def save_sensor_apparent_bi_overlay(
    run_dir,
    sensor_xy,
    theta_sensor_clean,
    theta_sensor_train,
    theta_sensor_pred,
    theta_y_method,
    x_plot,
    bi_true,
    bi_pred,
):
    clean_rows = sensor_apparent_bi_points(sensor_xy, theta_sensor_clean, theta_y_method)
    train_rows = sensor_apparent_bi_points(sensor_xy, theta_sensor_train, theta_y_method)
    pred_rows = sensor_apparent_bi_points(sensor_xy, theta_sensor_pred, theta_y_method)
    if not train_rows:
        return

    clean_by_x = {round(row["x"], 12): row for row in clean_rows}
    pred_by_x = {round(row["x"], 12): row for row in pred_rows}
    x_plot_flat = np.asarray(x_plot, dtype=float).ravel()
    bi_true_flat = np.asarray(bi_true, dtype=float).ravel()
    bi_pred_flat = np.asarray(bi_pred, dtype=float).ravel()

    rows = []
    for train_row in train_rows:
        x_key = round(train_row["x"], 12)
        clean_row = clean_by_x.get(x_key, {})
        pred_row = pred_by_x.get(x_key, {})
        bi_true_i = float(np.interp(train_row["x"], x_plot_flat, bi_true_flat))
        bi_pred_i = float(np.interp(train_row["x"], x_plot_flat, bi_pred_flat))
        rows.append(
            {
                "x": train_row["x"],
                "y_top": train_row["y_top"],
                "theta_top_clean": clean_row.get("theta_top", np.nan),
                "theta_top_train": train_row["theta_top"],
                "theta_top_pred": pred_row.get("theta_top", np.nan),
                "theta_y_clean": clean_row.get("theta_y", np.nan),
                "theta_y_train": train_row["theta_y"],
                "theta_y_pred": pred_row.get("theta_y", np.nan),
                "bi_app_clean": clean_row.get("bi_apparent", np.nan),
                "bi_app_noisy": train_row["bi_apparent"],
                "bi_app_model_theta": pred_row.get("bi_apparent", np.nan),
                "bi_true_at_sensor": bi_true_i,
                "bi_pinn_at_sensor": bi_pred_i,
                "noisy_app_error": train_row["bi_apparent"] - bi_true_i,
                "pinn_error": bi_pred_i - bi_true_i,
            }
        )

    with open(run_dir / "noisy_sensor_apparent_bi_overlay_points.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=170)
    ax.plot(x_plot_flat, bi_true_flat, color="black", lw=2.2, label="True Bi")
    ax.plot(x_plot_flat, bi_pred_flat, color="#4e79a7", lw=2.0, label="PINN recovered Bi")
    ax.scatter(
        [row["x"] for row in rows],
        [row["bi_app_noisy"] for row in rows],
        s=50,
        color="#f28e2b",
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
        label="Noisy sensor apparent Bi",
    )
    ax.scatter(
        [row["x"] for row in rows],
        [row["bi_app_clean"] for row in rows],
        s=30,
        color="#9aa0a6",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="Clean sensor apparent Bi",
    )
    ax.scatter(
        [row["x"] for row in rows],
        [row["bi_app_model_theta"] for row in rows],
        s=24,
        color="#59a14f",
        edgecolor="white",
        linewidth=0.5,
        zorder=4,
        label="PINN theta apparent Bi",
    )
    ax.set_xlabel("x*")
    ax.set_ylabel("Bi_top(x*)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(run_dir / "noisy_sensor_apparent_bi_overlay.png", bbox_inches="tight")
    plt.close(fig)


def gaussian_sensor_noise(shape, noise_level_theta, seed):
    raw = np.random.default_rng(seed).normal(size=shape)
    raw = raw - raw.mean()
    return noise_level_theta * raw / max(raw.std(), 1e-12)


def sensor_layout_summary_title(name, sensor_xy):
    sensor_xy = np.asarray(sensor_xy, dtype=float)
    unique_x = np.unique(np.round(sensor_xy[:, 0], 12))
    unique_y = np.unique(np.round(sensor_xy[:, 1], 12))
    layout_size = f"{len(unique_x)}x{len(unique_y)} sensors ({len(sensor_xy)} total)"
    name_key = str(name).lower()

    if "stage4_10sensors" in name_key:
        summary = "Very sparse final-layout subset"
    elif "stage4_14sensors" in name_key:
        summary = "Sparse final-layout subset"
    elif "stage4_18sensors" in name_key:
        summary = "Moderate final-layout subset"
    elif "stage4_22sensors" in name_key:
        summary = "Near-baseline final-layout subset"
    elif "stage4_26sensors" in name_key:
        summary = "Selected final-layout baseline"
    elif "stage4_34sensors" in name_key:
        summary = "Dense final-layout reference"
    elif "stage3_top_pair_gap_0p005" in name_key:
        summary = "Top-anchored pair with 0.005 vertical gap"
    elif "stage3_top_pair_gap_0p010" in name_key:
        summary = "Top-anchored baseline pair with 0.010 vertical gap"
    elif "stage3_top_pair_gap_0p020" in name_key:
        summary = "Top-anchored pair with 0.020 vertical gap"
    elif "stage3_top_pair_gap_0p050" in name_key:
        summary = "Top-anchored pair with 0.050 vertical gap"
    elif "stage3_shifted_pair_y0p98_0p99" in name_key:
        summary = "Shifted pair at y*=0.98-0.99"
    elif "stage3_shifted_pair_y0p95_0p96" in name_key:
        summary = "Shifted pair at y*=0.95-0.96"
    elif "stage3_shifted_pair_y0p90_0p91" in name_key:
        summary = "Far-from-top shifted pair at y*=0.90-0.91"
    elif "uniform" in name_key:
        summary = "Uniform symmetric x coverage"
    elif "center_heavy" in name_key or "center-heavy" in name_key:
        summary = "Symmetric center-focused coverage near the jet"
    elif "shoulder_heavy" in name_key or "shoulder-heavy" in name_key:
        summary = "Symmetric shoulder-focused coverage around the jet"
    elif "wide_coverage" in name_key or "wide-coverage" in name_key:
        summary = "Symmetric wide coverage with edge anchors"
    elif "one_side" in name_key or "one-side" in name_key:
        summary = "One-sided biased coverage stress test"
    elif "staggered" in name_key:
        summary = "Staggered two-row coverage"
    elif "top_only" in name_key:
        summary = "Single top-row temperature sensors"
    elif "near_top_only" in name_key:
        summary = "Single near-top interior sensor row"
    elif "mid_row" in name_key:
        summary = "Single midpoint sensor row"
    elif "current" in name_key or "baseline" in name_key:
        summary = "Baseline symmetric center-dense coverage"
    else:
        summary = "Near-top sensor coverage"

    return f"{summary} - {layout_size}"


def save_sensor_layout_diagram(run_dir, sensor_xy, theta_y_pair_xy=None, title=None):
    sensor_xy = np.asarray(sensor_xy, dtype=float)
    theta_y_pair_xy = np.asarray([] if theta_y_pair_xy is None else theta_y_pair_xy, dtype=float)

    unique_y = np.unique(np.round(sensor_xy[:, 1], 12))
    y_min = max(0.0, float(sensor_xy[:, 1].min()) - 0.006)
    y_max = min(1.005, max(1.0, float(sensor_xy[:, 1].max())) + 0.004)

    fig, ax = plt.subplots(figsize=(8.0, 2.4), dpi=180)
    ax.axhline(1.0, color="black", lw=1.5, label="top boundary")
    ax.axhline(0.99, color="0.75", lw=0.8, ls="--")
    ax.axvline(0.0, color="black", lw=1.0)
    ax.axvline(1.0, color="black", lw=1.0)

    for y_value in unique_y:
        mask = np.isclose(sensor_xy[:, 1], y_value)
        color = "tab:blue" if y_value >= 0.995 else "tab:orange"
        ax.scatter(
            sensor_xy[mask, 0],
            sensor_xy[mask, 1],
            s=64,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label=f"sensor y*={y_value:.3f}",
        )

    if theta_y_pair_xy.size:
        for x_value, y_mid in theta_y_pair_xy:
            ax.plot([x_value, x_value], [sensor_xy[:, 1].min(), sensor_xy[:, 1].max()], color="0.35", lw=0.7, alpha=0.55)
        ax.scatter(
            theta_y_pair_xy[:, 0],
            theta_y_pair_xy[:, 1],
            s=18,
            marker="x",
            color="black",
            zorder=4,
            label="FD pair midpoint",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x*")
    ax.set_ylabel("y*")
    ax.set_title(title or "near-top sensor layout")
    ax.grid(True, axis="x", alpha=0.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.32)
    fig.savefig(run_dir / "sensor_layout_top_boundary.png", bbox_inches="tight")
    plt.close(fig)


bi_interp = PchipInterpolator(R_OVER_D_DATA, H_DATA * L / K_S, extrapolate=True)


def true_bi_numpy(x_star):
    r_over_d = np.abs(np.asarray(x_star) - X_JET) * L / D_JET
    r_over_d = np.clip(r_over_d, R_OVER_D_DATA.min(), R_OVER_D_DATA.max())
    return np.maximum(bi_interp(r_over_d), 1e-12)


def sample_interior_lhs(n, seed=0, device=None, requires_grad=True):
    device = DEVICE if device is None else device
    rng = np.random.default_rng(seed)
    edges = np.linspace(0.0, 1.0, n + 1)
    x = rng.uniform(edges[:-1], edges[1:])
    y = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(x)
    rng.shuffle(y)
    xy = np.stack([x, y], axis=1)
    out = torch.tensor(xy, dtype=torch.float64, device=device)
    return out.requires_grad_(requires_grad)


def sample_interior_sobol(n, seed=0, device=None, requires_grad=True):
    device = DEVICE if device is None else device
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    out = engine.draw(n).to(dtype=torch.float64, device=device)
    return out.requires_grad_(requires_grad)


def edge_points(edge, n, mode="uniform", seed=0, device=None, requires_grad=False):
    device = DEVICE if device is None else device
    if mode == "uniform":
        s = torch.linspace(0.0, 1.0, n, dtype=torch.float64, device=device).reshape(-1, 1)
    elif mode == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        s = torch.rand(n, 1, generator=gen, dtype=torch.float64).to(device)
    else:
        raise ValueError(f"unknown edge sampling mode: {mode}")

    z = torch.zeros_like(s)
    o = torch.ones_like(s)
    if edge == "bottom":
        xy = torch.cat([s, z], dim=1)
    elif edge == "top":
        xy = torch.cat([s, o], dim=1)
    elif edge == "left":
        xy = torch.cat([z, s], dim=1)
    elif edge == "right":
        xy = torch.cat([o, s], dim=1)
    else:
        raise ValueError(f"unknown edge: {edge}")
    return xy.requires_grad_(requires_grad)


def make_training_points(n_domain, n_boundary, sampling="lhs", boundary_mode="uniform", seed=0):
    if sampling == "lhs":
        xy_f = sample_interior_lhs(n_domain, seed=seed, requires_grad=True)
    elif sampling == "sobol":
        xy_f = sample_interior_sobol(n_domain, seed=seed, requires_grad=True)
    else:
        raise ValueError("sampling must be either 'lhs' or 'sobol'.")
    boundary = {
        "bottom": edge_points("bottom", n_boundary, mode=boundary_mode, seed=seed + 1),
        "top": edge_points("top", n_boundary, mode=boundary_mode, seed=seed + 2),
        "left": edge_points("left", n_boundary, mode=boundary_mode, seed=seed + 3),
        "right": edge_points("right", n_boundary, mode=boundary_mode, seed=seed + 4),
    }
    return xy_f, boundary


def mse(x):
    return torch.mean(x ** 2)


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def make_activation(name):
    key = str(name).lower()
    if key == "tanh":
        return nn.Tanh
    if key == "sine":
        return Sine
    if key == "silu":
        return nn.SiLU
    if key == "gelu":
        return nn.GELU
    if key == "softplus":
        return nn.Softplus
    raise ValueError(f"unknown activation: {name}")


class PINN2D(nn.Module):
    def __init__(self, hidden_layers=4, neurons=64, activation="tanh"):
        super().__init__()
        hidden_layers = int(hidden_layers)
        neurons = int(neurons)
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")

        activation_cls = make_activation(activation)
        layers = [nn.Linear(2, neurons), activation_cls()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(neurons, neurons), activation_cls()]
        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.activation = str(activation).lower()

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xy):
        return self.net(xy)


class GaussianBiTop(nn.Module):
    def __init__(self, init_bi_max, init_bi_min):
        super().__init__()
        self.softplus = nn.Softplus()
        init_background = max(float(init_bi_min), 1e-4)
        init_primary_amp = max(float(init_bi_max) - init_background, 1e-4)
        init_ring_amp = max(0.35 * init_primary_amp, 1e-4)
        init_delta = 0.20
        init = torch.tensor(
            [
                self._inv_softplus(init_primary_amp),
                self._inv_softplus(0.08),
                self._inv_softplus(init_ring_amp),
                self._logit(init_delta / 0.5),
                self._inv_softplus(0.15),
                self._inv_softplus(init_background),
            ],
            dtype=torch.float64,
        )
        self.raw = nn.Parameter(init)

    @staticmethod
    def _inv_softplus(y):
        return math.log(math.exp(float(y)) - 1.0)

    @staticmethod
    def _logit(p):
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        return math.log(p / (1.0 - p))

    def physical_params(self):
        theta1 = self.softplus(self.raw[0])
        theta2 = self.softplus(self.raw[1]) + 1e-4
        theta3 = self.softplus(self.raw[2])
        theta4 = 0.5 * torch.sigmoid(self.raw[3])
        theta5 = self.softplus(self.raw[4]) + 1e-4
        theta6 = self.softplus(self.raw[5]) + 1e-12
        return theta1, theta2, theta3, theta4, theta5, theta6

    def forward(self, x_star):
        theta1, theta2, theta3, theta4, theta5, theta6 = self.physical_params()
        center = torch.as_tensor(X_JET, dtype=x_star.dtype, device=x_star.device)
        g_center = theta1 * torch.exp(-0.5 * ((x_star - center) / theta2) ** 2)
        g_left = torch.exp(-0.5 * ((x_star - (center - theta4)) / theta5) ** 2)
        g_right = torch.exp(-0.5 * ((x_star - (center + theta4)) / theta5) ** 2)
        return g_center + theta3 * (g_left + g_right) + theta6


class DiscreteNodalBiTop(nn.Module):
    def __init__(self, n_nodes, init_bi_min, init_bi_max):
        super().__init__()
        if n_nodes < 3:
            raise ValueError("n_nodes must be at least 3 for second-order smoothness.")
        self.n_nodes = int(n_nodes)
        self.softplus = nn.Softplus()
        self.register_buffer("x_nodes", torch.linspace(0.0, 1.0, self.n_nodes, dtype=torch.float64))

        init_value = 0.5 * (float(init_bi_min) + float(init_bi_max))
        init = torch.full((self.n_nodes,), self._inv_softplus(max(init_value, 1e-4)), dtype=torch.float64)
        self.raw_nodes = nn.Parameter(init)

    @staticmethod
    def _inv_softplus(y):
        return math.log(math.expm1(float(y)))

    def nodal_values(self):
        return self.softplus(self.raw_nodes) + 1e-12

    def physical_params(self):
        return self.nodal_values()

    def smoothness_loss(self):
        bi = self.nodal_values()
        second_diff = bi[:-2] - 2.0 * bi[1:-1] + bi[2:]
        return torch.mean(second_diff**2)

    def forward(self, x_star):
        x = torch.clamp(x_star, 0.0, 1.0)
        x_flat = x.reshape(-1)
        scaled = x_flat * (self.n_nodes - 1)
        idx_left = torch.floor(scaled).long().clamp(0, self.n_nodes - 2)
        frac = (scaled - idx_left.to(scaled.dtype)).unsqueeze(1)

        bi = self.nodal_values()
        left = bi[idx_left].unsqueeze(1)
        right = bi[idx_left + 1].unsqueeze(1)
        return ((1.0 - frac) * left + frac * right).reshape_as(x_star)


class AuxiliaryNNBiTop(nn.Module):
    def __init__(
        self,
        init_bi_min,
        init_bi_max,
        hidden_width=16,
        hidden_layers=2,
        activation="tanh",
        smooth_n_points=101,
        probe_n_points=21,
    ):
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least 1.")
        self.softplus = nn.Softplus()
        self.smooth_n_points = int(smooth_n_points)

        activation_layer = self._activation_layer(activation)
        layers = []
        in_features = 1
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_features, int(hidden_width)))
            layers.append(activation_layer())
            in_features = int(hidden_width)
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

        init_value = 0.5 * (float(init_bi_min) + float(init_bi_max))
        self._initialize_constant(max(init_value, 1e-4))
        self.register_buffer("x_probe", torch.linspace(0.0, 1.0, int(probe_n_points), dtype=torch.float64))

    @staticmethod
    def _activation_layer(name):
        key = str(name).lower()
        if key == "tanh":
            return nn.Tanh
        if key == "silu":
            return nn.SiLU
        if key == "gelu":
            return nn.GELU
        raise ValueError(f"unknown auxiliary NN activation: {name}")

    @staticmethod
    def _inv_softplus(y):
        return math.log(math.expm1(float(y)))

    def _initialize_constant(self, init_bi):
        linear_layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        for layer in linear_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        final = linear_layers[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, self._inv_softplus(init_bi))

    def physical_params(self):
        return self(self.x_probe.reshape(-1, 1)).reshape(-1)

    def smoothness_loss(self):
        if self.smooth_n_points < 3:
            return torch.zeros((), dtype=self.x_probe.dtype, device=self.x_probe.device)
        x = torch.linspace(
            0.0,
            1.0,
            self.smooth_n_points,
            dtype=self.x_probe.dtype,
            device=self.x_probe.device,
            requires_grad=True,
        ).reshape(-1, 1)
        bi = self(x)
        bi_x = torch.autograd.grad(bi, x, torch.ones_like(bi), create_graph=True)[0]
        bi_xx = torch.autograd.grad(bi_x, x, torch.ones_like(bi_x), create_graph=True)[0]
        return torch.mean(bi_xx**2)

    def forward(self, x_star):
        x = torch.clamp(x_star, 0.0, 1.0)
        return self.softplus(self.net(x)) + 1e-12


def build_bi_top_model(config, init_bi_min, init_bi_max, device):
    model_type = config["type"].lower()
    if model_type in {"gaussian", "gaussian6"}:
        model = GaussianBiTop(init_bi_max=init_bi_max, init_bi_min=init_bi_min)
        param_names = GAUSSIAN_PARAM_NAMES
    elif model_type in {"discrete_nodal", "discrete_points"}:
        model = DiscreteNodalBiTop(
            n_nodes=config.get("n_nodes", 21),
            init_bi_max=init_bi_max,
            init_bi_min=init_bi_min,
        )
        param_names = [f"bi_node_{i:02d}" for i in range(model.n_nodes)]
    elif model_type in {"auxiliary_nn", "aux_nn", "bi_nn", "method3"}:
        model = AuxiliaryNNBiTop(
            init_bi_max=init_bi_max,
            init_bi_min=init_bi_min,
            hidden_width=config.get("hidden_width", 16),
            hidden_layers=config.get("hidden_layers", 2),
            activation=config.get("activation", "tanh"),
            smooth_n_points=config.get("smooth_n_points", 101),
            probe_n_points=config.get("probe_n_points", 21),
        )
        param_names = [f"bi_nn_probe_{i:02d}" for i in range(len(model.x_probe))]
    else:
        raise ValueError(f"unknown Bi top model type: {config['type']}")
    return model.to(device), param_names


def bi_model_slug(config):
    return config["type"].lower().replace(" ", "_").replace("-", "_")


def bi_model_label(config):
    model_type = config["type"].lower()
    if model_type in {"gaussian", "gaussian6"}:
        return "inverse PINN Gaussian Bi"
    if model_type in {"discrete_nodal", "discrete_points"}:
        return "inverse PINN nodal Bi"
    if model_type in {"auxiliary_nn", "aux_nn", "bi_nn", "method3"}:
        return "inverse PINN auxiliary-NN Bi"
    return f"inverse PINN {config['type']} Bi"


def bi_model_values(model):
    return [float(v.detach().cpu()) for v in model.physical_params()]


def bi_model_status(model):
    values = np.asarray(bi_model_values(model), dtype=float)
    if hasattr(model, "x_nodes") or hasattr(model, "x_probe"):
        return f"Bi[min,max]=({values.min():.3g}, {values.max():.3g})"
    return f"params={['%.3g' % v for v in values]}"


def gradients(y, x, retain_graph=True):
    return torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=retain_graph,
    )[0]


def pde_residual_2d(model, xy):
    if not xy.requires_grad:
        xy = xy.clone().detach().requires_grad_(True)
    theta = model(xy)
    grad_theta = gradients(theta, xy)
    theta_x = grad_theta[:, 0:1]
    theta_y = grad_theta[:, 1:2]
    theta_xx = gradients(theta_x, xy)[:, 0:1]
    theta_yy = gradients(theta_y, xy)[:, 1:2]
    return theta_xx + theta_yy + Q_STAR


def dirichlet_bottom_loss(model, xy_bottom):
    return mse(model(xy_bottom) - 1.0)


def top_robin_loss(model, bi_model, xy_top):
    xy_top = xy_top.clone().detach().requires_grad_(True)
    theta = model(xy_top)
    grad_theta = gradients(theta, xy_top)
    residual = -grad_theta[:, 1:2] - bi_model(xy_top[:, 0:1]) * theta
    return mse(residual)


def side_robin_loss(model, xy_left, xy_right):
    xy_left = xy_left.clone().detach().requires_grad_(True)
    theta_left = model(xy_left)
    grad_left = gradients(theta_left, xy_left)
    left_residual = grad_left[:, 0:1] - BI_SIDE * theta_left

    xy_right = xy_right.clone().detach().requires_grad_(True)
    theta_right = model(xy_right)
    grad_right = gradients(theta_right, xy_right)
    right_residual = -grad_right[:, 0:1] - BI_SIDE * theta_right
    return mse(left_residual) + mse(right_residual)


def sensor_data_loss(model, x_sensor_t, theta_sensor_t):
    return mse(model(x_sensor_t) - theta_sensor_t)


def sensor_theta_y_loss(
    model,
    theta_y_lower_t,
    theta_y_upper_t,
    theta_y_sensor_t,
    theta_y_fit_points_t=None,
    theta_y_fit_weights_t=None,
):
    if theta_y_sensor_t.numel() == 0:
        return torch.zeros((), dtype=torch.float64, device=DEVICE)
    if theta_y_fit_weights_t is not None and theta_y_fit_weights_t.numel() > 0:
        theta_fit = model(theta_y_fit_points_t)
        theta_y_fit = theta_y_fit_weights_t @ theta_fit
        return mse(theta_y_fit - theta_y_sensor_t)
    theta_upper = model(theta_y_upper_t)
    theta_lower = model(theta_y_lower_t)
    dy_pair = theta_y_upper_t[:, 1:2] - theta_y_lower_t[:, 1:2]
    theta_y_fd = (theta_upper - theta_lower) / dy_pair
    return mse(theta_y_fd - theta_y_sensor_t)


def pinn_inverse_loss(
    model,
    bi_model,
    xy_f,
    boundary,
    weights,
    x_sensor_t,
    theta_sensor_t,
    theta_y_lower_t,
    theta_y_upper_t,
    theta_y_sensor_t,
    theta_y_fit_points_t=None,
    theta_y_fit_weights_t=None,
):
    losses = {
        "pde": mse(pde_residual_2d(model, xy_f)),
        "bottom": dirichlet_bottom_loss(model, boundary["bottom"]),
        "top": top_robin_loss(model, bi_model, boundary["top"]),
        "side": side_robin_loss(model, boundary["left"], boundary["right"]),
        "data": sensor_data_loss(model, x_sensor_t, theta_sensor_t),
        "theta_y_data": sensor_theta_y_loss(
            model,
            theta_y_lower_t,
            theta_y_upper_t,
            theta_y_sensor_t,
            theta_y_fit_points_t,
            theta_y_fit_weights_t,
        ),
    }
    if hasattr(bi_model, "smoothness_loss") and weights.get("smooth", 0.0) != 0.0:
        losses["smooth"] = bi_model.smoothness_loss()
    total = sum(weights.get(name, 0.0) * value for name, value in losses.items())
    return total, losses


def get_gaussian_params(bi_model):
    return [float(v.detach().cpu()) for v in bi_model.physical_params()]


def run_one_experiment(spec, ablation_dir, args):
    config = deep_update(DEFAULT_CONFIG, spec.get("config", {}))
    weights = deep_update(DEFAULT_WEIGHTS, spec.get("weights", {}))
    optimizer_config = deep_update(DEFAULT_OPTIMIZER_CONFIG, spec.get("optimizer", {}))
    bi_model_config = deep_update(DEFAULT_BI_MODEL_CONFIG, spec.get("bi_model", spec.get("bi_model_config", {})))
    bi_slug = bi_model_slug(bi_model_config)

    if args.n_adam is not None:
        optimizer_config["n_adam"] = args.n_adam
    if args.no_lbfgs:
        optimizer_config["run_lbfgs"] = False

    name = slugify(spec["name"])
    topic = slugify(spec.get("topic", "general"))
    topic_dir = ablation_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime(
        f"run_%Y%m%d_%H%M%S_{name}_{bi_slug}_{config['sampling']}_"
        f"Nf{config['n_domain']}_Nb{config['n_boundary']}_seed{config['seed']}"
    )
    run_dir = topic_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    set_all_seeds(int(config["seed"]))

    sensor_xy = make_sensor_points(config["sensor_x"], config["sensor_y"])
    theta_sensor = sample_bilinear(xs, ys, theta_ref, sensor_xy)
    theta_sensor_clean = theta_sensor.copy()
    theta_sensor_noise = gaussian_sensor_noise(
        theta_sensor_clean.shape,
        config["noise_level_theta"],
        config["noise_seed"],
    )
    theta_sensor_train = (
        theta_sensor_clean + theta_sensor_noise
        if config["use_noisy_sensor"]
        else theta_sensor_clean.copy()
    )
    theta_y_method = config.get("theta_y_method", "fd_pairs")
    (
        theta_y_lower_xy,
        theta_y_upper_xy,
        theta_y_pair_xy,
        theta_y_sensor_ref,
        theta_y_fit_points_xy,
        theta_y_fit_weights,
    ) = build_sensor_theta_y_data(
        sensor_xy,
        theta_sensor_train,
        method=theta_y_method,
    )

    x_sensor_t = torch.tensor(sensor_xy, dtype=torch.float64, device=DEVICE)
    theta_sensor_t = torch.tensor(theta_sensor_train, dtype=torch.float64, device=DEVICE)
    theta_y_lower_t = torch.tensor(theta_y_lower_xy, dtype=torch.float64, device=DEVICE)
    theta_y_upper_t = torch.tensor(theta_y_upper_xy, dtype=torch.float64, device=DEVICE)
    theta_y_sensor_t = torch.tensor(theta_y_sensor_ref, dtype=torch.float64, device=DEVICE)
    theta_y_fit_points_t = torch.tensor(theta_y_fit_points_xy, dtype=torch.float64, device=DEVICE)
    theta_y_fit_weights_t = torch.tensor(theta_y_fit_weights, dtype=torch.float64, device=DEVICE)
    save_sensor_layout_diagram(
        run_dir,
        sensor_xy,
        theta_y_pair_xy,
        title=sensor_layout_summary_title(spec["name"], sensor_xy),
    )

    xy_f, boundary = make_training_points(
        n_domain=config["n_domain"],
        n_boundary=config["n_boundary"],
        sampling=config["sampling"],
        boundary_mode=config["boundary_mode"],
        seed=config["seed"],
    )

    pinn = PINN2D(
        hidden_layers=config["hidden_layers"],
        neurons=config["neurons"],
        activation=config["activation"],
    ).to(DEVICE)
    bi_top_model, bi_param_names = build_bi_top_model(
        bi_model_config,
        init_bi_max=INIT_BI_MAX,
        init_bi_min=INIT_BI_MIN,
        device=DEVICE,
    )

    optimizer = torch.optim.Adam(
        [
            {"params": pinn.parameters(), "lr": optimizer_config["pinn_lr"]},
            {"params": bi_top_model.parameters(), "lr": optimizer_config["bi_lr"]},
        ]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=optimizer_config["n_adam"],
        eta_min=optimizer_config["adam_eta_min"],
    )

    history = []
    t0 = time.perf_counter()
    print("\n" + "=" * 88, flush=True)
    print(f"START {name}", flush=True)
    print(f"topic: {topic}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(DEVICE)}", flush=True)
    print(f"config: {config}", flush=True)
    print(
        "architecture: "
        f"{config['hidden_layers']} hidden layers x {config['neurons']} nodes, "
        f"activation={config['activation']}, "
        f"parameters={sum(p.numel() for p in pinn.parameters()):,}",
        flush=True,
    )
    print(f"weights: {weights}", flush=True)
    print(f"bi_model: {bi_model_config}", flush=True)
    print(f"optimizer: {optimizer_config}", flush=True)
    print(f"sensor_count: {len(sensor_xy)}", flush=True)
    print(f"theta_y method: {theta_y_method}", flush=True)
    print(f"theta_y data_count: {len(theta_y_sensor_ref)}", flush=True)
    if len(theta_y_sensor_ref):
        print(f"theta_y data range: {theta_y_sensor_ref.min():.4e} to {theta_y_sensor_ref.max():.4e}", flush=True)
    else:
        print("theta_y data range: no valid theta_y sensor data; theta_y_data loss is zero", flush=True)

    for epoch in range(1, optimizer_config["n_adam"] + 1):
        pinn.train()
        bi_top_model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, parts = pinn_inverse_loss(
            pinn,
            bi_top_model,
            xy_f,
            boundary,
            weights,
            x_sensor_t,
            theta_sensor_t,
            theta_y_lower_t,
            theta_y_upper_t,
            theta_y_sensor_t,
            theta_y_fit_points_t,
            theta_y_fit_weights_t,
        )
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        row = {"epoch": epoch, "total": float(total_loss.detach().cpu())}
        row.update({key: float(value.detach().cpu()) for key, value in parts.items()})

        if epoch == 1 or epoch % optimizer_config["print_every"] == 0:
            history.append(row)
            elapsed = time.perf_counter() - t0
            print(
                f"Adam {epoch:6d} | L={row['total']:.3e} "
                f"PDE={row['pde']:.2e} Bot={row['bottom']:.2e} Top={row['top']:.2e} "
                f"Side={row['side']:.2e} Data={row['data']:.2e} dTy={row['theta_y_data']:.2e} "
                f"Smooth={row.get('smooth', 0.0):.2e} | "
                f"{bi_model_status(bi_top_model)} | {elapsed:.1f}s",
                flush=True,
            )

    if optimizer_config["run_lbfgs"]:
        optimizer_lbfgs = torch.optim.LBFGS(
            list(pinn.parameters()) + list(bi_top_model.parameters()),
            lr=optimizer_config["lbfgs_lr"],
            max_iter=optimizer_config["lbfgs_max_iter"],
            history_size=optimizer_config["lbfgs_history_size"],
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            loss, _ = pinn_inverse_loss(
                pinn,
                bi_top_model,
                xy_f,
                boundary,
                weights,
                x_sensor_t,
                theta_sensor_t,
                theta_y_lower_t,
                theta_y_upper_t,
                theta_y_sensor_t,
                theta_y_fit_points_t,
                theta_y_fit_weights_t,
            )
            loss.backward()
            return loss

        for step in range(1, optimizer_config["lbfgs_steps"] + 1):
            optimizer_lbfgs.step(closure)
            if step == 1 or step % 50 == 0:
                total_loss, parts = pinn_inverse_loss(
                    pinn,
                    bi_top_model,
                    xy_f,
                    boundary,
                    weights,
                    x_sensor_t,
                    theta_sensor_t,
                    theta_y_lower_t,
                    theta_y_upper_t,
                    theta_y_sensor_t,
                    theta_y_fit_points_t,
                    theta_y_fit_weights_t,
                )
                row = {"epoch": optimizer_config["n_adam"] + step, "total": float(total_loss.detach().cpu())}
                row.update({key: float(value.detach().cpu()) for key, value in parts.items()})
                history.append(row)
                print(
                    f"L-BFGS {step:4d} | L={row['total']:.3e} "
                    f"PDE={row['pde']:.2e} Top={row['top']:.2e} "
                    f"Data={row['data']:.2e} dTy={row['theta_y_data']:.2e} "
                    f"Smooth={row.get('smooth', 0.0):.2e} | "
                    f"{bi_model_status(bi_top_model)}",
                    flush=True,
                )

    pinn.eval()
    bi_top_model.eval()

    xx, yy = np.meshgrid(xs, ys)
    xy_eval = np.column_stack([xx.ravel(), yy.ravel()])
    with torch.no_grad():
        theta_pred = (
            pinn(torch.tensor(xy_eval, dtype=torch.float64, device=DEVICE))
            .cpu()
            .numpy()
            .reshape(len(ys), len(xs))
        )
        sensor_pred = pinn(x_sensor_t).cpu().numpy()
        if theta_y_fit_weights_t.numel():
            theta_y_sensor_pred = (theta_y_fit_weights_t @ pinn(theta_y_fit_points_t)).cpu().numpy()
        elif theta_y_sensor_t.numel():
            theta_y_sensor_pred = (
                (pinn(theta_y_upper_t) - pinn(theta_y_lower_t))
                / (theta_y_upper_t[:, 1:2] - theta_y_lower_t[:, 1:2])
            ).cpu().numpy()
        else:
            theta_y_sensor_pred = np.empty((0, 1), dtype=float)

    abs_err = np.abs(theta_pred - theta_ref)
    rel_l2 = np.linalg.norm(theta_pred.ravel() - theta_ref.ravel()) / np.linalg.norm(theta_ref.ravel())
    max_abs = abs_err.max()
    sensor_clean_max_abs = np.max(np.abs(sensor_pred - theta_sensor_clean))
    sensor_train_max_abs = np.max(np.abs(sensor_pred - theta_sensor_train))
    sensor_theta_y_max_abs = (
        np.max(np.abs(theta_y_sensor_pred - theta_y_sensor_ref))
        if len(theta_y_sensor_ref)
        else float("nan")
    )

    x_plot = np.linspace(0.0, 1.0, 501)[:, None]
    with torch.no_grad():
        bi_pred = (
            bi_top_model(torch.tensor(x_plot, dtype=torch.float64, device=DEVICE))
            .cpu()
            .numpy()
            .ravel()
        )
    bi_true = true_bi_numpy(x_plot.ravel())
    bi_rel_l2 = np.linalg.norm(bi_pred - bi_true) / np.linalg.norm(bi_true)
    center_idx = int(np.argmin(np.abs(x_plot.ravel() - X_JET)))
    bi_center_true = float(bi_true[center_idx])
    bi_center_pred = float(bi_pred[center_idx])
    bi_center_abs_error = abs(bi_center_pred - bi_center_true)
    h_center_true = bi_center_true * K_S / L
    h_center_pred = bi_center_pred * K_S / L
    h_center_abs_error = abs(h_center_pred - h_center_true)
    h_center_rel_error_pct = 100.0 * h_center_abs_error / max(abs(h_center_true), 1e-12)

    bi_param_values = bi_model_values(bi_top_model)
    bi_node_x = None
    bi_node_values = None
    h_node_values = None
    bi_probe_x = None
    bi_probe_values = None
    h_probe_values = None
    if hasattr(bi_top_model, "x_nodes"):
        bi_node_x = bi_top_model.x_nodes.detach().cpu().numpy()
        bi_node_values = np.asarray(bi_param_values, dtype=float)
        h_node_values = bi_node_values * K_S / L
    elif hasattr(bi_top_model, "x_probe"):
        bi_probe_x = bi_top_model.x_probe.detach().cpu().numpy()
        bi_probe_values = np.asarray(bi_param_values, dtype=float)
        h_probe_values = bi_probe_values * K_S / L

    summary = {
        "run_id": run_id,
        "ablation_topic": topic,
        "ablation_name": name,
        "description": spec.get("description", ""),
        "run_dir": str(run_dir),
        "bi_model_config": bi_model_config,
        "bi_param_names": bi_param_names,
        "bi_param_values": bi_param_values,
        "config": config,
        "weights": weights,
        "optimizer_config": optimizer_config,
        "sensor_count": int(len(sensor_xy)),
        "sensor_xy": sensor_xy.tolist(),
        "theta_y_method": str(theta_y_method),
        "theta_y_fd_pair_count": int(len(theta_y_sensor_ref)),
        "theta_y_pair_xy": theta_y_pair_xy.tolist(),
        "rel_l2": float(rel_l2),
        "max_abs": float(max_abs),
        "sensor_clean_max_abs": float(sensor_clean_max_abs),
        "sensor_train_max_abs": float(sensor_train_max_abs),
        "sensor_theta_y_max_abs": float(sensor_theta_y_max_abs),
        "bi_rel_l2": float(bi_rel_l2),
        "bi_center_true": float(bi_center_true),
        "bi_center_pred": float(bi_center_pred),
        "bi_center_abs_error": float(bi_center_abs_error),
        "h_center_true": float(h_center_true),
        "h_center_pred": float(h_center_pred),
        "h_center_abs_error": float(h_center_abs_error),
        "h_center_rel_error_pct": float(h_center_rel_error_pct),
        "h_min_guess": float(H_MIN_GUESS),
        "h_max_guess": float(H_MAX_GUESS),
        "init_bi_min": float(INIT_BI_MIN),
        "init_bi_max": float(INIT_BI_MAX),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    if bi_model_config["type"].lower() in {"gaussian", "gaussian6"}:
        summary["gaussian_param_names"] = GAUSSIAN_PARAM_NAMES
        summary["gaussian_params"] = bi_param_values
    if bi_node_x is not None:
        summary.update(
            {
                "bi_node_x": bi_node_x.tolist(),
                "bi_node_values": bi_node_values.tolist(),
                "h_node_values": h_node_values.tolist(),
            }
        )
    if bi_probe_x is not None:
        summary.update(
            {
                "bi_probe_x": bi_probe_x.tolist(),
                "bi_probe_values": bi_probe_values.tolist(),
                "h_probe_values": h_probe_values.tolist(),
            }
        )

    torch.save(
        {
            "pinn_state_dict": pinn.state_dict(),
            "bi_top_state_dict": bi_top_model.state_dict(),
            "history": history,
            **summary,
        },
        run_dir / f"pinn_inverse_{bi_slug}.pt",
    )

    with open(run_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if history:
        with open(run_dir / "loss_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    with open(run_dir / "sensor_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sensor_id",
                "x_star",
                "y_star",
                "theta_clean",
                "theta_train",
                "theta_noise",
                "theta_pred",
            ],
        )
        writer.writeheader()
        for sensor_id, ((x, y), clean, train, noise, pred) in enumerate(
            zip(
                sensor_xy,
                theta_sensor_clean.ravel(),
                theta_sensor_train.ravel(),
                theta_sensor_noise.ravel(),
                sensor_pred.ravel(),
            )
        ):
            writer.writerow(
                {
                    "sensor_id": sensor_id,
                    "x_star": float(x),
                    "y_star": float(y),
                    "theta_clean": float(clean),
                    "theta_train": float(train),
                    "theta_noise": float(noise),
                    "theta_pred": float(pred),
                }
            )

    with open(run_dir / "theta_y_fd_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "x_star",
                "y_lower",
                "y_upper",
                "theta_y_fd_ref",
                "theta_y_fd_pred",
            ],
        )
        writer.writeheader()
        for pair_id, (lower_xy, upper_xy, theta_y_ref_i, theta_y_pred_i) in enumerate(
            zip(theta_y_lower_xy, theta_y_upper_xy, theta_y_sensor_ref.ravel(), theta_y_sensor_pred.ravel())
        ):
            writer.writerow(
                {
                    "pair_id": pair_id,
                    "x_star": float(theta_y_pair_xy[pair_id, 0]),
                    "y_lower": float(lower_xy[1]),
                    "y_upper": float(upper_xy[1]),
                    "theta_y_fd_ref": float(theta_y_ref_i),
                    "theta_y_fd_pred": float(theta_y_pred_i),
                }
            )

    np.savez_compressed(
        run_dir / f"pinn_inverse_{bi_slug}_fields.npz",
        xs=xs,
        ys=ys,
        theta_ref=theta_ref,
        theta_pred=theta_pred,
        abs_err=abs_err,
        sensor_xy=sensor_xy,
        theta_sensor=theta_sensor,
        theta_sensor_clean=theta_sensor_clean,
        theta_sensor_train=theta_sensor_train,
        theta_sensor_noise=theta_sensor_noise,
        theta_sensor_pred=sensor_pred,
        theta_y_lower_xy=theta_y_lower_xy,
        theta_y_upper_xy=theta_y_upper_xy,
        theta_y_pair_xy=theta_y_pair_xy,
        theta_y_sensor_ref=theta_y_sensor_ref,
        theta_y_sensor_pred=theta_y_sensor_pred,
        theta_y_method=np.asarray(str(theta_y_method)),
        theta_y_fit_points_xy=theta_y_fit_points_xy,
        theta_y_fit_weights=theta_y_fit_weights,
        use_noisy_sensor=config["use_noisy_sensor"],
        noise_level_theta=config["noise_level_theta"],
        noise_seed=config["noise_seed"],
        h_min_guess=H_MIN_GUESS,
        h_max_guess=H_MAX_GUESS,
        init_bi_min=INIT_BI_MIN,
        init_bi_max=INIT_BI_MAX,
        bi_model_type=bi_model_config["type"],
        bi_param_names=np.asarray(bi_param_names),
        bi_param_values=np.asarray(bi_param_values, dtype=float),
        bi_node_x=np.asarray([] if bi_node_x is None else bi_node_x),
        bi_node_values=np.asarray([] if bi_node_values is None else bi_node_values),
        h_node_values=np.asarray([] if h_node_values is None else h_node_values),
        bi_probe_x=np.asarray([] if bi_probe_x is None else bi_probe_x),
        bi_probe_values=np.asarray([] if bi_probe_values is None else bi_probe_values),
        h_probe_values=np.asarray([] if h_probe_values is None else h_probe_values),
        x_plot=x_plot.ravel(),
        bi_true=bi_true,
        bi_pred=bi_pred,
    )

    save_plots(
        run_dir,
        history,
        sensor_xy,
        theta_sensor_noise,
        theta_pred,
        abs_err,
        x_plot,
        bi_true,
        bi_pred,
        max_abs,
        bi_model_config,
        bi_node_x=bi_node_x,
        bi_node_values=bi_node_values,
    )
    if config["use_noisy_sensor"]:
        save_sensor_apparent_bi_overlay(
            run_dir,
            sensor_xy,
            theta_sensor_clean,
            theta_sensor_train,
            sensor_pred,
            theta_y_method,
            x_plot,
            bi_true,
            bi_pred,
        )

    print(
        f"END {name} | rel_l2={rel_l2:.4e} max_abs={max_abs:.4e} "
        f"sensor_dTy={sensor_theta_y_max_abs:.4e} bi_rel_l2={bi_rel_l2:.4e}",
        flush=True,
    )

    del pinn, bi_top_model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def save_plots(
    run_dir,
    history,
    sensor_xy,
    theta_sensor_noise,
    theta_pred,
    abs_err,
    x_plot,
    bi_true,
    bi_pred,
    max_abs,
    bi_model_config,
    bi_node_x=None,
    bi_node_values=None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160, constrained_layout=True)
    im0 = axes[0].contourf(xs, ys, theta_ref, levels=40, cmap="inferno")
    axes[0].contour(xs, ys, theta_ref, levels=10, colors="white", linewidths=0.4, alpha=0.55)
    axes[0].scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=28, facecolor="none", edgecolor="cyan", linewidth=1.0)
    axes[0].set_title("FDM reference theta")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(xs, ys, theta_pred, levels=40, cmap="inferno")
    axes[1].contour(xs, ys, theta_pred, levels=10, colors="white", linewidths=0.4, alpha=0.55)
    axes[1].scatter(sensor_xy[:, 0], sensor_xy[:, 1], s=28, facecolor="none", edgecolor="cyan", linewidth=1.0)
    axes[1].set_title("PINN theta")
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].contourf(xs, ys, abs_err, levels=40, cmap="viridis")
    axes[2].set_title(f"absolute error, max={max_abs:.2e}")
    fig.colorbar(im2, ax=axes[2])
    for ax in axes:
        ax.set_xlabel("x*")
        ax.set_ylabel("y*")
        ax.set_aspect("equal")
    fig.savefig(run_dir / "field_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=160)
    ax.plot(x_plot.ravel(), bi_true, "k-", lw=2, label="FDM/empirical Bi")
    ax.plot(x_plot.ravel(), bi_pred, "r--", lw=2, label=bi_model_label(bi_model_config))
    if bi_node_x is not None:
        ax.plot(bi_node_x, bi_node_values, "ro", ms=3.5, label="trainable nodes")
    ax.set_xlabel("x*")
    ax.set_ylabel("Bi_top(x*)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(run_dir / "bi_profile.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=160)
    ax.stem(np.arange(len(theta_sensor_noise)), theta_sensor_noise.ravel(), basefmt="k-")
    ax.set_xlabel("sensor index")
    ax.set_ylabel("noise in theta")
    ax.grid(True, alpha=0.3)
    fig.savefig(run_dir / "sensor_noise.png", bbox_inches="tight")
    plt.close(fig)

    if history:
        epochs = np.array([row["epoch"] for row in history])
        fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
        for key in ["total", "pde", "bottom", "top", "side", "data", "theta_y_data", "smooth"]:
            vals = np.array([row[key] for row in history if key in row])
            ax.semilogy(epochs[: len(vals)], vals, label=key)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(ncol=3)
        fig.savefig(run_dir / "loss_history.png", bbox_inches="tight")
        plt.close(fig)


def select_suite(name):
    if name == "default":
        return DEFAULT_ABLATIONS
    if name == "all_topics":
        return DEFAULT_ABLATIONS
    if name == "depth":
        return DEPTH_ABLATIONS
    if name == "node_count":
        return NODE_COUNT_ABLATIONS
    if name == "activation":
        return ACTIVATION_ABLATIONS
    if name == "sensor":
        return SENSOR_PLACEMENT_ABLATIONS
    if name == "sensor_stage1":
        return STAGE1_TWO_ROW_SENSOR_ABLATIONS
    if name in {"sensor_stage2", "sensor_stage2_fd"}:
        return STAGE2_X_PLACEMENT_SENSOR_ABLATIONS
    if name in {"sensor_stage3", "sensor_stage3_fd"}:
        return STAGE3_Y_PLACEMENT_SENSOR_ABLATIONS
    if name in {"sensor_stage4", "sensor_stage4_fd"}:
        return STAGE4_SENSOR_COUNT_ABLATIONS
    if name in {"sensor_stage4_nested", "sensor_stage4_nested_fd", "sensor_placement_update", "sensor_placement_update_fd"}:
        return STAGE4_NESTED_SENSOR_COUNT_ABLATIONS
    if name in {"sensor_stage5", "sensor_stage5_noise", "sensor_stage5_noise_fd"}:
        return STAGE5_NOISE_LEVEL_SENSOR_ABLATIONS
    if name in {"sensor_stage5_noise_continue", "sensor_stage5_noise_continue_fd"}:
        return STAGE5_NOISE_LEVEL_CONTINUATION_ABLATIONS
    if name in {"sensor_stage5_noise_seed", "sensor_stage5_noise_seed_fd"}:
        return STAGE5_NOISE_SEED_ABLATIONS
    if name in {"sensor_stage5_2", "sensor_stage5_regularization", "sensor_stage5_regularization_fd"}:
        return STAGE5_REGULARIZATION_SENSOR_ABLATIONS
    if name in {"sensor_stage5_3", "sensor_stage5_local_linear", "sensor_stage5_local_linear_fd", "sensor_stage5_local_linear_4row"}:
        return STAGE5_LOCAL_LINEAR_4ROW_SENSOR_ABLATIONS
    if name in {
        "sensor_stage5_4",
        "sensor_stage5_local_linear_tune",
        "sensor_stage5_local_linear_4row_tune",
        "sensor_stage5_local_linear_4row_tune_0p002",
    }:
        return STAGE5_LOCAL_LINEAR_4ROW_TUNE_0P002_SENSOR_ABLATIONS
    if name in {"final_seed_robustness", "final_seed_repeat", "final_seed_ablation"}:
        return FINAL_SEED_ROBUSTNESS_ABLATIONS
    if name in {"sensor_stage1_stage2", "sensor_stage12"}:
        return STAGE1_STAGE2_SENSOR_ABLATIONS
    if name == "legacy":
        return LEGACY_ABLATIONS
    if name == "quick":
        return QUICK_ABLATIONS
    if name == "smoke":
        return SMOKE_ABLATIONS
    if name == "bi_best_weights":
        return BI_MODEL_BEST_WEIGHT_RETRAIN_ABLATIONS
    if name == "bi_capacity":
        return BI_MODEL_CAPACITY_DEPENDENCY_ABLATIONS
    raise ValueError(f"unknown suite: {name}")


def write_ablation_summary(ablation_dir, results):
    with open(ablation_dir / "ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    flat_rows = []
    for row in results:
        if "error" in row:
            flat_rows.append(
                {
                    "ablation_topic": row.get("ablation_topic", ""),
                    "ablation_name": row.get("ablation_name", ""),
                    "run_id": row.get("run_id", ""),
                    "error": row["error"],
                }
            )
            continue
        flat = {
            "ablation_topic": row["ablation_topic"],
            "ablation_name": row["ablation_name"],
            "run_id": row["run_id"],
            "sensor_count": row["sensor_count"],
            "theta_y_fd_pair_count": row.get("theta_y_fd_pair_count", ""),
            "theta_y_method": row.get("theta_y_method", row["config"].get("theta_y_method", "fd_pairs")),
            "sampling": row["config"]["sampling"],
            "seed": row["config"].get("seed", ""),
            "hidden_layers": row["config"]["hidden_layers"],
            "neurons": row["config"]["neurons"],
            "activation": row["config"]["activation"],
            "use_noisy_sensor": row["config"]["use_noisy_sensor"],
            "noise_level_theta": row["config"].get("noise_level_theta", ""),
            "noise_seed": row["config"].get("noise_seed", ""),
            "theta_y_weight": row["weights"]["theta_y_data"],
            "smooth_weight": row["weights"].get("smooth", 0.0),
            "bi_model_type": row.get("bi_model_config", {}).get("type", "gaussian"),
            "rel_l2": row["rel_l2"],
            "max_abs": row["max_abs"],
            "sensor_clean_max_abs": row["sensor_clean_max_abs"],
            "sensor_train_max_abs": row["sensor_train_max_abs"],
            "sensor_theta_y_max_abs": row["sensor_theta_y_max_abs"],
            "bi_rel_l2": row["bi_rel_l2"],
            "bi_center_true": row.get("bi_center_true", ""),
            "bi_center_pred": row.get("bi_center_pred", ""),
            "bi_center_abs_error": row.get("bi_center_abs_error", ""),
            "h_center_true": row.get("h_center_true", ""),
            "h_center_pred": row.get("h_center_pred", ""),
            "h_center_abs_error": row.get("h_center_abs_error", ""),
            "h_center_rel_error_pct": row.get("h_center_rel_error_pct", ""),
            "elapsed_sec": row["elapsed_sec"],
            "run_dir": row["run_dir"],
        }
        for name, value in zip(row.get("bi_param_names", row.get("gaussian_param_names", [])), row.get("bi_param_values", row.get("gaussian_params", []))):
            flat[name] = value
        flat_rows.append(flat)

    if flat_rows:
        with open(ablation_dir / "ablation_results.csv", "w", newline="") as f:
            fieldnames = sorted({key for row in flat_rows for key in row.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)


def write_topic_summaries(ablation_dir, results):
    topics = sorted({row.get("ablation_topic", "general") for row in results})
    for topic in topics:
        topic_dir = ablation_dir / topic
        topic_dir.mkdir(parents=True, exist_ok=True)
        topic_results = [row for row in results if row.get("ablation_topic", "general") == topic]
        write_ablation_summary(topic_dir, topic_results)


def _load_completed_summary(run_dir):
    run_dir = Path(run_dir)
    summary_path = run_dir / "run_summary.json"
    field_paths = sorted(run_dir.glob("pinn_inverse_*_fields.npz"))
    if not summary_path.exists() or not field_paths:
        return None
    with open(summary_path) as f:
        summary = json.load(f)
    summary["_field_path"] = str(field_paths[0])
    summary["_run_dir_mtime"] = run_dir.stat().st_mtime
    return summary


def collect_completed_runs(output_root, topic=None, names=None):
    output_root = Path(output_root)
    names = set(names or [])
    summaries = []
    for summary_path in output_root.glob("ablation_*/*/run_*/run_summary.json"):
        run_dir = summary_path.parent
        summary = _load_completed_summary(run_dir)
        if summary is None:
            continue
        if topic is not None and summary.get("ablation_topic") != topic:
            continue
        if names and summary.get("ablation_name") not in names:
            continue
        summaries.append(summary)
    summaries.sort(key=lambda row: row.get("_run_dir_mtime", 0.0))
    return summaries


def latest_by_key(rows, key_fn):
    selected = {}
    for row in rows:
        selected[key_fn(row)] = row
    return list(selected.values())


def _noise_level_label(value):
    return f"sigma={float(value):.3f}".replace("0.", ".")


def save_bi_overlay_for_summaries(out_dir, summaries, filename, label_fn):
    if not summaries:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    first = np.load(summaries[0]["_field_path"])
    ax.plot(first["x_plot"].ravel(), first["bi_true"].ravel(), "k-", lw=2.3, label="True Bi")
    for summary in summaries:
        fields = np.load(summary["_field_path"])
        ax.plot(fields["x_plot"].ravel(), fields["bi_pred"].ravel(), lw=1.9, label=label_fn(summary))
    ax.set_xlabel("x*")
    ax.set_ylabel("Bi_top(x*)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bars_for_summaries(out_dir, summaries, filename, label_fn):
    if not summaries:
        return
    labels = [label_fn(row) for row in summaries]
    metrics = [
        ("bi_rel_l2", "Bi relative L2"),
        ("sensor_theta_y_max_abs", "sensor FD theta_y max abs"),
        ("rel_l2", "temperature relative L2"),
        ("h_center_rel_error_pct", "center h error (%)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.2), dpi=170, constrained_layout=True)
    for ax, (key, title) in zip(axes.ravel(), metrics):
        vals = [float(row.get(key, np.nan)) for row in summaries]
        colors = ["#3b82f6"] * len(vals)
        finite_vals = [(i, value) for i, value in enumerate(vals) if np.isfinite(value)]
        if finite_vals:
            best_idx = min(finite_vals, key=lambda item: item[1])[0]
            colors[best_idx] = "#f97316"
        ax.bar(labels, vals, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_apparent_bi_points_for_summaries(out_dir, summaries, filename, label_fn):
    summaries = [row for row in summaries if (Path(row["run_dir"]) / "noisy_sensor_apparent_bi_overlay_points.csv").exists()]
    if not summaries:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=170)
    first = np.load(summaries[0]["_field_path"])
    ax.plot(first["x_plot"].ravel(), first["bi_true"].ravel(), "k-", lw=2.2, label="True Bi")
    for summary in summaries:
        points_path = Path(summary["run_dir"]) / "noisy_sensor_apparent_bi_overlay_points.csv"
        xs_plot = []
        bi_noisy = []
        with open(points_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs_plot.append(float(row["x"]))
                bi_noisy.append(float(row["bi_app_noisy"]))
        ax.scatter(xs_plot, bi_noisy, s=42, alpha=0.82, edgecolor="white", linewidth=0.5, label=label_fn(summary))
    ax.set_xlabel("x*")
    ax.set_ylabel("apparent Bi from noisy sensors")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.09, 1, 1])
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_summary_table(out_dir, summaries, filename, label_fn):
    if not summaries:
        return
    keys = [
        "label",
        "ablation_name",
        "run_id",
        "noise_level_theta",
        "noise_seed",
        "seed",
        "theta_y_weight",
        "smooth_weight",
        "rel_l2",
        "max_abs",
        "sensor_theta_y_max_abs",
        "bi_rel_l2",
        "h_center_rel_error_pct",
        "run_dir",
    ]
    rows = []
    for summary in summaries:
        rows.append(
            {
                "label": label_fn(summary),
                "ablation_name": summary.get("ablation_name", ""),
                "run_id": summary.get("run_id", ""),
                "noise_level_theta": summary.get("config", {}).get("noise_level_theta", ""),
                "noise_seed": summary.get("config", {}).get("noise_seed", ""),
                "seed": summary.get("config", {}).get("seed", ""),
                "theta_y_weight": summary.get("weights", {}).get("theta_y_data", ""),
                "smooth_weight": summary.get("weights", {}).get("smooth", ""),
                "rel_l2": summary.get("rel_l2", ""),
                "max_abs": summary.get("max_abs", ""),
                "sensor_theta_y_max_abs": summary.get("sensor_theta_y_max_abs", ""),
                "bi_rel_l2": summary.get("bi_rel_l2", ""),
                "h_center_rel_error_pct": summary.get("h_center_rel_error_pct", ""),
                "run_dir": summary.get("run_dir", ""),
            }
        )
    with open(out_dir / filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def copy_today_report_group(source_dir, group_name, filenames):
    destination = TODAY_RESULTS_DIR / group_name
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for filename in filenames:
        source = Path(source_dir) / filename
        if source.exists():
            target = destination / filename
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def write_stage5_noise_level_report(ablation_dir, output_root):
    names = {
        "stage5_noise_0p002_seed2126",
        "stage5_noise_0p005_seed2126",
        "stage5_noise_0p010_seed2126",
        "stage5_noise_0p020_seed2126",
    }
    summaries = collect_completed_runs(output_root, topic="sensor_stage5_noise_level", names=names)
    summaries = latest_by_key(summaries, lambda row: float(row["config"]["noise_level_theta"]))
    summaries.sort(key=lambda row: float(row["config"]["noise_level_theta"]))
    out_dir = ablation_dir / "sensor_stage5_noise_level"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_fn = lambda row: _noise_level_label(row["config"]["noise_level_theta"])
    save_bi_overlay_for_summaries(out_dir, summaries, "noise_level_bi_profile_overlay.png", label_fn)
    save_apparent_bi_points_for_summaries(out_dir, summaries, "noise_level_apparent_bi_points_overlay.png", label_fn)
    save_metrics_bars_for_summaries(out_dir, summaries, "noise_level_metrics_bar.png", label_fn)
    save_summary_table(out_dir, summaries, "noise_level_combined_summary.csv", label_fn)
    copy_today_report_group(
        out_dir,
        "Stage5_Noise_Level_Leftover",
        [
            "noise_level_bi_profile_overlay.png",
            "noise_level_apparent_bi_points_overlay.png",
            "noise_level_metrics_bar.png",
            "noise_level_combined_summary.csv",
        ],
    )


def write_stage5_noise_seed_report(ablation_dir, output_root):
    summaries = collect_completed_runs(output_root, topic="sensor_stage5_noise_seed")
    summaries = latest_by_key(summaries, lambda row: int(row["config"]["noise_seed"]))
    summaries.sort(key=lambda row: int(row["config"]["noise_seed"]))
    out_dir = ablation_dir / "sensor_stage5_noise_seed"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_fn = lambda row: f"noise seed {int(row['config']['noise_seed'])}"
    save_bi_overlay_for_summaries(out_dir, summaries, "noise_seed_bi_profile_overlay.png", label_fn)
    save_apparent_bi_points_for_summaries(out_dir, summaries, "noise_seed_apparent_bi_points_overlay.png", label_fn)
    save_metrics_bars_for_summaries(out_dir, summaries, "noise_seed_metrics_bar.png", label_fn)
    save_summary_table(out_dir, summaries, "noise_seed_combined_summary.csv", label_fn)
    copy_today_report_group(
        out_dir,
        "Stage5_Noise_Seed",
        [
            "noise_seed_bi_profile_overlay.png",
            "noise_seed_apparent_bi_points_overlay.png",
            "noise_seed_metrics_bar.png",
            "noise_seed_combined_summary.csv",
        ],
    )


STAGE4_NESTED_REPORT_CASES = [
    {
        "label": "10 nested",
        "name": "stage4_nested_10sensors_5x2",
        "sensor_x": [0.02, 0.24, 0.50, 0.76, 0.98],
    },
    {
        "label": "14 existing",
        "name": "stage4_14sensors_7x2",
        "sensor_x": [0.02, 0.24, 0.40, 0.50, 0.60, 0.76, 0.98],
    },
    {
        "label": "18 nested",
        "name": "stage4_nested_18sensors_9x2",
        "sensor_x": [0.02, 0.24, 0.34, 0.40, 0.50, 0.60, 0.66, 0.76, 0.98],
    },
    {
        "label": "22 nested",
        "name": "stage4_nested_22sensors_11x2",
        "sensor_x": [0.02, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.98],
    },
    {
        "label": "26 existing final",
        "name": "stage4_26sensors_13x2_baseline",
        "sensor_x": SHOULDER_HEAVY_SENSOR_X,
    },
    {
        "label": "34 nested",
        "name": "stage4_nested_34sensors_17x2_dense",
        "sensor_x": [0.02, 0.08, 0.12, 0.20, 0.24, 0.34, 0.40, 0.46, 0.50, 0.54, 0.60, 0.66, 0.76, 0.80, 0.88, 0.92, 0.98],
    },
]


def save_stage4_nested_layout(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(STAGE4_NESTED_REPORT_CASES),
        1,
        figsize=(12.0, 10.5),
        dpi=170,
        sharex=True,
        constrained_layout=True,
    )
    if len(STAGE4_NESTED_REPORT_CASES) == 1:
        axes = [axes]

    titles = {
        "10 nested": "Nested bare-bone subset - 5x2 sensors (10 total)",
        "14 existing": "Existing sparse subset reused - 7x2 sensors (14 total)",
        "18 nested": "Nested shoulder-refined subset - 9x2 sensors (18 total)",
        "22 nested": "Nested near-baseline subset - 11x2 sensors (22 total)",
        "26 existing final": "Selected final-layout baseline reused - 13x2 sensors (26 total)",
        "34 nested": "Nested dense oversampling reference - 17x2 sensors (34 total)",
    }
    lower_color = "#f97316"
    upper_color = "#2563eb"
    for ax, case in zip(axes, STAGE4_NESTED_REPORT_CASES):
        xs_case = case["sensor_x"]
        ax.scatter(xs_case, [FINAL_SENSOR_Y[0]] * len(xs_case), s=70, color=lower_color, label="lower row")
        ax.scatter(xs_case, [FINAL_SENSOR_Y[1]] * len(xs_case), s=70, color=upper_color, label="upper row")
        ax.axhline(FINAL_SENSOR_Y[0], color="0.35", lw=0.8)
        ax.axhline(FINAL_SENSOR_Y[1], color="0.35", lw=0.8)
        ax.axvline(0.5, color="0.4", lw=0.9, ls="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.973, 1.004)
        ax.set_yticks([0.98, 1.00])
        ax.set_ylabel("y*")
        ax.set_title(titles.get(case["label"], case["label"]), loc="left", fontsize=13)
        ax.grid(axis="x", alpha=0.35)
    axes[-1].set_xlabel("x*")
    axes[-1].set_xticks(np.linspace(0.0, 1.0, 6))
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.savefig(out_dir / "nested_sensor_count_layout.png", bbox_inches="tight")
    fig.savefig(out_dir / "nested_sensor_count_layout_overview.png", bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "nested_sensor_count_layout.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "ablation_name", "n_sensors", "n_x", "sensor_x", "sensor_y", "adds_from_previous"])
        writer.writeheader()
        previous = set()
        for case in STAGE4_NESTED_REPORT_CASES:
            current = set(case["sensor_x"])
            added = sorted(current - previous)
            writer.writerow(
                {
                    "label": case["label"],
                    "ablation_name": case["name"],
                    "n_sensors": len(case["sensor_x"]) * len(FINAL_SENSOR_Y),
                    "n_x": len(case["sensor_x"]),
                    "sensor_x": " ".join(f"{x:.2f}" for x in case["sensor_x"]),
                    "sensor_y": " ".join(f"{y:.2f}" for y in FINAL_SENSOR_Y),
                    "adds_from_previous": " ".join(f"{x:.2f}" for x in added),
                }
            )
            previous = current


def save_stage4_nested_trend(out_dir, summaries, label_fn):
    if not summaries:
        return
    x_counts = [int(row["sensor_count"]) for row in summaries]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9), dpi=170, constrained_layout=True)
    axes[0].plot(x_counts, [float(row["bi_rel_l2"]) for row in summaries], "o-", color="#4e79a7")
    axes[0].set_title("Bi profile error")
    axes[0].set_ylabel("Bi rel L2")
    axes[1].plot(x_counts, [float(row["rel_l2"]) for row in summaries], "o-", color="#f28e2b")
    axes[1].set_title("Temperature field error")
    axes[1].set_ylabel("theta rel L2")
    for ax in axes:
        ax.set_xlabel("sensor count")
        ax.set_xticks(x_counts)
        ax.grid(True, alpha=0.25)
    fig.savefig(out_dir / "nested_sensor_count_trend.png", bbox_inches="tight")
    plt.close(fig)


def write_stage4_nested_sensor_count_report(ablation_dir, output_root):
    all_rows = collect_completed_runs(output_root)
    by_name = {row["ablation_name"]: row for row in all_rows}
    summaries = []
    for case in STAGE4_NESTED_REPORT_CASES:
        row = by_name.get(case["name"])
        if row is not None:
            summaries.append(row)

    summaries.sort(key=lambda row: int(row["sensor_count"]))
    out_dir = ablation_dir / "sensor_stage4_nested_sensor_count"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_by_name = {case["name"]: case["label"] for case in STAGE4_NESTED_REPORT_CASES}
    label_fn = lambda row: label_by_name.get(row["ablation_name"], row["ablation_name"])

    save_stage4_nested_layout(out_dir)
    save_bi_overlay_for_summaries(out_dir, summaries, "nested_sensor_count_bi_profile_overlay.png", label_fn)
    save_metrics_bars_for_summaries(out_dir, summaries, "nested_sensor_count_metric_bars.png", label_fn)
    save_stage4_nested_trend(out_dir, summaries, label_fn)
    save_summary_table(out_dir, summaries, "nested_sensor_count_combined_summary.csv", label_fn)

    completed_names = {row["ablation_name"] for row in summaries}
    with open(out_dir / "nested_sensor_count_missing_cases.txt", "w") as f:
        missing = [case["name"] for case in STAGE4_NESTED_REPORT_CASES if case["name"] not in completed_names]
        if missing:
            f.write("Missing cases:\n")
            for name in missing:
                f.write(f"- {name}\n")
        else:
            f.write("All nested Stage 4 sensor-count cases are available.\n")

    copy_today_report_group(
        out_dir,
        "Stage4_Nested_Sensor_Count",
        [
            "nested_sensor_count_layout.png",
            "nested_sensor_count_layout_overview.png",
            "nested_sensor_count_layout.csv",
            "nested_sensor_count_bi_profile_overlay.png",
            "nested_sensor_count_metric_bars.png",
            "nested_sensor_count_trend.png",
            "nested_sensor_count_combined_summary.csv",
            "nested_sensor_count_missing_cases.txt",
        ],
    )


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=[
            "default",
            "all_topics",
            "depth",
            "node_count",
            "activation",
            "sensor",
            "sensor_stage1",
            "sensor_stage2",
            "sensor_stage2_fd",
            "sensor_stage3",
            "sensor_stage3_fd",
            "sensor_stage4",
            "sensor_stage4_fd",
            "sensor_stage4_nested",
            "sensor_stage4_nested_fd",
            "sensor_placement_update",
            "sensor_placement_update_fd",
            "sensor_stage5",
            "sensor_stage5_noise",
            "sensor_stage5_noise_fd",
            "sensor_stage5_noise_continue",
            "sensor_stage5_noise_continue_fd",
            "sensor_stage5_noise_seed",
            "sensor_stage5_noise_seed_fd",
            "sensor_stage5_2",
            "sensor_stage5_regularization",
            "sensor_stage5_regularization_fd",
            "sensor_stage5_3",
            "sensor_stage5_local_linear",
            "sensor_stage5_local_linear_fd",
            "sensor_stage5_local_linear_4row",
            "sensor_stage5_4",
            "sensor_stage5_local_linear_tune",
            "sensor_stage5_local_linear_4row_tune",
            "sensor_stage5_local_linear_4row_tune_0p002",
            "final_seed_robustness",
            "final_seed_repeat",
            "final_seed_ablation",
            "sensor_stage1_stage2",
            "sensor_stage12",
            "legacy",
            "quick",
            "smoke",
            "bi_best_weights",
            "bi_capacity",
        ],
        default="default",
    )
    parser.add_argument("--only", nargs="*", help="Run only ablations whose name contains one of these strings.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--n-adam", type=int, default=None, help="Override Adam iterations for every run.")
    parser.add_argument("--no-lbfgs", action="store_true", help="Skip L-BFGS for every run.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, cpu, cuda, cuda:0, cuda:1, ...",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    DEVICE = resolve_device(args.device)

    suite = select_suite(args.suite)
    if args.only:
        suite = [spec for spec in suite if any(token in spec["name"] for token in args.only)]
    if args.max_runs is not None:
        suite = suite[: args.max_runs]
    if not suite:
        raise ValueError("No ablation specs selected.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    ablation_dir = args.output_root / time.strftime(f"ablation_%Y%m%d_%H%M%S_{args.suite}")
    ablation_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "requested_device": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(DEVICE) if DEVICE.type == "cuda" else None,
        "ref_path": str(REF_PATH),
        "suite": args.suite,
        "topics": sorted({spec.get("topic", "general") for spec in suite}),
        "specs": suite,
        "default_config": DEFAULT_CONFIG,
        "default_weights": DEFAULT_WEIGHTS,
        "default_bi_model_config": DEFAULT_BI_MODEL_CONFIG,
        "default_optimizer_config": DEFAULT_OPTIMIZER_CONFIG,
        "sensor_ablation_roadmap": SENSOR_ABLATION_ROADMAP,
    }
    with open(ablation_dir / "ablation_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"ablation_dir: {ablation_dir}", flush=True)
    print(f"selected runs: {[(spec.get('topic', 'general'), spec['name']) for spec in suite]}", flush=True)

    results = []
    for index, spec in enumerate(suite, start=1):
        print(f"\n[{index}/{len(suite)}] queued: {spec.get('topic', 'general')} / {spec['name']}", flush=True)
        try:
            result = run_one_experiment(spec, ablation_dir, args)
            results.append(result)
            write_ablation_summary(ablation_dir, results)
            write_topic_summaries(ablation_dir, results)
        except Exception as exc:
            error = {
                "ablation_topic": slugify(spec.get("topic", "general")),
                "ablation_name": slugify(spec["name"]),
                "description": spec.get("description", ""),
                "error": repr(exc),
            }
            results.append(error)
            write_ablation_summary(ablation_dir, results)
            write_topic_summaries(ablation_dir, results)
            raise

    if args.suite in {"sensor_stage5", "sensor_stage5_noise", "sensor_stage5_noise_fd", "sensor_stage5_noise_continue", "sensor_stage5_noise_continue_fd"}:
        write_stage5_noise_level_report(ablation_dir, args.output_root)
    if args.suite in {"sensor_stage5_noise_seed", "sensor_stage5_noise_seed_fd"}:
        write_stage5_noise_seed_report(ablation_dir, args.output_root)
    if args.suite in {"sensor_stage4_nested", "sensor_stage4_nested_fd", "sensor_placement_update", "sensor_placement_update_fd"}:
        write_stage4_nested_sensor_count_report(ablation_dir, args.output_root)

    print("\nAblation finished.", flush=True)
    print(f"summary: {ablation_dir / 'ablation_results.csv'}", flush=True)


if __name__ == "__main__":
    main()
