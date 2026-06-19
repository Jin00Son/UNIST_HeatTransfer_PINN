# FD `theta_y` Label Motivation From Loss History

This note summarizes the objective evidence used to justify adding the finite-difference `theta_y` label to the inverse CHTC PINN. The key point is not simply that the Bi result was poor. The stronger evidence is that the temperature data loss and the top Robin residual had already converged to very small values, while the recovered Bi profile remained inaccurate.

## 1. Physical Identifiability Issue

The unknown Bi profile appears in the top Robin boundary condition through the normal temperature gradient:

```text
-theta_y(x, 1) = Bi(x) theta(x, 1)
```

Therefore,

```text
Bi(x) = -theta_y(x, 1) / theta(x, 1)
```

If the training data only constrains temperature values, the PINN can still satisfy the top residual using a compensating combination of an internally generated `theta_y` and an incorrect `Bi(x)`. In that case, small temperature loss does not guarantee correct Bi recovery.

## 2. Loss History Evidence Before Adding `theta_y_data`

The following runs were performed before the FD `theta_y` data term was introduced. Their loss histories contain only:

```text
epoch,total,pde,bottom,top,side,data
```

There is no `theta_y_data` column.

| Run | Sensors | Final data loss | Final top loss | Full-field `theta_rel_l2` | `Bi_rel_l2` |
|---|---:|---:|---:|---:|---:|
| `run_20260521_210302` | 16 | `1.29e-6` | `4.58e-7` | `3.72e-4` | `9.68e-2` |
| `run_20260521_212149` | 16 | `1.28e-6` | `5.41e-7` | `3.64e-4` | `9.65e-2` |
| `run_20260521_214104` | 48 | `8.95e-7` | `6.20e-7` | `2.87e-4` | `9.41e-2` |
| `run_20260521_221030` | 48 | `1.59e-6` | `6.41e-7` | `2.98e-4` | `9.39e-2` |
| `run_20260523_212937` | 48 | `3.00e-6` | `6.84e-7` | `2.88e-4` | `9.36e-2` |
| `run_20260523_215713` | 48 | `4.20e-6` | `1.69e-6` | `3.86e-4` | `9.90e-2` |

These results show that the temperature observations and the Robin residual were already being minimized well. However, the Bi relative error stayed around 9-10%. This indicates that the failure was not mainly a temperature fitting failure; it was an inverse identifiability problem.

## 3. Why More Temperature Sensors Were Not Enough

The evidence is clearer when comparing the 16-sensor and 48-sensor cases:

| Run | Sensors | `theta_rel_l2` | `Bi_rel_l2` |
|---|---:|---:|---:|
| `run_20260521_212149` | 16 | `3.64e-4` | `9.65e-2` |
| `run_20260523_212937` | 48 | `2.88e-4` | `9.36e-2` |

Increasing the number of temperature sensors improved the temperature field slightly, but the Bi error barely changed. This suggests that simply adding more `theta` observations did not provide the missing boundary-flux information. Since Bi is directly tied to `theta_y` at the top boundary, the next logical step was to add a normal-gradient label.

## 4. Result After Adding FD `theta_y_data`

The first runs with the additional loss term contain:

```text
epoch,total,pde,bottom,top,side,data,theta_y_data
```

| Run | `theta_y_data` | Sensors | Final data loss | Final `theta_y_data` loss | `theta_rel_l2` | `Bi_rel_l2` |
|---|---:|---:|---:|---:|---:|---:|
| `run_20260523_223359` | yes | 16 | `1.08e-7` | `1.88e-4` | `5.70e-4` | `7.89e-2` |
| `run_20260523_230354` | yes | 16 | `1.03e-7` | `6.85e-5` | `1.89e-4` | `3.77e-2` |

This was not a perfectly isolated one-variable ablation because the sensor placement also changed. However, the sequence is still useful as engineering evidence: after introducing a direct normal-gradient constraint, the Bi error decreased from roughly 9-10% to 3.77%.

## 5. Report-Ready Interpretation

Before adding the finite-difference `theta_y` label, the PINN already achieved small temperature data loss and small top Robin residual. For example, `run_20260523_212937` reached a final data loss of `3.00e-6`, a top loss of `6.84e-7`, and a full-field temperature relative L2 error of `2.88e-4`. Despite this, the recovered Bi profile still had a relative L2 error of `9.36e-2`.

This means the model could fit the temperature observations and satisfy the Robin equation numerically while still recovering an incorrect Bi profile. Since the top Robin condition directly relates Bi to the normal temperature gradient, an additional finite-difference `theta_y` label was introduced to supply boundary-flux information and improve inverse identifiability.

## 6. Files Collected Here

The loss history CSV files used for this analysis are stored in:

```text
loss_history_evidence/
```

The compact summary table is stored as:

```text
theta_y_label_motivation_summary.csv
```

## 7. Word Figure Extraction and Source Matching

The Word file `20221225_손진우_Phase3_Final.docx` contained 19 visualizations. They are collected in:

```text
final_report_images/
```

Most figures were matched to exact project images by SHA-256 hash. Two energy-balance figures were matched semantically to their project originals because the image binaries in the Word file did not exactly match the saved project files.

| Word order | Output image | Match type |
|---:|---|---|
| 01 | `word_visual_01_field_comparison_final.png` | exact hash |
| 02 | `word_visual_02_local_pde_residual_heatmap.png` | semantic original |
| 03 | `word_visual_03_global_energy_balance_bar.png` | semantic original |
| 04 | `word_visual_04_final_sensor_layout.png` | exact hash |
| 05 | `word_visual_05_final_bi_profile.png` | exact hash |
| 06 | `word_visual_06_stage2_sensor_layouts.png` | exact hash |
| 07 | `word_visual_07_stage2_bi_overlay.png` | exact hash |
| 08 | `word_visual_08_stage2_metric_bars.png` | exact hash |
| 09 | `word_visual_09_stage3_sensor_layouts.png` | exact hash |
| 10 | `word_visual_10_stage3_bi_overlay.png` | exact hash |
| 11 | `word_visual_11_stage3_metric_bars.png` | exact hash |
| 12 | `word_visual_12_stage4_sensor_count_layouts.png` | exact hash |
| 13 | `word_visual_13_stage4_bi_overlay.png` | exact hash |
| 14 | `word_visual_14_stage4_metric_bars.png` | exact hash |
| 15 | `word_visual_15_sensor_noise.png` | exact hash |
| 16 | `word_visual_16_noise_apparent_bi_points.png` | exact hash |
| 17 | `word_visual_17_noise_bi_profile_overlay.png` | exact hash |
| 18 | `word_visual_18_noise_metric_bars.png` | exact hash |
| 19 | `word_visual_19_noisy_sensor_apparent_bi_overlay.png` | exact hash |

The detailed source mapping is stored as:

```text
final_report_image_source_index.tsv
```
