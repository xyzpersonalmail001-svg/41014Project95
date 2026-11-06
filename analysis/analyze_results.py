"""Recompute saved evaluation metrics and figures without running the robot.

Only NumPy and Matplotlib are needed. Inputs are CSV subsets and experiment
metadata under evaluation/, grouped by assessment parts A to E.
Positions remain in metres and angles in radians until a plot converts them.
This entry does not execute an EKF replay, regenerate images or audit collisions.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import uuid

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "project95-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODES = ("feedforward_only", "wheel_PI_only", "pose_feedback")
COLORS = ("#777777", "#d48920", "#078e9c")
PARTS = {"A": "A_Modelling", "B": "B_Sensing", "C": "C_Estimation",
         "D": "D_Control", "E": "E_Integration_Robustness"}


def records(path):
    with path.open(encoding="utf-8", newline="") as stream:
        result = list(csv.DictReader(stream))
    if not result:
        raise ValueError(f"No data rows: {path}")
    return result


def array(rows, *keys):
    values = np.asarray([[float(row[k]) for k in keys] for row in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {keys}")
    return values[:, 0] if len(keys) == 1 else values


def pose(rows, prefix):
    return array(rows, prefix + "_x_m", prefix + "_y_m", prefix + "_yaw_rad")


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def errors(first, second):
    error = first - second
    error[:, 2] = wrap(error[:, 2])
    return error


def metrics(error):
    return {
        "position_rmse_m": float(np.sqrt(np.mean(np.sum(error[:, :2] ** 2, axis=1)))),
        "yaw_rmse_degrees": float(np.degrees(np.sqrt(np.mean(error[:, 2] ** 2)))),
        "final_position_error_m": float(np.linalg.norm(error[-1, :2])),
        "final_yaw_error_degrees": float(np.degrees(error[-1, 2])),
    }


def verify(actual, saved, keys=None):
    """Check computed values against the preserved original evaluation."""
    for key in keys or actual.keys():
        if not np.isclose(actual[key], saved[key], rtol=1e-9, atol=1e-10):
            raise ValueError(f"Saved metric differs: {key}: {actual[key]} vs {saved[key]}")


def save(fig, output, name):
    fig.savefig(output / name, dpi=160)
    plt.close(fig)


def ekf_analysis(data, meta, output):
    rows = records(data / "ekf.csv")
    time = array(rows, "elapsed_s")
    truth = pose(rows, "truth")
    error = errors(pose(rows, "ekf"), truth)
    baseline = errors(pose(rows, "encoder"), truth)
    result = {"ekf": metrics(error), "encoder_only": metrics(baseline)}
    for mode in result:
        verify(result[mode], meta["saved_metrics"][mode])
    fig, axes = plt.subplots(3, 1, figsize=(9, 4.8), sharex=True, layout="constrained")
    for i, (axis, label, scale) in enumerate(zip(axes,
            ("x error (mm)", "y error (mm)", "Yaw error (deg)"), (1000, 1000, 180 / np.pi))):
        variance = array(rows, f"P{i}{i}")
        if (variance < 0).any():
            raise ValueError("Negative covariance diagonal")
        band = 2 * np.sqrt(variance) * scale
        axis.fill_between(time, -band, band, color=COLORS[2], alpha=.2,
                          label="EKF +/-2 model std (uncalibrated)")
        axis.plot(time, error[:, i] * scale, color=COLORS[2], label="EKF error")
        axis.plot(time, baseline[:, i] * scale, "--", color=COLORS[1], label="Encoder-only error")
        axis.axvspan(4, 7, color="gray", alpha=.12)
        axis.axvspan(7, 10, color="#dc9279", alpha=.18)
        axis.set_ylabel(label)
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Estimator module: pair / single message / physical occlusion / reacquired")
    axes[-1].set_xlabel("Elapsed simulation time after initialization (s)")
    save(fig, output, "ekf_errors_uncertainty.png")
    return result


def tracking_analysis(data, meta, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    result = {}
    for mode, color in zip(MODES, COLORS):
        rows = records(data / f"tracking_{mode}.csv")
        truth, reference = pose(rows, "truth"), pose(rows, "reference")
        error = errors(truth, reference)
        result[mode] = metrics(error)
        saved = meta["saved_metrics"][mode]
        expected = {"position_rmse_m": saved["position_tracking_rmse_m"],
                    "yaw_rmse_degrees": saved["yaw_tracking_rmse_degrees"],
                    "final_position_error_m": saved["final_position_error_m"],
                    "final_yaw_error_degrees": saved["final_yaw_error_degrees"]}
        verify(result[mode], expected)
        axes[0].plot(truth[:, 0], truth[:, 1], color=color, label=mode)
        axes[1].plot(array(rows, "elapsed_s"), np.linalg.norm(error[:, :2], axis=1) * 1000,
                     color=color, label=mode)
    axes[0].plot(reference[:, 0], reference[:, 1], "k--", label="Time-indexed reference")
    axes[0].set(xlabel="World x (m)", ylabel="World y (m)", title="Module trajectories; same start and seed")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[1].set(xlabel="Elapsed simulation time (s)", ylabel="Position tracking error (mm)",
                title="Truth versus reference at the same time")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    save(fig, output, "tracking_comparison.png")
    return result


def camera_analysis(data, meta, output):
    rows = records(data / "camera.csv")
    error = errors(pose(rows, "image"), pose(rows, "truth"))
    separation = array(rows, "station_x_separation_m")
    result = []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for saved in meta["camera_saved_summary"]:
        distance = saved["station_x_separation_m"]
        selected = np.isclose(separation, distance)
        selected_error = error[selected]
        metric = {"station_x_separation_m": distance, "valid": int(selected.sum()),
                  "position_RMSE_mm": metrics(selected_error)["position_rmse_m"] * 1000,
                  "max_position_error_mm": float(np.linalg.norm(selected_error[:, :2], axis=1).max() * 1000),
                  "max_abs_yaw_error_degrees": float(np.degrees(np.abs(selected_error[:, 2])).max())}
        verify(metric, saved)
        result.append(metric)
        axes[0].scatter(separation[selected], np.linalg.norm(selected_error[:, :2], axis=1) * 1000, s=22)
        axes[1].scatter(separation[selected], np.degrees(np.abs(selected_error[:, 2])), s=22)
    axes[0].set(ylabel="Position error norm (mm)", title="36 fixed rendered-image fixtures")
    axes[1].set(ylabel="Absolute yaw error (degrees)", title="View and pixel-quantization effects")
    for axis in axes:
        axis.set_xlabel("World x separation from station (m)")
        axis.grid(alpha=.2)
    save(fig, output, "image_error_vs_distance.png")
    return result


def filter_analysis(data, output):
    # These are saved conditional-replay summaries, not new filter executions.
    rows = records(data / "filter_replay.csv")
    variants = ("baseline", "Q_model_quarter", "Q_model_four", "R_quarter", "R_four")
    labels = ("Nominal", "Qmodel /4", "Qmodel x4", "R /4", "R x4")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout="constrained")
    for axis, trial in zip(axes[:2], ("T08", "T01")):
        selected = [next(r for r in rows if r["trial"] == trial and r["variant"] == v) for v in variants]
        axis.bar(labels, array(selected, "position_RMSE_m") * 1000, color=[COLORS[2]] + ["#aabcc5"] * 4)
        axis.set(title=f"{trial}: fixed-input position RMSE", ylabel="mm")
    selected = [next(r for r in rows if r["trial"] == "T08" and r["variant"] == v) for v in variants]
    axes[2].bar(labels, array(selected, "NIS_rejected_frames"), color="#bf9148")
    axes[2].set(title="T08: NIS-rejected frames", ylabel="Frames")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
    save(fig, output, "noise_weight_contrasts.png")
    return {"saved_summary_rows": len(rows), "scope": "Replot only; no estimator replay or closed-loop task is executed."}


def gain_analysis(data, recovery_data, meta, output):
    rows = records(data / "gain_metrics.csv")
    goal = np.asarray(meta["goal_pose_m_m_rad"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    result = []
    for row in rows:
        variant = row["variant"]
        if variant == "nominal":
            # The nominal gain case is also the E recovery development case.
            # Read its single shared record rather than storing a duplicate.
            trajectory = records(recovery_data / "recovery_v2.csv")
            actual = pose(trajectory, "truth")
        else:
            trajectory = records(data / f"gain_{variant}.csv")
            actual = array(trajectory, "x_m", "y_m", "yaw_rad")
        time = array(trajectory, "elapsed_s")
        handoff = float(row["handoff_elapsed_s"])
        error = errors(actual, np.broadcast_to(goal, actual.shape))
        final = metrics(error)
        value = {"terminal_seconds": float(time[-1] - handoff),
                 "final_position_error_m": final["final_position_error_m"],
                 "final_yaw_error_degrees": final["final_yaw_error_degrees"]}
        verify(value, {k: float(row[k]) for k in value})
        result.append({"variant": variant, "k_rho_per_s": float(row["k_rho_per_s"]), **value})
        selected = time >= handoff - 1e-7
        axes[0].plot(time[selected] - handoff, np.linalg.norm(error[selected, :2], axis=1) * 1000,
                     label=f"k_rho={row['k_rho_per_s']}/s")
    axes[0].axhline(25, color="gray", linestyle=":", label="Project limit 25 mm")
    axes[0].set(xlabel="Time since handoff (s)", ylabel="True distance to final goal (mm)",
                title="Existing terminal-gain development comparison")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=.2)
    axes[1].bar([str(r["k_rho_per_s"]) for r in result], [r["terminal_seconds"] for r in result],
                color=["#aabcc5", COLORS[2], "#aabcc5"])
    axes[1].set(xlabel="k_rho (1/s)", ylabel="Terminal seconds", title="Gain versus completion time")
    save(fig, output, "gain_contrast.png")
    return result


def recovery_analysis(data, meta, output):
    goal = np.asarray(meta["goal_pose_m_m_rad"])
    result = {}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.1), layout="constrained")
    for name, color in zip(("original", "v1", "v2"), COLORS):
        rows = records(data / f"recovery_{name}.csv")
        time = array(rows, "elapsed_s")
        truth, estimate = pose(rows, "truth"), pose(rows, "estimate")
        localization = metrics(errors(estimate, truth))
        goal_error = metrics(errors(truth, np.broadcast_to(goal, truth.shape)))
        value = {"position_localization_rmse_m": localization["position_rmse_m"],
                 "final_position_error_m": goal_error["final_position_error_m"],
                 "final_yaw_error_degrees": goal_error["final_yaw_error_degrees"],
                 "elapsed_seconds": float(time[-1]),
                 "min_signed_geom_clearance_m": float(array(rows, "signed_geom_clearance_m").min())}
        verify(value, meta["cases"][name]["saved_metrics"])
        result[name] = {"outcome": meta["cases"][name]["outcome"], **value}
        label = name + ": " + result[name]["outcome"]
        axes[0, 0].plot(truth[:, 0], truth[:, 1], color=color, label=label)
        axes[0, 1].plot(time, np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1) * 1000, color=color)
        axes[1, 0].plot(time, array(rows, "signed_geom_clearance_m"), color=color, label=label)
    axes[0, 0].plot(estimate[:, 0], estimate[:, 1], "--", color=COLORS[2], linewidth=.9, label="v2 EKF")
    axes[0, 0].scatter([goal[0]], [goal[1]], color="red", marker="*", s=90, label="Final goal")
    axes[0, 0].set(xlabel="World x (m)", ylabel="World y (m)", title="Development trajectories; truth for evaluation")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set(xlabel="Elapsed simulation time (s)", ylabel="Position localization error (mm)", title="Estimate versus truth")
    axes[1, 0].axhline(.01, color="red", linestyle=":", label="Project clearance limit 0.01 m")
    axes[1, 0].set(xlabel="Elapsed simulation time (s)", ylabel="Signed geometry clearance (m)", title="Sampled at 10 Hz and capped at 0.5 m")
    axes[1, 0].legend(fontsize=7)
    controls = records(data / "recovery_v2_controls.csv")
    axes[1, 1].plot(array(controls, "elapsed_s"), array(controls, "command_v_m_s"), color=COLORS[2])
    axes[1, 1].set(xlabel="Elapsed simulation time (s)", ylabel="Command v (m/s)", title="v2 outer command; shade marks recovery")
    for event in meta["events"]:
        if event["event"] == "attempt_started":
            end = next(e["elapsed_s"] for e in meta["events"] if e["event"] == "confirmed_oriented_pose_recovery" and e["elapsed_s"] > event["elapsed_s"])
            axes[1, 1].axvspan(event["elapsed_s"], end, color="#995999", alpha=.2)
        else:
            axes[0, 1].axvline(event["elapsed_s"], color="#995999", linestyle=":", linewidth=1)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    save(fig, output, "recovery_comparison.png")
    return result


def batch_analysis(path, meta, output):
    rows = records(path)
    protocol = meta["protocol"]
    conditions = [r["id"] for r in protocol["conditions"]]
    starts = [r["id"] for r in protocol["starts"]]
    n = protocol["replicates_per_cell"]
    expected_total = len(conditions) * len(starts) * n
    if len(rows) != expected_total or len({r["trial_id"] for r in rows}) != len(rows):
        raise ValueError("Missing or duplicate first attempts")
    seeds = [int(r["encoder_seed"]) for r in rows]
    if len(set(seeds)) != expected_total or set(seeds) != set(range(protocol["seed_start"], protocol["seed_start"] + expected_total)):
        raise ValueError("Trial seeds differ from the preserved protocol")
    matrix = np.zeros((len(conditions), len(starts)))
    cells = []
    for i, condition in enumerate(conditions):
        for j, start in enumerate(starts):
            selected = [r for r in rows if r["condition_id"] == condition and r["start_id"] == start]
            if len(selected) != n:
                raise ValueError("Incomplete condition/start cell")
            success = sum(r["outcome"] == "success" for r in selected)
            matrix[i, j] = success / n
            cells.append({"condition_id": condition, "start_id": start, "successes": success, "attempted": n})
    successes = sum(r["outcome"] == "success" for r in rows)
    verify({"attempted": len(rows), "successes": successes}, meta["saved_overall"])
    result = {"attempted": len(rows), "successes": successes, "cells": cells,
              "final_position_min_m": float(array(rows, "final_position_error_m").min()),
              "final_position_max_m": float(array(rows, "final_position_error_m").max()),
              "max_abs_final_yaw_degrees": float(np.abs(array(rows, "final_yaw_error_degrees")).max()),
              "elapsed_min_seconds": float(array(rows, "elapsed_seconds").min()),
              "elapsed_max_seconds": float(array(rows, "elapsed_seconds").max()),
              "max_localization_rmse_m": float(array(rows, "position_localization_rmse_m").max()),
              "saved_obstacle_contact_steps_sum": int(array(rows, "obstacle_contact_physics_steps").sum()),
              "scope": meta["scope"]}
    fig, axis = plt.subplots(figsize=(6.8, 3.5), layout="constrained")
    axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    axis.set_xticks(range(len(starts)), starts)
    axis.set_yticks(range(len(conditions)), conditions)
    for i in range(len(conditions)):
        for j in range(len(starts)):
            axis.text(j, i, f"{int(matrix[i,j]*n)}/{n}", ha="center", va="center")
    axis.set_title("R04 saved first attempts; finite matrix only")
    save(fig, output, "robustness_matrix.png")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "evaluation",
                        help="Directory containing the A-E result folders and experiments.json")
    parser.add_argument("--trials", type=Path,
                        help="Override the E_Integration_Robustness/data/trial_metrics.csv input")
    parser.add_argument("--output", type=Path, help="New output directory; an existing directory is never overwritten")
    args = parser.parse_args()
    data = args.data.resolve()
    meta = json.loads((data / "experiments.json").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = (args.output or ROOT / "results/analysis" / run_id).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = {part: data / name / "data" for part, name in PARTS.items()}
    figures = {part: output / PARTS[part] / "figures" for part in "BCDE"}
    for folder in figures.values():
        folder.mkdir(parents=True)
    trials = args.trials or inputs["E"] / "trial_metrics.csv"
    result = {"ekf": ekf_analysis(inputs["C"], meta["ekf"], figures["C"]),
              "tracking": tracking_analysis(inputs["D"], meta["tracking"], figures["D"]),
              "camera": camera_analysis(inputs["B"], meta["parameters"], figures["B"]),
              "filter_replay": filter_analysis(inputs["C"], figures["C"]),
              "gain": gain_analysis(inputs["D"], inputs["E"], meta["parameters"], figures["D"]),
              "recovery": recovery_analysis(inputs["E"], meta["recovery"], figures["E"]),
              "final_batch": batch_analysis(trials, meta["final_batch"], figures["E"]),
              "scope": "Offline reanalysis of saved data. No new experiments, reliability samples or robot algorithm changes."}
    (output / "evaluation_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("ANALYSIS_OUTPUT", str(output))
    print("ANALYSIS_PASSED: saved metrics reproduced; seven figures generated")


if __name__ == "__main__":
    main()
