"""第二阶段第一步：固定命令运动、理想轮关节采样及独立真实状态评估。"""

# Permit both package imports and direct execution of this existing driver.
import sys as _project_sys
from pathlib import Path as _ProjectPath
if __package__ in (None, ""):
    _project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[3]))


import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import uuid

import mujoco
import numpy as np
from evaluation.common.main import ROOT, load_model, validate_state


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def wheel_indices(model):
    """按名称查询，避免模型修改后固定下标失效。"""
    return {name: {
        "actuator": model.actuator(name).id,
        "angle": int(model.jnt_qposadr[model.joint(name).id]),
        "speed": int(model.jnt_dofadr[model.joint(name).id]),
    } for name in ["wheel_left", "wheel_right"]}


def read_wheels(data, indices):
    """只读取轮关节；不读取底盘真实位置。暂未添加编码器噪声或量化。"""
    return {f"{side}_{kind}": float(array[indices[name][address]])
            for side, name in [("left", "wheel_left"), ("right", "wheel_right")]
            for kind, array, address in [("angle_rad", data.qpos, "angle"),
                                         ("speed_rad_s", data.qvel, "speed")]}


def read_truth_for_evaluation(model, data):
    """真实机身状态只供本文件的日志与验收计算，不参与命令生成。"""
    base_id = model.body("base").id
    rotation = data.xmat[base_id].reshape(3, 3)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, velocity, 0)
    return {"x_m": float(data.xpos[base_id, 0]), "y_m": float(data.xpos[base_id, 1]),
            "z_m": float(data.xpos[base_id, 2]),
            "yaw_rad": math.atan2(rotation[1, 0], rotation[0, 0]),
            "body_speed_m_s": float(np.linalg.norm(velocity[3:5])),
            "yaw_rate_rad_s": float(velocity[2]),
            "tilt_rad": math.acos(float(np.clip(rotation[2, 2], -1, 1)))}


def evaluate_phase(phase, start, wheels, truth, config):
    limits = config["acceptance"]
    tail = [i for i, row in enumerate(wheels)
            if row["phase_seconds"] >= phase["seconds"] - config["tail_window_seconds"] - 1e-8]
    if not tail:
        raise RuntimeError("No samples in the phase evaluation window.")
    mean_left = float(np.mean([wheels[i]["left_speed_rad_s"] for i in tail]))
    mean_right = float(np.mean([wheels[i]["right_speed_rad_s"] for i in tail]))
    end = truth[-1]
    dx, dy = end["x_m"] - start["x_m"], end["y_m"] - start["y_m"]
    forward = dx * math.cos(start["yaw_rad"]) + dy * math.sin(start["yaw_rad"])
    yaw = np.unwrap([start["yaw_rad"]] + [row["yaw_rad"] for row in truth])
    yaw_change = float(yaw[-1] - yaw[0])
    max_wheel = max(abs(wheels[i][key]) for i in tail
                    for key in ["left_speed_rad_s", "right_speed_rad_s"])
    max_body = max(truth[i]["body_speed_m_s"] for i in tail)
    checks = {"base_remains_upright": max(row["tilt_rad"] for row in truth) < limits["max_base_tilt_rad"]}
    expected = phase["expected"]
    if expected == "stop":
        checks.update(wheels_settled=max_wheel <= limits["stop_wheel_speed_rad_s"],
                      body_settled=max_body <= limits["stop_body_speed_m_s"])
    else:
        checks["left_wheel_direction"] = mean_left * math.copysign(1, phase["left_rad_s"]) > limits["min_signed_wheel_speed_rad_s"]
        checks["right_wheel_direction"] = mean_right * math.copysign(1, phase["right_rad_s"]) > limits["min_signed_wheel_speed_rad_s"]
        if expected == "forward":
            checks.update(moves_forward=forward >= limits["min_forward_distance_m"],
                          heading_stays_near_start=abs(yaw_change) <= limits["max_forward_heading_change_rad"])
        else:
            checks["turn_direction_and_angle"] = yaw_change * (1 if expected == "left" else -1) >= limits["min_turn_angle_rad"]
    result = {"phase": phase["name"], "checks": checks, "passed": all(checks.values()),
              "command_left_rad_s": phase["left_rad_s"], "command_right_rad_s": phase["right_rad_s"],
              "tail_mean_left_rad_s": mean_left, "tail_mean_right_rad_s": mean_right,
              "tail_left_command_error_rad_s": phase["left_rad_s"] - mean_left,
              "tail_right_command_error_rad_s": phase["right_rad_s"] - mean_right,
              "forward_displacement_m": forward, "planar_displacement_m": math.hypot(dx, dy),
              "yaw_change_rad": yaw_change, "yaw_change_degrees": math.degrees(yaw_change),
              "tail_max_abs_wheel_speed_rad_s": max_wheel, "tail_max_body_speed_m_s": max_body}
    if expected == "stop":
        stopped = np.array([max(abs(w["left_speed_rad_s"]), abs(w["right_speed_rad_s"])) <= limits["stop_wheel_speed_rad_s"]
                            and t["body_speed_m_s"] <= limits["stop_body_speed_m_s"]
                            for w, t in zip(wheels, truth)])
        remains_stopped = np.logical_and.accumulate(stopped[::-1])[::-1]
        candidates = np.flatnonzero(remains_stopped)
        result["observed_settling_time_s"] = wheels[int(candidates[0])]["phase_seconds"] if len(candidates) else None
    return result


def plot_results(output, wheels, truth):
    # 图表从已保存的采样记录生成，不使用示意数据。
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [row["time_s"] for row in wheels]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for axis, side in zip(axes[:2], ["left", "right"]):
        axis.step(t, [row[f"command_{side}_rad_s"] for row in wheels], where="post", label="Command", color="#777777", linestyle="--")
        axis.plot(t, [row[f"{side}_speed_rad_s"] for row in wheels], label="Actual joint speed", color="#126b9c")
        axis.set_ylabel(f"{side.title()} wheel (rad/s)")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[2].plot(t, np.degrees(np.unwrap([row["yaw_rad"] for row in truth])), color="#126b9c")
    axes[2].set(ylabel="True yaw (deg)\nevaluation only", xlabel="Simulation time (s)")
    axes[2].grid(alpha=0.2)
    for i, row in enumerate(wheels):
        if i == 0 or row["phase"] != wheels[i - 1]["phase"]:
            for axis in axes:
                axis.axvline(row["time_s"], color="#aaaaaa", linewidth=0.6)
            axes[0].text(row["time_s"] + 0.04, 1.01, row["phase"].replace("stop_after_", "stop_").replace("_", " "),
                         transform=axes[0].get_xaxis_transform(), rotation=35, fontsize=8)
    fig.suptitle("Fixed wheel commands and measured response - no navigation controller")
    fig.savefig(output / "wheel_response.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot([r["x_m"] for r in truth], [r["y_m"] for r in truth], color="#126b9c")
    axes[0].scatter([truth[0]["x_m"]], [truth[0]["y_m"]], label="Start", marker="o")
    axes[0].scatter([truth[-1]["x_m"]], [truth[-1]["y_m"]], label="End", marker="x")
    axes[0].set(xlabel="World x (m)", ylabel="World y (m)", title="True path - evaluation only")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend()
    for side in ["left", "right"]:
        axes[1].plot(t, [r[f"{side}_angle_rad"] for r in wheels], label=side)
    axes[1].set(xlabel="Simulation time (s)", ylabel="Wheel angle (rad)", title="Ideal encoder samples")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output / "motion_and_encoders.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: fixed wheel motion and ideal joint samples")
    parser.add_argument("--viewer", action="store_true", help="show the full sequence in real time")
    args = parser.parse_args()
    config_path = ROOT / "configs/wheel_check.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "results/stage2" / run_id
    output.mkdir(parents=True, exist_ok=False)
    git_path = shutil.which("git")
    bundled_git = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe"
    if git_path is None and bundled_git.is_file():
        git_path = str(bundled_git)
    revision = subprocess.run([git_path, "-C", str(ROOT), "rev-parse", "--verify", "HEAD"],
                              capture_output=True, text=True) if git_path else None
    metadata = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
                "mode": "viewer" if args.viewer else "headless", "random_seed": None,
                "git_commit": revision.stdout.strip() if revision and revision.returncode == 0 else None,
                "python": sys.version, "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ["mujoco", "numpy", "matplotlib"]},
                "status": "started", "scope": "fixed_open_loop_commands_and_ideal_encoder_sampling"}
    for relative in ["wheel_check.py", "main.py", "requirements.txt", "resources.lock.json", "configs/wheel_check.json"]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(output / "config.json", config)
    write_json(output / "run.json", metadata)
    wheels, truth, phases = [], [], []
    exit_code = 1
    try:
        model, data, resource, resource_hashes = load_model()
        indices = wheel_indices(model)
        dt = float(model.opt.timestep)
        stride = round(config["sample_period_seconds"] / dt)
        if stride < 1 or not math.isclose(stride * dt, config["sample_period_seconds"], abs_tol=1e-9):
            raise ValueError("Sample period must be an integer multiple of the physics timestep.")
        for phase in config["phases"]:
            if phase["expected"] not in {"stop", "forward", "left", "right"}:
                raise ValueError("Unknown phase expectation.")
            if phase["seconds"] <= config["tail_window_seconds"]:
                raise ValueError("Phase duration must exceed the evaluation window.")
            if not math.isclose(round(phase["seconds"] / dt) * dt, phase["seconds"], abs_tol=1e-9):
                raise ValueError("Phase duration must be an integer multiple of timestep.")
            for side, name in [("left", "wheel_left"), ("right", "wheel_right")]:
                value = phase[f"{side}_rad_s"]
                low, high = model.actuator_ctrlrange[indices[name]["actuator"]]
                if not math.isfinite(value) or not low <= value <= high:
                    raise ValueError("Command outside model control range.")
        metadata.update(physics_timestep_s=dt, sample_period_s=stride * dt,
                        resource=resource, resource_sha256=resource_hashes, wheel_indices=indices)
        if args.viewer:
            from mujoco import viewer as viewer_api
            context = viewer_api.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
        else:
            context = nullcontext(None)
        with context as viewer:
            if viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0.1, 0, 0.08]
                    viewer.cam.distance = 1.15
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -40
                viewer.sync()
                metadata["viewer_opened"] = viewer.is_running()
            for phase in config["phases"]:
                print(f"PHASE: {phase['name']} ({phase['seconds']} s)", flush=True)
                mujoco.mj_forward(model, data)
                start = read_truth_for_evaluation(model, data)
                phase_start = float(data.time)
                first = len(wheels)
                for step in range(round(phase["seconds"] / dt)):
                    if viewer and not viewer.is_running():
                        raise InterruptedError("Window closed before the motion sequence completed.")
                    tick = time.perf_counter()
                    # 命令仅来自预定阶段，不接收真实位置或真实速度作为反馈。
                    data.ctrl[indices["wheel_left"]["actuator"]] = phase["left_rad_s"]
                    data.ctrl[indices["wheel_right"]["actuator"]] = phase["right_rad_s"]
                    mujoco.mj_step(model, data)
                    validate_state(model, data)
                    if (step + 1) % stride == 0:
                        mujoco.mj_forward(model, data)
                        common = {"time_s": float(data.time), "phase": phase["name"],
                                  "phase_seconds": float(data.time) - phase_start}
                        wheels.append({**common, "command_left_rad_s": phase["left_rad_s"],
                                       "command_right_rad_s": phase["right_rad_s"], **read_wheels(data, indices)})
                        truth.append({**common, **read_truth_for_evaluation(model, data)})
                    if viewer:
                        viewer.sync()
                        time.sleep(max(0, dt - (time.perf_counter() - tick)))
                result = evaluate_phase(phase, start, wheels[first:], truth[first:], config)
                phases.append(result)
                print(f"RESULT: {phase['name']} {'PASS' if result['passed'] else 'FAIL'}", flush=True)
        summary = {"status": "passed" if all(p["passed"] for p in phases) else "failed",
                   "simulation_seconds": float(data.time), "samples": len(wheels),
                   "warnings": validate_state(model, data), "phases": phases,
                   "limitations": ["Fixed commands, not autonomous navigation or tracking control.",
                                   "Ideal joint sampling only; no encoder noise, quantization or odometry yet.",
                                   "Acceptance thresholds are project checks, not course grading thresholds."]}
        write_json(output / "summary.json", summary)
        metadata["status"] = summary["status"]
        exit_code = 0 if summary["status"] == "passed" else 1
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "error",
                        error=f"{type(error).__name__}: {error}")
        print(metadata["error"], file=sys.stderr, flush=True)
    finally:
        write_csv(output / "wheel_readings.csv", wheels)
        write_csv(output / "ground_truth_evaluation.csv", truth)
        if wheels:
            try:
                plot_results(output, wheels, truth)
            except Exception as error:
                metadata.update(plot_error=str(error), status="error")
                exit_code = 1
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", metadata)
        hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in output.rglob("*") if p.is_file()}
        write_json(output / "manifest.json", {"files_sha256": hashes})
    print(f"RUN_{metadata['status'].upper()}: {output.relative_to(ROOT).as_posix()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
