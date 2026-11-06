"""一次物理运动、在线带噪编码器/真实RGB、EKF与同输入里程计对照。"""



# Permit both package imports and direct execution of this existing driver.
import sys as _project_sys
from pathlib import Path as _ProjectPath
if __package__ in (None, ""):
    _project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[3]))

from evaluation.common.release_support import project_python_sources


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
import traceback
import uuid

import mujoco
import numpy as np
from PIL import Image

from evaluation.common.camera_scene import load_camera_scene
from evaluation.C_Estimation.code.ekf import LocalizationEKF, initialize_pair_covariance, vision_covariance
from evaluation.C_Estimation.code.ekf_synthetic import run_synthetic_study
from evaluation.B_Sensing.code.lidar_check import set_static_pose
from evaluation.common.main import ROOT, load_model, validate_state
from evaluation.B_Sensing.code.online_encoder import IncrementalEncoder
from evaluation.A_Modelling.code.state_model import wrap_angle
from evaluation.B_Sensing.code.vision import ArucoVision
from evaluation.B_Sensing.code.wheel_check import read_truth_for_evaluation, read_wheels, wheel_indices, write_csv, write_json


def append_json(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def public_observation(item):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in vars(item).items()}


def phase_at(elapsed, phases):
    end = 0.
    for phase in phases:
        end += phase["seconds"]
        if elapsed < end - 1e-8:
            return phase
    return phases[-1]


def evaluate_records(estimates, truth, frames, config, output):
    states = np.array([row["ekf_state"] for row in estimates])
    baseline = np.array([row["encoder_only_state"] for row in estimates])
    actual = np.array([[row["x_m"], row["y_m"], row["yaw_rad"]] for row in truth])
    error, baseline_error = states - actual, baseline - actual
    error[:, 2], baseline_error[:, 2] = wrap_angle(error[:, 2]), wrap_angle(baseline_error[:, 2])
    covariances = np.array([row["ekf_P"] for row in estimates])
    normalized = [float(e @ np.linalg.solve(P, e)) for e, P in zip(error, covariances)]
    metrics = {}
    for label, values in [("ekf", error), ("encoder_only", baseline_error)]:
        metrics[label] = {"position_rmse_m": float(np.sqrt(np.mean(np.sum(values[:, :2]**2, axis=1)))),
                          "yaw_rmse_degrees": float(np.degrees(np.sqrt(np.mean(values[:, 2]**2)))),
                          "final_position_error_m": float(np.linalg.norm(values[-1, :2])),
                          "final_yaw_error_degrees": float(np.degrees(values[-1, 2]))}
    phase_metrics = []
    for phase in config["phases"]:
        selected = [frame for frame in frames[1:] if frame["phase"] == phase["name"]]
        counts = {status: sum(frame["update"]["status"] == status for frame in selected)
                  for status in ["accepted", "rejected_nis", "no_observation"]}
        phase_metrics.append({"phase": phase["name"], "frames": len(selected), "update_counts": counts,
                              "raw_detection_counts": sorted(set(len(frame["raw_observations"]) for frame in selected)),
                              "delivered_detection_counts": sorted(set(len(frame["delivered_observations"]) for frame in selected))})
    blackout = [row for row in estimates if row["phase"] == "right_occluded"]
    blackout_start_index = estimates.index(blackout[0]) - 1
    start_trace = float(np.trace(np.asarray(estimates[blackout_start_index]["ekf_P"])))
    end_trace = float(np.trace(np.asarray(blackout[-1]["ekf_P"])))
    reacquisition = next(frame for frame in frames if frame["phase"] == "forward_reacquired" and frame["update"]["status"] == "accepted")
    checks = {"finite_P_and_PSD": bool(np.isfinite(covariances).all() and np.linalg.eigvalsh(covariances).min() >= -1e-12),
              "blackout_has_no_delivered_observations": all(not frame["delivered_observations"] for frame in frames if frame["phase"] == "right_occluded"),
              "blackout_has_no_raw_detections": all(not frame["raw_observations"] for frame in frames if frame["phase"] == "right_occluded"),
              "blackout_covariance_trace_grows": end_trace > start_trace,
              "reacquisition_reduces_trace": reacquisition["posterior_trace"] < reacquisition["prior_trace"],
              "single_marker_updates_accepted": any(frame["phase"] == "left_single_message" and len(frame["delivered_observations"]) == 1 and frame["update"]["status"] == "accepted" for frame in frames),
              "base_upright": max(row["tilt_rad"] for row in truth) < config["checks"]["max_tilt_rad"],
              "actual_motion": float(np.linalg.norm(np.diff(actual[:, :2], axis=0), axis=1).sum()) >= config["checks"]["min_true_path_length_m"],
              "position_rmse_within_predeclared_module_limit": metrics["ekf"]["position_rmse_m"] <= config["checks"]["max_position_rmse_m"],
              "yaw_rmse_within_predeclared_module_limit": metrics["ekf"]["yaw_rmse_degrees"] <= config["checks"]["max_yaw_rmse_degrees"]}
    result = {"checks": checks, "passed": all(checks.values()), "metrics": metrics, "phase_metrics": phase_metrics,
              "path_length_m": float(np.linalg.norm(np.diff(actual[:, :2], axis=0), axis=1).sum()),
              "blackout_covariance_trace_mixed_units": {"before": start_trace, "after": end_trace},
              "reacquisition_covariance_trace_mixed_units": {"before": reacquisition["prior_trace"], "after": reacquisition["posterior_trace"]},
              "descriptive_mean_nees_3d": float(np.mean(normalized)),
              "descriptive_fraction_inside_95pct_3d_ellipsoid": float(np.mean(np.asarray(normalized) <= config["synthetic_study"]["nees_95_limit_3d"])),
              "physical_uncertainty_status": "One time-correlated physical run and uncalibrated design Q/R; coverage is descriptive, not proof of confidence calibration. Trace sums m² and rad² only as a fixed-coordinate diagnostic.",
              "expected_error_reduction_observed": metrics["ekf"]["position_rmse_m"] < metrics["encoder_only"]["position_rmse_m"]}
    write_json(output / "physical_evaluation.json", result)
    return result, error, baseline_error


def plot_records(estimates, truth, frames, error, baseline_error, output):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = np.array([row["elapsed_s"] for row in estimates])
    states = np.array([row["ekf_state"] for row in estimates])
    actual = np.array([[row["x_m"], row["y_m"], row["yaw_rad"]] for row in truth])
    baseline = np.array([row["encoder_only_state"] for row in estimates])
    Ps = np.array([row["ekf_P"] for row in estimates])
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    scales, labels = [1000, 1000, 180/np.pi], ["x error (mm)", "y error (mm)", "Yaw error (deg)"]
    for i, axis in enumerate(axes):
        band = 2*np.sqrt(Ps[:, i, i])*scales[i]
        axis.fill_between(times, -band, band, color="#168fa4", alpha=.2, label="EKF +/-2 model std (uncalibrated)")
        axis.plot(times, error[:, i]*scales[i], color="#168fa4", label="EKF error")
        axis.plot(times, baseline_error[:, i]*scales[i], color="#d38a31", linestyle="--", label="Encoder-only error")
        axis.axvspan(4, 7, color="#cccccc", alpha=.2)
        axis.axvspan(7, 10, color="#e8a28a", alpha=.2)
        axis.set(ylabel=labels[i])
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("One physical run: pair -> single message -> physical occlusion -> reacquired")
    axes[-1].set_xlabel("Elapsed simulation time after image initialization (s)")
    fig.savefig(output / "ekf_errors_uncertainty.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    for values, label in [(actual, "Physical truth (evaluation)"), (states, "EKF"), (baseline, "Encoder only")]:
        axes[0].plot(values[:, 0], values[:, 1], label=label)
    axes[0].set(xlabel="World x (m)", ylabel="World y (m)", title="Observed physical trajectory")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].plot(times, np.sqrt(Ps[:, 0, 0])*1000, label="x std (mm)")
    axes[1].plot(times, np.sqrt(Ps[:, 1, 1])*1000, label="y std (mm)")
    axes[1].axvspan(4, 7, color="#cccccc", alpha=.2, label="ID18 message withheld")
    axes[1].axvspan(7, 10, color="#e8a28a", alpha=.2, label="Physical camera occlusion")
    axes[1].set(xlabel="Elapsed simulation time (s)", ylabel="Position standard deviation (mm)", title="Internal model uncertainty, not calibrated coverage")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=.2)
    fig.savefig(output / "ekf_trajectory.png", dpi=150)
    plt.close(fig)


def run_physical(config, camera_config, known_map, output):
    room = json.loads((ROOT / camera_config["lidar_config"]).read_text(encoding="utf-8"))
    model, data, provenance, robot_xml, scene_xml = load_camera_scene(room, camera_config)
    write_json(output / "scene_provenance.json", provenance)
    (output / "derived_robot.xml").write_bytes(robot_xml)
    (output / "derived_scene.xml").write_bytes(scene_xml)
    original, _, _, _ = load_model()
    for name in ["base", "wheel_left", "wheel_right"]:
        np.testing.assert_array_equal(model.body(name).mass, original.body(name).mass)
        np.testing.assert_array_equal(model.body(name).inertia, original.body(name).inertia)
    set_static_pose(model, data, config["start_pose_m_m_rad"])
    dt = float(model.opt.timestep)
    settle_steps = round(config["settle_seconds"]/dt)
    encoder_stride = round(1/(config["encoder_rate_hz"]*dt))
    camera_stride = round(1/(config["camera_rate_hz"]*dt))
    assert abs(encoder_stride*dt*config["encoder_rate_hz"]-1) < 1e-10
    assert abs(camera_stride*dt*config["camera_rate_hz"]-1) < 1e-10 and camera_stride % encoder_stride == 0
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    validate_state(model, data)
    start_time = float(data.time)
    sensor = IncrementalEncoder(**config["encoder_noise"])
    indices = wheel_indices(model)
    detector = ArucoVision(camera_config)
    write_json(output / "calibration.json", {"K": detector.K.tolist(), "distortion": detector.distortion.tolist()})
    mocap_id = int(model.body_mocapid[model.body("vision_occluder").id])
    frames, estimates, truths, encoder_rows = [], [], [], []
    rgb_dir = output / "rgb"
    rgb_dir.mkdir()
    steps = round(sum(phase["seconds"] for phase in config["phases"])/dt)
    with (output / "vision_frames.jsonl").open("x", encoding="utf-8") as vision_stream, \
         (output / "estimates.jsonl").open("x", encoding="utf-8") as estimate_stream, \
         (output / "encoder_sensor_audit.jsonl").open("x", encoding="utf-8") as sensor_stream, \
         mujoco.Renderer(model, height=detector.height, width=detector.width) as renderer:
        wheels = read_wheels(data, indices)
        measured, audit = sensor.sample([wheels["left_angle_rad"], wheels["right_angle_rad"]], start_time)
        renderer.update_scene(data, camera=camera_config["camera"]["name"])
        rgb = renderer.render().copy()
        Image.fromarray(rgb).save(rgb_dir / "frame0000.png")
        raw = sorted(detector.detect(rgb, start_time), key=lambda item: item.marker_id)
        public = [public_observation(item) for item in raw]
        if [item["marker_id"] for item in public] != [17, 18]:
            raise RuntimeError("Initial image requires both known markers; no truth fallback.")
        R0 = vision_covariance(2, config["vision_R"])
        pose, P0, initialization = initialize_pair_covariance(known_map, public, R0, config["initialization_floor_std_m_m_rad"])
        estimator = LocalizationEKF(pose, P0, known_map, config["wheel_radius_m"], config["track_width_m"], measured,
                                    start_time, config["process"], initial_frame_used=True)
        baseline = LocalizationEKF(pose, P0, known_map, config["wheel_radius_m"], config["track_width_m"], measured,
                                   start_time, config["process"], initial_frame_used=True)
        write_json(output / "initialization.json", {"time_s": start_time, "state": pose.tolist(), "P": P0.tolist(),
                                                    "details": initialization, "frame_consumed_once": True})
        first_frame = {"frame_id": "frame0000", "time_s": start_time, "elapsed_s": 0., "phase": "initialization",
                       "raw_observations": public, "delivered_observations": public,
                       "update": {"status": "initialization_only", "nis": None}}
        frames.append(first_frame)
        append_json(vision_stream, first_frame)

        def record(elapsed, phase, prediction=None, update_status="none"):
            row = {"time_s": float(data.time), "elapsed_s": elapsed, "phase": phase["name"],
                   "ekf_state": estimator.state.tolist(), "ekf_P": estimator.P.tolist(),
                   "encoder_only_state": baseline.state.tolist(), "encoder_only_P": baseline.P.tolist(),
                   "prediction": prediction, "vision_status": update_status}
            estimates.append(row)
            append_json(estimate_stream, row)
            # 真实状态独立记录，未传给估计器或命令计算。
            truths.append(dict(time_s=float(data.time), elapsed_s=elapsed, **read_truth_for_evaluation(model, data)))
            encoder_rows.append({"time_s": float(data.time), "elapsed_s": elapsed, "phase": phase["name"],
                                 "left_angle_rad": float(measured[0]), "right_angle_rad": float(measured[1]),
                                 "left_target_rad_s": phase["wheel_targets_rad_s"][0], "right_target_rad_s": phase["wheel_targets_rad_s"][1]})
            append_json(sensor_stream, dict(time_s=float(data.time), **audit))

        record(0., config["phases"][0], update_status="initialization_only")
        for step in range(1, steps+1):
            elapsed = step*dt
            phase = phase_at((step-1)*dt, config["phases"])
            for index, name in enumerate(["wheel_left", "wheel_right"]):
                data.ctrl[indices[name]["actuator"]] = phase["wheel_targets_rad_s"][index]
            data.mocap_pos[mocap_id] = config["occluder_world_m"] if phase["visibility"] == "physical_occluder" else [0, 0, -10]
            mujoco.mj_step(model, data)
            if step % encoder_stride:
                continue
            mujoco.mj_forward(model, data)
            validate_state(model, data)
            wheels = read_wheels(data, indices)
            measured, audit = sensor.sample([wheels["left_angle_rad"], wheels["right_angle_rad"]], float(data.time))
            prediction = estimator.predict_encoders(measured, float(data.time))
            baseline.predict_encoders(measured, float(data.time))
            status = "none"
            if step % camera_stride == 0:
                frame_id = f"frame{step//camera_stride:04d}"
                renderer.update_scene(data, camera=camera_config["camera"]["name"])
                rgb = renderer.render().copy()
                Image.fromarray(rgb).save(rgb_dir / f"{frame_id}.png")
                observations = sorted(detector.detect(rgb, float(data.time)), key=lambda item: item.marker_id)
                public = [public_observation(item) for item in observations]
                delivered = [item for item in public if not (phase["visibility"] == "drop_id18" and item["marker_id"] == 18)]
                R = vision_covariance(len(delivered), config["vision_R"]) if delivered else np.empty((0, 0))
                prior_state, prior_P = estimator.state.copy(), estimator.P.copy()
                update = estimator.update_vision(delivered, float(data.time), frame_id, R, config["nis_limits"].get(str(2*len(delivered)), 1))
                status = update["status"]
                frame = {"frame_id": frame_id, "time_s": float(data.time), "elapsed_s": elapsed, "phase": phase["name"],
                         "visibility_policy": phase["visibility"], "raw_observations": public, "delivered_observations": delivered,
                         "prior_state": prior_state.tolist(), "prior_P": prior_P.tolist(), "posterior_state": estimator.state.tolist(),
                         "posterior_P": estimator.P.tolist(), "prior_trace": float(np.trace(prior_P)),
                         "posterior_trace": float(np.trace(estimator.P)), "update": update}
                frames.append(frame)
                append_json(vision_stream, frame)
            record(elapsed, phase, prediction, status)
    write_csv(output / "encoder_measurements.csv", encoder_rows)
    write_csv(output / "ground_truth_evaluation.csv", truths)
    evaluation, error, baseline_error = evaluate_records(estimates, truths, frames, config, output)
    plot_records(estimates, truths, frames, error, baseline_error, output)
    return {"physics_steps": steps+settle_steps, "settle_steps": settle_steps, "motion_steps": steps, "motion_seconds": steps*dt,
            "encoder_samples_including_baseline": len(estimates), "camera_frames_including_initialization": len(frames),
            "warning_counts": validate_state(model, data), "robot_mass_and_inertia_unchanged": True, "evaluation": evaluation}


def main():
    config = json.loads((ROOT / "configs/ekf_check.json").read_text(encoding="utf-8"))
    camera_config = json.loads((ROOT / config["camera_config"]).read_text(encoding="utf-8"))
    model_config = json.loads((ROOT / config["model_config"]).read_text(encoding="utf-8"))
    camera_config["additional_markers"] = model_config["additional_markers"]
    known_map = {item["id"]: item["position_world_m"][:2] for item in [camera_config["marker"], *camera_config["additional_markers"]]}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "results/stage4_ekf" / run_id
    output.mkdir(parents=True, exist_ok=False)
    git = shutil.which("git") or str(Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe")
    revision = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
    metadata = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
                "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
                "python": sys.version, "platform": platform.platform(), "git_commit": revision.stdout.strip() if revision.returncode == 0 else None,
                "packages": {name: importlib.metadata.version(name) for name in ["mujoco", "numpy", "matplotlib", "pillow", "opencv-python-headless"]},
                "random_generator": "NumPy Generator PCG64", "encoder_seed": config["encoder_noise"]["seed"],
                "parameter_basis": config["parameter_basis"], "scope": config["scope"]}
    sources = [*project_python_sources(ROOT), *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.json"), ROOT / "scenes/room.xml",
               ROOT / "requirements.txt", ROOT / "resources.lock.json", ROOT / "launch_ekf_check.cmd"]
    for source in sources:
        target = output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name, value in [("config.json", config), ("camera_config.json", camera_config), ("known_map.json", known_map), ("run.json", metadata)]:
        write_json(output / name, value)
    started, exit_code = time.perf_counter(), 1
    try:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
        (output / "tests.txt").write_text(tests.stdout+tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Module tests failed; see tests.txt.")
        dependency = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
        (output / "dependency_check.txt").write_text(dependency.stdout+dependency.stderr, encoding="utf-8")
        if dependency.returncode:
            raise RuntimeError("Dependency check failed.")
        synthetic = run_synthetic_study(config, known_map, output)
        print(f"Synthetic study completed: NEES={synthetic['mean_final_nees_3d']:.3f}, coverage={synthetic['final_95pct_3d_coverage']:.3f}", flush=True)
        physical = run_physical(config, camera_config, known_map, output)
        passed = synthetic["passed"] and physical["evaluation"]["passed"]
        summary = {"status": "completed" if passed else "failed", "module_tests_passed": True, "dependency_check_passed": True,
                   "synthetic_study": synthetic, "physical_run": physical,
                   "truth_boundary": "EKF gets noisy wheel angles, timestamped image measurements and a fixed map only. Open-loop commands depend only on time. Truth logs are evaluation-only.",
                   "limits": "No controller, LiDAR navigation, docking, empirical physical Q/R calibration or final task reproduction."}
        write_json(output / "summary.json", summary)
        metadata["status"], exit_code = summary["status"], 0 if passed else 1
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as error:
        metadata.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(metadata["error"], file=sys.stderr, flush=True)
    finally:
        metadata["completed_at_utc"], metadata["wall_elapsed_seconds"] = datetime.now(timezone.utc).isoformat(), time.perf_counter()-started
        write_json(output / "run.json", metadata)
        write_json(output / "manifest.json", {"files_sha256": {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                                              for p in output.rglob("*") if p.is_file()}})
    print(f"EKF_{metadata['status'].upper()}: {output.relative_to(ROOT).as_posix()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
