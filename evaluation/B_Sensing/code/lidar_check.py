"""房间、理想二维扫描及独立几何验收；含静态姿态和零输入物理采样。"""

# Permit both package imports and direct execution of this existing driver.
import sys as _project_sys
from pathlib import Path as _ProjectPath
if __package__ in (None, ""):
    _project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[3]))


import argparse
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

from evaluation.B_Sensing.code.lidar import HIT, Lidar2D
from evaluation.B_Sensing.code.lidar_reference import expected_scan, read_reference_geometries
from evaluation.common.main import ROOT, load_model, validate_state
from evaluation.common.room_scene import load_room
from evaluation.B_Sensing.code.wheel_check import write_csv, write_json


def set_static_pose(model, data, pose):
    """仅验收时直接设置已知姿态，不是导航或定位算法。"""
    mujoco.mj_resetData(model, data)
    address = int(model.jnt_qposadr[model.joint("base_joint").id])
    x, y, yaw = pose
    data.qpos[address:address + 3] = [x, y, 0.0]
    data.qpos[address + 3:address + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    mujoco.mj_forward(model, data)


def expected_static_transform(pose, config):
    """从声明姿态及安装参数独立计算，不读取 data.site_xpos/site_xmat。"""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    base_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    quaternion = np.asarray(config["sensor_quaternion_in_base_wxyz"], dtype=float)
    w, qx, qy, qz = quaternion / np.linalg.norm(quaternion)
    mounting_rotation = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*w), 2*(qx*qz+qy*w)],
        [2*(qx*qy+qz*w), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*w)],
        [2*(qx*qz-qy*w), 2*(qy*qz+qx*w), 1-2*(qx*qx+qy*qy)]])
    origin = np.array([x, y, 0.0]) + base_rotation @ np.asarray(config["sensor_position_in_base_m"])
    return origin, base_rotation @ mounting_rotation


def verify_scan(scan, diagnostic, reference, config, declared_pose=None):
    origin, directions = diagnostic["origin_world_m"], diagnostic["directions_world"]
    transform_error = 0.0
    if declared_pose is not None:
        origin, rotation = expected_static_transform(declared_pose, config)
        local_directions = np.column_stack((np.cos(scan.angles_rad), np.sin(scan.angles_rad), np.zeros(len(scan.angles_rad))))
        directions = local_directions @ rotation.T
        transform_error = float(max(np.max(np.abs(origin - diagnostic["origin_world_m"])),
                                    np.max(np.abs(rotation - diagnostic["rotation_world_sensor"]))))
    expected_ranges, expected_status = expected_scan(origin, directions, reference,
                                                     config["lidar"]["range_min_m"], config["lidar"]["range_max_m"])
    error = np.abs(scan.ranges_m - expected_ranges)
    return {"beams": len(scan.ranges_m), "hit_count": int(np.sum(scan.status == 1)),
            "no_return_count": int(np.sum(scan.status == 0)), "too_close_count": int(np.sum(scan.status == 2)),
            "max_absolute_range_error_m": float(error.max()),
            "declared_static_transform_max_error": transform_error if declared_pose is not None else None,
            "status_mismatches": int(np.sum(scan.status != expected_status)),
            "passed": bool(error.max() <= config["geometry_tolerance_m"] and transform_error <= 1e-12
                           and np.array_equal(scan.status, expected_status))}, expected_ranges


def save_scan(folder, name, scan, diagnostic, expected_ranges, model):
    # 公共测量 CSV 和带世界坐标/命中对象的评估 CSV 分开保存。
    write_csv(folder / f"{name}_measurement.csv", [
        {"time_s": scan.time_s, "frame_id": scan.frame_id, "beam": index, "angle_rad": float(angle),
         "range_m": float(distance), "status": int(status)}
        for index, (angle, distance, status) in enumerate(zip(scan.angles_rad, scan.ranges_m, scan.status))])
    write_csv(folder / f"{name}_evaluation.csv", [
        {"beam": index, "raw_distance_m": float(raw), "expected_reported_range_m": float(expected),
         "hit_geom": model.geom(int(geom_id)).name if geom_id >= 0 else "",
         "world_direction_x": float(direction[0]), "world_direction_y": float(direction[1]), "world_direction_z": float(direction[2])}
        for index, (raw, expected, geom_id, direction) in enumerate(zip(diagnostic["raw_distances_m"], expected_ranges,
                                                                        diagnostic["geom_ids"], diagnostic["directions_world"]))])
    write_json(folder / f"{name}_pose_evaluation.json", {
        "origin_world_m": diagnostic["origin_world_m"].tolist(),
        "rotation_world_sensor": diagnostic["rotation_world_sensor"].tolist(),
        "purpose": "Simulator sensor generation and evaluation only; not supplied in LidarScan."})


def plot_scans(output, scans, reference):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    for item in reference:
        x, y = item["position_m"][:2]
        if item["type"] == "box":
            hx, hy = item["size_m"][:2]
            axes[0].add_patch(Rectangle((x - hx, y - hy), 2 * hx, 2 * hy, color="#bec5c9"))
        elif item["type"] == "cylinder":
            axes[0].add_patch(Circle((x, y), item["size_m"][0], color="#bec5c9"))
    colors = ["#007da5", "#d47b16", "#4c963c", "#9557a5"]
    for (name, scan, diagnostic), color in zip(scans, colors):
        valid = scan.status == HIT
        points = diagnostic["origin_world_m"] + scan.ranges_m[valid, None] * diagnostic["directions_world"][valid]
        axes[0].scatter(points[:, 0], points[:, 1], s=4, color=color, label=name)
        axes[0].plot(*diagnostic["origin_world_m"][:2], marker="x", color=color)
    name, scan, diagnostic = scans[0]
    valid = scan.status == HIT
    x, y = scan.ranges_m[valid] * np.cos(scan.angles_rad[valid]), scan.ranges_m[valid] * np.sin(scan.angles_rad[valid])
    for px, py in zip(x[::4], y[::4]):
        axes[1].plot([0, px], [0, py], color="#007da5", alpha=0.12, linewidth=0.7)
    axes[1].scatter(x, y, s=8, color="#007da5", label="Valid surface returns")
    axes[1].plot(0, 0, marker="x", color="#c43b2b", label="LiDAR origin")
    axes[1].arrow(0, 0, 0.5, 0, color="#c43b2b", width=0.01, length_includes_head=True)
    axes[0].set(title="Known-pose validation; world coordinates for evaluation", xlabel="World x (m)", ylabel="World y (m)", xlim=(-3.25, 3.25), ylim=(-2.75, 2.75))
    axes[1].set(title="Origin scan in sensor frame; +x forward, +y left", xlabel="Sensor x (m)", ylabel="Sensor y (m)")
    for axis in axes:
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(output / "lidar_scans.png", dpi=160)
    plt.close(fig)


def add_scan_overlay(scene, scan, diagnostic):
    """可视化线段不参与碰撞或射线查询，每 6 条有效射线显示一条。"""
    for index in np.flatnonzero(scan.status == HIT)[::6]:
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Visualization geometry capacity exceeded.")
        geom = scene.geoms[scene.ngeom]
        origin = diagnostic["origin_world_m"]
        endpoint = origin + scan.ranges_m[index] * diagnostic["directions_world"][index]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                          np.array([0.1, 0.95, 0.35, 0.75], dtype=np.float32))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1.5, origin, endpoint)
        scene.ngeom += 1


def render_room(model, data, output, scan, diagnostic):
    from PIL import Image
    from OpenGL import GL
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [0, 0, 0.15]
    camera.distance, camera.azimuth, camera.elevation = 9, 115, -55
    with mujoco.Renderer(model, height=800, width=1100) as renderer:
        renderer.update_scene(data, camera=camera)
        add_scan_overlay(renderer.scene, scan, diagnostic)
        pixels = renderer.render().copy()
        graphics = {name: GL.glGetString(value).decode("utf-8") for name, value in
                    [("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION)]}
    if pixels.shape != (800, 1100, 3) or pixels.std() < 1:
        raise RuntimeError("Room rendering is empty.")
    Image.fromarray(pixels).save(output / "room_render.png")
    return {"shape": list(pixels.shape), "opengl": graphics, "camera": "External visualization, not an onboard sensor"}


def run_viewer(model, data, seconds, scan, diagnostic):
    from mujoco import viewer as viewer_api
    started, frames = time.perf_counter(), 0
    with viewer_api.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0, 0, 0.15]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 9, 115, -55
            add_scan_overlay(viewer.user_scn, scan, diagnostic)
        viewer.sync()
        if not viewer.is_running():
            raise RuntimeError("Room viewer did not open.")
        while viewer.is_running() and (seconds == 0 or time.perf_counter() - started < seconds):
            viewer.sync()
            frames += 1
            time.sleep(0.02)
        result = {"opened": True, "display_frames": frames, "additional_physics_steps": 0,
                  "elapsed_wall_seconds": time.perf_counter() - started,
                  "end_reason": "time_limit" if viewer.is_running() else "window_closed",
                  "warnings": validate_state(model, data), "purpose": "Paused room/scan snapshot after physics check; no autonomous navigation"}
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate room geometry and ideal 2D LiDAR")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--viewer-seconds", type=float, default=15.0, help="0 keeps window open")
    args = parser.parse_args()
    if not math.isfinite(args.viewer_seconds) or args.viewer_seconds < 0:
        parser.error("viewer-seconds must be finite and nonnegative")
    config = json.loads((ROOT / "configs/lidar_check.json").read_text(encoding="utf-8"))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output = ROOT / "results/stage3_lidar" / run_id
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"run_id": run_id, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "started",
                "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
                "operation": "Static known-pose ray validation and zero-input physical simulation with scans",
                "random_seed": None, "git_commit": None, "viewer_requested": args.viewer,
                "python": sys.version, "platform": platform.platform(),
                "packages": {name: importlib.metadata.version(name) for name in ["mujoco", "numpy", "matplotlib", "pillow"]}}
    git = shutil.which("git")
    if git:
        revision = subprocess.run([git, "-C", str(ROOT), "rev-parse", "--verify", "HEAD"], capture_output=True, text=True)
        if revision.returncode == 0:
            metadata["git_commit"] = revision.stdout.strip()
    for relative in ["lidar.py", "lidar_reference.py", "room_scene.py", "lidar_check.py", "main.py", "wheel_check.py",
                     "configs/lidar_check.json", "scenes/room.xml", "requirements.txt", "resources.lock.json",
                     "tests/test_lidar.py", "tests/test_odometry.py", "tests/test_encoder_noise.py", "odometry.py", "encoder_noise.py",
                     "launch_lidar_check.cmd"]:
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    write_json(output / "run.json", metadata)
    write_json(output / "config.json", config)
    exit_code = 1
    try:
        tests = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT, capture_output=True, text=True)
        (output / "tests.txt").write_text(tests.stdout + tests.stderr, encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("LiDAR or preceding module tests failed.")
        model, data, provenance, derived_robot = load_room(config)
        (output / "derived_robot.xml").write_bytes(derived_robot)
        write_json(output / "scene_provenance.json", provenance)
        original, _, _, original_hashes = load_model()
        unchanged_dynamics = all(np.array_equal(getattr(model, name), getattr(original, name)) for name in
                                 ["body_mass", "body_inertia", "body_ipos", "body_iquat", "jnt_type", "jnt_axis",
                                  "dof_frictionloss", "dof_armature", "actuator_gainprm", "actuator_biasprm", "qpos0"])
        if not unchanged_dynamics or original_hashes != provenance["official_resource_sha256"]:
            raise RuntimeError("Room augmentation changed pinned robot dynamics or resource provenance.")
        reference = read_reference_geometries(ROOT / config["scene"])
        write_json(output / "reference_geometry.json", reference)
        scanner = Lidar2D(model, config["site_name"], robot_geom_group=config["robot_geom_group"], **config["lidar"])
        scans, static_results = [], []
        for item in config["static_cases"]:
            set_static_pose(model, data, item["base_xy_yaw"])
            scan, diagnostic = scanner.simulator_scan_with_diagnostics(data)
            result, expected = verify_scan(scan, diagnostic, reference, config, declared_pose=item["base_xy_yaw"])
            result.update(case=item["name"], base_xy_yaw=item["base_xy_yaw"])
            save_scan(output, item["name"], scan, diagnostic, expected, model)
            scans.append((item["name"], scan, diagnostic))
            static_results.append(result)
            if not result["passed"]:
                raise RuntimeError(f"Static scan validation failed: {item['name']}")
        cardinal = {name: float(scans[0][1].ranges_m[index]) for name, index in
                    [("front", 180), ("left", 270), ("back", 0), ("right", 90)]}
        hand_prediction = bool(np.allclose(list(cardinal.values()), [1.049, 2.5, 2.951, 2.5], atol=config["geometry_tolerance_m"], rtol=0)
                               and abs(scans[1][1].ranges_m[180] - 2.549) <= config["geometry_tolerance_m"])
        if not hand_prediction:
            raise RuntimeError("Hand-computed cardinal predictions did not match.")
        plot_scans(output, scans, reference)
        set_static_pose(model, data, [0, 0, 0])
        dt = float(model.opt.timestep)
        steps = round(config["physics_seconds"] / dt)
        period_steps = round(1 / config["scan_rate_hz"] / dt)
        if not math.isclose(period_steps * dt, 1 / config["scan_rate_hz"], abs_tol=1e-12):
            raise ValueError("Scan rate must divide the physics sampling rate exactly in this check.")
        dynamic_results, truth_rows = [], []
        for step in range(1, steps + 1):
            mujoco.mj_step(model, data)
            warnings = validate_state(model, data)
            if step * dt > config["settle_seconds"] + 1e-12 and step % period_steps == 0:
                mujoco.mj_forward(model, data)
                scan, diagnostic = scanner.simulator_scan_with_diagnostics(data)
                result, expected = verify_scan(scan, diagnostic, reference, config)
                result.update(time_s=float(data.time))
                dynamic_results.append(result)
                save_scan(output, f"physics_{len(dynamic_results):02}", scan, diagnostic, expected, model)
                truth_rows.append({"time_s": float(data.time), "sensor_x_m": float(diagnostic["origin_world_m"][0]),
                                   "sensor_y_m": float(diagnostic["origin_world_m"][1]), "sensor_z_m": float(diagnostic["origin_world_m"][2])})
                if not result["passed"]:
                    raise RuntimeError("Physics-state ray validation failed.")
        write_csv(output / "physics_pose_evaluation.csv", truth_rows)
        rendering = render_room(model, data, output, scan, diagnostic)
        viewer_status = run_viewer(model, data, args.viewer_seconds, scan, diagnostic) if args.viewer else {"requested": False}
        summary = {"status": "completed", "module_tests_passed": True, "hand_predictions_passed": hand_prediction,
                   "robot_dynamics_unchanged": unchanged_dynamics,
                   "cardinal_ranges_at_origin_m": cardinal, "static_cases": static_results,
                   "physics": {"steps": steps, "simulated_seconds": steps * dt, "scan_rate_hz": config["scan_rate_hz"],
                               "scans": dynamic_results, "warning_counts": warnings, "controls": "Both wheel commands zero"},
                   "rendering": rendering, "viewer": viewer_status,
                   "measurement_contract": {"status": {"0": "no return in range; range field capped to maximum, not a surface", "1": "valid hit", "2": "too close; range field capped to minimum"},
                                            "frame": "+x forward, +y left, +z up at lidar_site; plane follows site tilt", "scan_is_instantaneous": True},
                   "quality_status": "Ideal geometry module verified; no navigation, sensor calibration, camera, EKF or task success claimed."}
        write_json(output / "summary.json", summary)
        metadata.update(status="completed")
        exit_code = 0
        print(json.dumps({key: value for key, value in summary.items() if key not in ["physics", "rendering"]}, indent=2), flush=True)
        print(f"PHYSICS: {steps} steps; {len(dynamic_results)} scans; warnings all zero", flush=True)
    except Exception as error:
        metadata.update(status="error", error=f"{type(error).__name__}: {error}")
        print(metadata["error"], file=sys.stderr, flush=True)
    finally:
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", metadata)
        hashes = {p.relative_to(output).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in output.rglob("*") if p.is_file()}
        write_json(output / "manifest.json", {"files_sha256": hashes})
    print(f"LIDAR_{metadata['status'].upper()}: {output.relative_to(ROOT).as_posix()}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
