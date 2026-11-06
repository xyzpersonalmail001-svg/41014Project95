"""第一阶段：加载官方模型、检查接口、验证物理步进及图形显示。"""

# Permit both package imports and direct execution of this existing driver.
import sys as _project_sys
from pathlib import Path as _ProjectPath
if __package__ in (None, ""):
    _project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "stage1"


def save_json(name, content):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(
        json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_model():
    """用内存资源加载，避开 Windows 上 MuJoCo 的中文文件路径问题。"""
    source = json.loads((ROOT / "resources.lock.json").read_text(encoding="utf-8"))["robotis"]
    scene = ROOT / source["directory"] / source["scene"]
    if not scene.is_file():
        raise FileNotFoundError("Official model is missing. Run scripts/setup.ps1 first.")
    directory = scene.parent
    # 官方模型的 include 和 meshdir 使用相对路径，VFS 中保留同样的结构。
    files = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj"}
    )
    assets = {p.relative_to(directory).as_posix(): p.read_bytes() for p in files}
    model = mujoco.MjModel.from_xml_string(scene.read_text(encoding="utf-8"), assets=assets)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in assets.items()}
    return model, mujoco.MjData(model), source, hashes


def inspect_model(model, source, hashes):
    joints = []
    for index in range(model.njnt):
        joint = model.joint(index)
        joints.append({
            "id": index, "name": joint.name,
            "type": mujoco.mjtJoint(int(model.jnt_type[index])).name,
            "qpos_address": int(model.jnt_qposadr[index]),
            "qvel_address": int(model.jnt_dofadr[index]),
            "local_axis": model.jnt_axis[index].tolist(),
        })
    actuators = []
    for index in range(model.nu):
        joint_id = int(model.actuator_trnid[index, 0])
        actuators.append({
            "id": index, "name": model.actuator(index).name,
            "joint": model.joint(joint_id).name,
            "control_limited": bool(model.actuator_ctrllimited[index]),
            "control_range": model.actuator_ctrlrange[index].tolist(),
            "gain_parameters": model.actuator_gainprm[index].tolist(),
            "bias_parameters": model.actuator_biasprm[index].tolist(),
            "command_unit_from_xml": "rad/s (velocity actuator)",
        })
    inventory = {
        "source": source, "resource_sha256": hashes,
        "loading_method": "XML string and in-memory virtual file system (VFS)",
        "timestep_seconds": float(model.opt.timestep),
        "counts": {name: int(getattr(model, name)) for name in
                   ["nq", "nv", "nu", "njnt", "nbody", "ngeom", "ncam", "nsensor"]},
        "joints": joints, "actuators": actuators,
        "cameras": [model.camera(i).name for i in range(model.ncam)],
        "sensors": [model.sensor(i).name for i in range(model.nsensor)],
        "wheel_body_positions": {name: model.body(name).pos.tolist()
                                 for name in ["wheel_left", "wheel_right"]},
        "notes": [
            "Free viewing camera is a rendering tool, not an onboard camera sensor.",
            "LDS mesh is geometry, not a configured LiDAR scan.",
            "Wheel joint positions/velocities are available; encoder sampling and noise are not implemented.",
        ],
    }
    save_json("model_inventory.json", inventory)
    return inventory


def validate_state(model, data):
    if not all(np.isfinite(values).all() for values in [data.qpos, data.qvel, data.qacc]):
        raise RuntimeError("Non-finite physical state detected.")
    warnings = {mujoco.mjtWarning(i).name: int(data.warning[i].number)
                for i in range(int(mujoco.mjtWarning.mjNWARNING))}
    if any(warnings.values()):
        raise RuntimeError(f"MuJoCo warnings: {warnings}")
    return warnings


def render_image(model, data):
    """保存真实渲染结果；这里的自由视角不充当机器人相机。"""
    from PIL import Image
    from OpenGL import GL

    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = [-0.04, 0.0, 0.10]
    camera.distance = 0.75
    camera.azimuth = 135
    camera.elevation = -25
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render().copy()
        graphics = {}
        for name, constant in [("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                               ("version", GL.GL_VERSION)]:
            value = GL.glGetString(constant)
            graphics[name] = value.decode("utf-8", errors="replace") if value else None
    if pixels.shape != (720, 960, 3) or float(pixels.std()) < 1.0:
        raise RuntimeError("Rendered image is empty or has an unexpected shape.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(OUTPUT / "waffle_pi.png")
    return {"image": "results/stage1/waffle_pi.png", "shape": list(pixels.shape),
            "pixel_standard_deviation": float(pixels.std()), "opengl": graphics}


def run_headless(model, data, seconds, render):
    # 这里读取真实位姿仅用于模型加载评估；本阶段没有导航或反馈控制器。
    initial_qpos = data.qpos.copy()
    steps = math.ceil(seconds / model.opt.timestep)
    started = time.perf_counter()
    for _ in range(steps):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
        validate_state(model, data)
    report = {
        "status": "passed", "scope": "stage1_model_load_zero_input_physics_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps": steps, "simulation_seconds": float(data.time),
        "wall_seconds": time.perf_counter() - started,
        "initial_qpos": initial_qpos.tolist(), "final_qpos": data.qpos.tolist(),
        "final_qvel": data.qvel.tolist(), "warnings": validate_state(model, data),
        "navigation_estimation_or_control_verified": False,
    }
    if not math.isclose(float(data.time), steps * model.opt.timestep, abs_tol=1e-8):
        raise RuntimeError("Simulation clock did not advance as expected.")
    if render:
        report["rendering"] = render_image(model, data)
    save_json("verification.json", report)
    print(f"PASS: {steps} physics steps, {data.time:.3f} simulated seconds.", flush=True)


def run_viewer(model, data, seconds):
    import mujoco.viewer

    status = {"status": "starting", "requested_seconds": seconds,
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "sync_count": 0}
    save_json("viewer_status.json", status)
    try:
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                         show_right_ui=False) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [-0.04, 0, 0.10]
                viewer.cam.distance = 0.75
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
            viewer.sync()
            if not viewer.is_running():
                raise RuntimeError("Viewer closed before the first synchronization.")
            status["status"] = "opened"
            save_json("viewer_status.json", status)
            print("VIEWER_OPENED: passive viewer running; close window to finish.", flush=True)
            started = time.perf_counter()
            while viewer.is_running() and (seconds == 0 or time.perf_counter() - started < seconds):
                step_start = time.perf_counter()
                data.ctrl[:] = 0.0
                mujoco.mj_step(model, data)
                validate_state(model, data)
                viewer.sync()
                status["sync_count"] += 1
                time.sleep(max(0.0, model.opt.timestep - (time.perf_counter() - step_start)))
            status.update(status="completed", wall_seconds=time.perf_counter() - started,
                          simulation_seconds=float(data.time),
                          end_reason="time_limit" if viewer.is_running() else "window_closed")
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_json("viewer_status.json", status)
    print(f"VIEWER_COMPLETED: {status['sync_count']} synchronized steps.", flush=True)


def main():
    parser = argparse.ArgumentParser(description="41014 Track A - stage 1 official model verification")
    parser.add_argument("--headless", action="store_true", help="run physics without an interactive window")
    parser.add_argument("--seconds", type=float, default=None, help="0 keeps viewer open; headless requires >0")
    parser.add_argument("--render", action="store_true", help="save a PNG during headless verification")
    args = parser.parse_args()
    seconds = args.seconds if args.seconds is not None else (5.0 if args.headless else 0.0)
    if not math.isfinite(seconds) or seconds < 0 or (args.headless and seconds == 0):
        parser.error("seconds must be finite and non-negative; headless requires seconds > 0")
    if args.render and not args.headless:
        parser.error("--render is used with --headless")
    save_json("environment.json", {
        "python_version": sys.version, "python_executable": sys.executable,
        "base_python_executable": sys._base_executable, "python_prefix": sys.prefix,
        "platform": platform.platform(), "architecture": platform.machine(),
        "windows_version": list(platform.win32_ver()),
        "packages": {name: importlib.metadata.version(name)
                     for name in ["mujoco", "numpy", "Pillow", "glfw", "PyOpenGL"]},
    })
    model, data, source, hashes = load_model()
    inventory = inspect_model(model, source, hashes)
    print(f"MODEL_LOADED: {inventory['counts']}", flush=True)
    if args.headless:
        run_headless(model, data, seconds, args.render)
    else:
        run_viewer(model, data, seconds)


if __name__ == "__main__":
    main()
