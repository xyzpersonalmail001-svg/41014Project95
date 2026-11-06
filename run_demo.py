"""Continuous sensor-driven task; no historical results/evidence are required."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

from evaluation.common.release_support import (ROOT, digest, execution_paths, git_state, new_output,
                             read_json, save_manifest, verify_assets, write_json)


CASE_NAMES = {"normal": "normal_room", "obstacle": "appearing_obstacle",
              "blackout": "dock_visual_blackout"}


def nonnegative_seed(value):
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError("Seed must be a non-negative integer")
    return seed


def load_configuration(case_selection, seed):
    docking = read_json(ROOT / "configs/docking_check.json")
    config = dict(read_json(ROOT / docking["navigation_config"]), **docking)
    selected = list(CASE_NAMES.values()) if case_selection == "all" else [CASE_NAMES[case_selection]]
    config["cases"] = [deepcopy(c) for c in config["cases"] if c["name"] in selected]
    config["encoder_seed"] = seed
    ekf = read_json(ROOT / config["ekf_config"])
    ekf["encoder_noise"]["seed"] = seed
    tracking = read_json(ROOT / config["tracking_config"])
    camera = read_json(ROOT / ekf["camera_config"])
    camera["camera"]["position_in_base_m"] = config["camera_mount_override_m"]
    camera["additional_markers"] = read_json(ROOT / ekf["model_config"])["additional_markers"]
    room = read_json(ROOT / camera["lidar_config"])
    known = {str(item["id"]): item["position_world_m"][:2]
             for item in [camera["marker"], *camera["additional_markers"]]}
    return config, ekf, tracking, camera, room, known


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=[*CASE_NAMES, "all"], default="normal")
    parser.add_argument("--seed", type=nonnegative_seed, default=4101901,
                        help="Repeatable demo seed, not a newly registered validation sample")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/demo",
                        help="Parent directory; every attempt creates a unique child")
    args = parser.parse_args(argv)
    output = new_output(args.output_root)
    print(f"DEMO_OUTPUT: {output}", flush=True)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "status": "started", "python": sys.version, "platform": platform.platform(),
                "command": subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
                "case_selection": args.case, "seed": args.seed, "git": git_state(),
                "scope": "Demonstration / packaging check, not a statistical validation batch. "
                         "Offscreen RGB rendering still requires working OpenGL."}
    started = time.perf_counter()
    exit_code = 1
    write_json(output / "run.json", metadata)
    try:
        sources = execution_paths()
        before = {name: digest(ROOT / name) for name in sources}
        for name in sources:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write_json(output / "freeze.json", {"execution_source_sha256": before})
        metadata["assets"] = verify_assets()
        metadata["packages"] = {name: importlib.metadata.version(name) for name in
                                 ["mujoco", "numpy", "matplotlib", "pillow", "opencv-python-headless"]}
        from evaluation.E_Integration_Robustness.code.docking_check import check_close_images, make_plot, run_case
        config, ekf, tracking, camera, room, known = load_configuration(args.case, args.seed)
        if (camera["marker"]["position_world_m"] != config["station"]["origin_world_m"]
                or camera["marker"]["id"] != config["station"]["marker_id"]):
            raise ValueError("Station prior must match the configured fixed marker map")
        for name, value in [("config", config), ("ekf_config", ekf), ("tracking_config", tracking),
                            ("camera_config", camera), ("room_config", room), ("known_map", known)]:
            write_json(output / f"{name}.json", value)
        check_close_images(config, camera, room, output / "close_images")
        summaries = []
        for case in config["cases"]:
            result = run_case(case, config, ekf, tracking, camera, room,
                              {int(k): v for k, v in known.items()}, output / case["name"])
            summaries.append(result)
            write_json(output / "partial_results.json", summaries)
            metrics = result["evaluation"]["metrics"]
            print(f"{case['name']}: passed={result['evaluation']['passed']}, "
                  f"t={metrics['elapsed_seconds']:.2f}s, "
                  f"position={1000*metrics['final_position_error_m']:.3f}mm, "
                  f"yaw={metrics['final_yaw_error_degrees']:.3f}deg", flush=True)
        make_plot(summaries, output, config)
        unchanged = all(digest(ROOT / name) == expected for name, expected in before.items())
        passed = unchanged and all(s["evaluation"]["passed"] for s in summaries)
        write_json(output / "summary.json", {"passed": passed, "execution_sources_unchanged": unchanged,
                                            "cases": summaries, "scope": metadata["scope"]})
        metadata["status"], exit_code = ("passed", 0) if passed else ("failed", 1)
    except Exception as error:
        metadata.update(status="error", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(metadata["error"], file=sys.stderr, flush=True)
    finally:
        metadata.update(completed_at_utc=datetime.now(timezone.utc).isoformat(),
                        wall_elapsed_seconds=time.perf_counter() - started)
        write_json(output / "run.json", metadata)
        save_manifest(output)
    print(f"DEMO_{metadata['status'].upper()}: {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
