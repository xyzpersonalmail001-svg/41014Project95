"""在内存副本中扩展官方模型；不修改第三方 XML 或网格。"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from evaluation.common.main import ROOT


def load_room(config, *, robot_editor=None, scene_editor=None, extra_assets=None):
    source = json.loads((ROOT / "resources.lock.json").read_text(encoding="utf-8"))["robotis"]
    directory = ROOT / source["directory"] / "robotis_tb3"
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in {".xml", ".stl", ".obj"})
    assets = {p.relative_to(directory).as_posix(): p.read_bytes() for p in files}
    original_hashes = {name: hashlib.sha256(raw).hexdigest() for name, raw in assets.items()}
    robot = ET.fromstring(assets["turtlebot3_waffle_pi.xml"])
    base = robot.find("worldbody/body[@name='base']")
    if base is None:
        raise ValueError("Expected the pinned official robot base body.")
    # mj_ray 的 bodyexclude 只排除一个 body；用分组排除整个机器人及子 body。
    for geom in base.iter("geom"):
        geom.set("group", str(config["robot_geom_group"]))
    ET.SubElement(base, "site", {
        "name": config["site_name"], "pos": " ".join(map(str, config["sensor_position_in_base_m"])),
        "quat": " ".join(map(str, config["sensor_quaternion_in_base_wxyz"])),
        "size": "0.012", "rgba": "0.1 0.9 0.2 1"})
    if robot_editor is not None:
        robot_editor(base)
    derived_robot = ET.tostring(robot, encoding="utf-8")
    assets["turtlebot3_waffle_pi.xml"] = derived_robot
    scene_path = ROOT / config["scene"]
    scene_xml = scene_path.read_bytes()
    original_scene_xml = scene_xml
    if scene_editor is not None:
        scene_root = ET.fromstring(scene_xml)
        scene_editor(scene_root)
        scene_xml = ET.tostring(scene_root, encoding="utf-8")
    if extra_assets:
        assets.update(extra_assets)
    model = mujoco.MjModel.from_xml_string(scene_xml.decode("utf-8"), assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    provenance = {"official_resource": source, "official_resource_sha256": original_hashes,
                  "project_scene": config["scene"], "project_scene_sha256": hashlib.sha256(original_scene_xml).hexdigest(),
                  "derived_robot_sha256": hashlib.sha256(derived_robot).hexdigest(),
                  "changes_in_memory_only": ["All robot geoms assigned ray-excluded group", "Massless lidar site added to base"],
                  "asset_loading": "Python bytes and MuJoCo VFS for Unicode paths"}
    if scene_editor is not None or robot_editor is not None:
        provenance["derived_scene_sha256"] = hashlib.sha256(scene_xml).hexdigest()
    return model, data, provenance, derived_robot
