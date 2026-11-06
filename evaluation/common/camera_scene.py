"""将真实可渲染的 ArUco 黑白格和机载相机加入项目场景。"""

import xml.etree.ElementTree as ET

import cv2
import numpy as np

from evaluation.common.room_scene import load_room


def load_camera_scene(room_config, config):
    marker = config["marker"]
    markers = [marker, *config.get("additional_markers", [])]
    if any(item["dictionary"] != "DICT_4X4_50" for item in markers):
        raise ValueError("The physical 6x6 board builder currently supports DICT_4X4_50 only.")
    if len({item["id"] for item in markers}) != len(markers):
        raise ValueError("Scene marker IDs must be unique.")
    derived_scene = []

    def edit_robot(base):
        camera = config["camera"]
        ET.SubElement(base, "camera", {
            "name": camera["name"], "pos": " ".join(map(str, camera["position_in_base_m"])),
            "xyaxes": "0 -1 0 0 0 1", "fovy": str(camera["fovy_degrees"]),
            "resolution": f"{camera['width']} {camera['height']}"})

    def edit_scene(scene):
        world = scene.find("worldbody")
        for index, item in enumerate(markers):
            suffix = "" if index == 0 else f"_{item['id']}"
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, item["dictionary"]))
            # 4x4 数据加一格黑边共 6x6；每格是实际渲染表面。
            grid = cv2.aruco.generateImageMarker(dictionary, item["id"], 6, borderBits=1)
            board = ET.SubElement(world, "body", {"name": f"charging_marker{suffix}", "pos": " ".join(map(str, item["position_world_m"])),
                                                 "xyaxes": "0 -1 0 0 0 1"})
            half_board = item["side_length_m"] / 2 + item["quiet_margin_m"]
            ET.SubElement(board, "geom", {"name": f"marker_white_board{suffix}", "type": "box", "pos": "0 0 -0.0015",
                                           "size": f"{half_board} {half_board} 0.001", "rgba": "1 1 1 1", "group": "0"})
            cell = item["side_length_m"] / 6
            for row, column in np.argwhere(grid == 0):
                x, y = (column + 0.5 - 3) * cell, (3 - row - 0.5) * cell
                ET.SubElement(board, "geom", {"name": f"marker_cell_{row}_{column}{suffix}", "type": "box",
                                               "pos": f"{x} {y} -0.00025", "size": f"{cell / 2} {cell / 2} 0.00025",
                                               "rgba": "0 0 0 1", "contype": "0", "conaffinity": "0", "density": "0", "group": "0"})
        occluder = ET.SubElement(world, "body", {"name": "vision_occluder", "mocap": "true", "pos": "0 0 -10"})
        ET.SubElement(occluder, "geom", {"name": "vision_occluder_box", "type": "box", "size": "0.05 0.35 0.35",
                                         "rgba": "0.25 0.35 0.4 1", "group": "0"})
        derived_scene.append(ET.tostring(scene, encoding="utf-8"))

    model, data, provenance, robot_xml = load_room(room_config, robot_editor=edit_robot, scene_editor=edit_scene)
    provenance["changes_in_memory_only"].extend(["Body-fixed forward camera", "ArUco station made from dictionary-generated black cells and white board", "Movable test occluder"])
    return model, data, provenance, robot_xml, derived_scene[0]
