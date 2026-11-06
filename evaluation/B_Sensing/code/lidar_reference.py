"""验收专用解析几何，不调用 MuJoCo 射线函数，不供控制器使用。"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def read_reference_geometries(path):
    objects = []
    for element in ET.parse(path).getroot().findall("worldbody/geom"):
        if any(key in element.attrib for key in ["quat", "euler", "axisangle", "fromto"]):
            raise ValueError("This independent reference expects axis-aligned room geoms.")
        kind = element.get("type")
        if kind not in {"box", "cylinder", "plane"}:
            raise ValueError(f"Reference intersection not implemented for {kind}.")
        objects.append({"name": element.get("name"), "type": kind,
                        "position_m": [float(v) for v in element.get("pos", "0 0 0").split()],
                        "size_m": [float(v) for v in element.get("size").split()]})
    return objects


def intersection_distance(origin, direction, item):
    """独立 slab/二次方程/平面交点；返回最近非负表面参数。"""
    p = np.asarray(origin) - np.asarray(item["position_m"])
    d = np.asarray(direction)
    size = item["size_m"]
    candidates = []
    if item["type"] == "plane":
        if abs(d[2]) > 1e-14:
            candidates.append(-p[2] / d[2])
    elif item["type"] == "box":
        lower, upper = -math.inf, math.inf
        for coordinate, slope, half in zip(p, d, size):
            if abs(slope) < 1e-14:
                if abs(coordinate) > half:
                    return math.inf
            else:
                first, second = sorted(((-half - coordinate) / slope, (half - coordinate) / slope))
                lower, upper = max(lower, first), min(upper, second)
        if lower <= upper:
            candidates.extend([lower, upper])
    else:
        radius, half_height = size
        a = float(d[0] ** 2 + d[1] ** 2)
        b = float(2 * (p[0] * d[0] + p[1] * d[1]))
        c = float(p[0] ** 2 + p[1] ** 2 - radius ** 2)
        discriminant = b * b - 4 * a * c
        if a > 1e-14 and discriminant >= 0:
            for t in [(-b - math.sqrt(discriminant)) / (2 * a), (-b + math.sqrt(discriminant)) / (2 * a)]:
                if abs(p[2] + t * d[2]) <= half_height + 1e-12:
                    candidates.append(t)
        if abs(d[2]) > 1e-14:
            for z in [-half_height, half_height]:
                t = (z - p[2]) / d[2]
                if np.linalg.norm(p[:2] + t * d[:2]) <= radius + 1e-12:
                    candidates.append(t)
    return min((t for t in candidates if t >= 0), default=math.inf)


def expected_scan(origin, directions, objects, range_min, range_max):
    ranges, statuses = [], []
    for direction in directions:
        distance = min(intersection_distance(origin, direction, item) for item in objects)
        if not math.isfinite(distance) or distance > range_max:
            ranges.append(range_max)
            statuses.append(0)
        elif distance < range_min:
            ranges.append(range_min)
            statuses.append(2)
        else:
            ranges.append(distance)
            statuses.append(1)
    return np.asarray(ranges), np.asarray(statuses, dtype=np.uint8)
