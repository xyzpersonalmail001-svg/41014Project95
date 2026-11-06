"""理想二维 LiDAR 测量生成器；公开测量不携带真实位姿或命中物体 ID。"""

from dataclasses import dataclass
import math

import mujoco
import numpy as np


NO_RETURN, HIT, TOO_CLOSE = 0, 1, 2


@dataclass
class LidarScan:
    time_s: float
    frame_id: str
    angles_rad: np.ndarray
    ranges_m: np.ndarray
    status: np.ndarray


class Lidar2D:
    def __init__(self, model, site_name="lidar_site", *, beam_count=360, angle_start_rad=-math.pi,
                 range_min_m=0.05, range_max_m=4.0, robot_geom_group=1):
        if isinstance(beam_count, bool) or not isinstance(beam_count, int) or beam_count < 4:
            raise ValueError("Use an integer beam count >= 4.")
        if not all(math.isfinite(v) for v in [angle_start_rad, range_min_m, range_max_m]):
            raise ValueError("LiDAR angles and limits must be finite.")
        if not 0 <= range_min_m < range_max_m or robot_geom_group not in range(1, 6):
            raise ValueError("Require 0 <= minimum < maximum and robot group in 1..5.")
        self.model = model
        self.site_id = model.site(site_name).id
        self.frame_id = site_name
        self.range_min_m, self.range_max_m = float(range_min_m), float(range_max_m)
        self.angles_rad = angle_start_rad + np.arange(beam_count) * (2 * math.pi / beam_count)
        self.directions_sensor = np.column_stack((np.cos(self.angles_rad), np.sin(self.angles_rad), np.zeros(beam_count)))
        self.geom_groups = np.ones(6, dtype=np.uint8)
        self.geom_groups[robot_geom_group] = 0

    def scan(self, data):
        """应用接口：机器人局部扫描平面内的角度/距离/状态，无真实世界位姿。"""
        measurement, _ = self.simulator_scan_with_diagnostics(data)
        return measurement

    def simulator_scan_with_diagnostics(self, data):
        """仿真生成与验收接口；第二项诊断只能用于记录、绘图和检查。"""
        origin_world = data.site_xpos[self.site_id].copy()
        rotation_world_sensor = data.site_xmat[self.site_id].reshape(3, 3).copy()
        # 列向量公式 d_world=R*d_sensor；数组按行存射线，所以写成 D*R.T。
        directions_world = self.directions_sensor @ rotation_world_sensor.T
        raw_distances, geom_ids = [], []
        for direction in directions_world:
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(self.model, data, origin_world, direction,
                                     self.geom_groups, True, -1, geom_id)
            raw_distances.append(distance)
            geom_ids.append(int(geom_id[0]))
        raw = np.asarray(raw_distances)
        status = np.full(len(raw), NO_RETURN, dtype=np.uint8)
        hit = (raw >= self.range_min_m) & (raw <= self.range_max_m)
        too_close = (raw >= 0) & (raw < self.range_min_m)
        status[hit], status[too_close] = HIT, TOO_CLOSE
        ranges = np.full(len(raw), self.range_max_m)
        ranges[hit], ranges[too_close] = raw[hit], self.range_min_m
        measurement = LidarScan(float(data.time), self.frame_id, self.angles_rad.copy(), ranges, status)
        diagnostics = {"origin_world_m": origin_world, "rotation_world_sensor": rotation_world_sensor,
                       "directions_world": directions_world, "raw_distances_m": raw,
                       "geom_ids": np.asarray(geom_ids, dtype=np.int32)}
        return measurement, diagnostics
