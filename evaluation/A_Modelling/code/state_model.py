"""平面差速运动与点地标测量模型；只接收测量/地图先验，不访问仿真器。"""

import math
import numpy as np


def finite_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values.")
    return result


def wrap_angle(angle):
    """角度差规范到 [-pi, pi)；连续累计的状态朝向不需要强制折回。"""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def geometry(radius_m, track_m):
    if not all(math.isfinite(v) and v > 0 for v in (radius_m, track_m)):
        raise ValueError("Wheel radius and track must be finite and positive.")


def continuous_model(state, twist):
    """s=[x(m),y(m),yaw(rad)]，u=[v(m/s),omega(rad/s)]；返回导数、A、B。"""
    _, _, yaw = finite_vector(state, 3, "State")
    v, omega = finite_vector(twist, 2, "Body twist")
    c, s = math.cos(yaw), math.sin(yaw)
    rate = np.array([v * c, v * s, omega])
    A = np.array([[0., 0., -v * s], [0., 0., v * c], [0., 0., 0.]])
    B = np.array([[c, 0.], [s, 0.], [0., 1.]])
    return rate, A, B


def twist_to_wheel_rates(twist, radius_m, track_m):
    """理想运动学逆解；输出并不保证速度执行器实际达到该轮速。"""
    geometry(radius_m, track_m)
    v, omega = finite_vector(twist, 2, "Body twist")
    return np.array([(v - track_m * omega / 2) / radius_m,
                     (v + track_m * omega / 2) / radius_m])


def wheel_step(state, wheel_increments, radius_m, track_m):
    """常曲率精确积分。返回 next_state(3), F(3,3), G(3,2)。

    wheel_increments 是实际编码器角度差 [left,right]，单位 rad；不是轮速目标。
    F=∂next/∂state，G=∂next/∂wheel_increments，供后续 EKF 使用。
    """
    geometry(radius_m, track_m)
    state = finite_vector(state, 3, "State")
    left, right = finite_vector(wheel_increments, 2, "Wheel increments")
    distance = radius_m * (left + right) / 2
    turn = radius_m * (right - left) / track_m
    if abs(turn) < 1e-4:
        # 泰勒展开避免很小转角时除以零或浮点相消。
        chord = 1 - turn**2 / 24 + turn**4 / 1920
        chord_derivative = -turn / 12 + turn**3 / 480
    else:
        half = turn / 2
        chord = math.sin(half) / half
        chord_derivative = (half * math.cos(half) - math.sin(half)) / (2 * half**2)
    middle = state[2] + turn / 2
    c, s = math.cos(middle), math.sin(middle)
    result = state + [distance * chord * c, distance * chord * s, turn]
    F = np.eye(3)
    F[:2, 2] = [-distance * chord * s, distance * chord * c]
    turn_derivative = np.array([-radius_m / track_m, radius_m / track_m])
    G = np.empty((3, 2))
    G[0] = radius_m / 2 * chord * c + distance * turn_derivative * (chord_derivative * c - chord * s / 2)
    G[1] = radius_m / 2 * chord * s + distance * turn_derivative * (chord_derivative * s + chord * c / 2)
    G[2] = turn_derivative
    return result, F, G


def landmark_model(state, landmark_xy):
    """地图中已知点 -> 机身水平距离(m)/方位(rad)，以及 H(2,3)。"""
    x, y, yaw = finite_vector(state, 3, "State")
    mx, my = finite_vector(landmark_xy, 2, "Known map point")
    dx, dy = mx - x, my - y
    squared = dx * dx + dy * dy
    if squared < 1e-12:
        raise ValueError("Bearing is undefined at the landmark centre.")
    distance = math.sqrt(squared)
    measurement = np.array([distance, float(wrap_angle(math.atan2(dy, dx) - yaw))])
    H = np.array([[-dx / distance, -dy / distance, 0.], [dy / squared, -dx / squared, -1.]])
    return measurement, H


def measurement_difference(first, second):
    """向量按 [range,bearing,range,bearing,...] 排列，仅角度差折回。"""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if first.ndim != 1 or first.shape != second.shape or first.size % 2 or not np.isfinite([first, second]).all():
        raise ValueError("Measurements must be matching finite even-length vectors.")
    difference = first - second
    difference[1::2] = wrap_angle(difference[1::2])
    return difference


def initialize_from_pair(known_map, observations, max_baseline_relative_error=0.25):
    """两已知地图点与同帧 range/bearing -> 位姿初值；不读取任何真值。

    这是几何初始化，不是滤波器，也不输出已标定的协方差。
    observations: 两个 {marker_id, range_m, bearing_rad, time_s} 字典。
    """
    if len(observations) != 2 or not math.isfinite(max_baseline_relative_error) or max_baseline_relative_error <= 0:
        raise ValueError("Need exactly two observations and a positive consistency threshold.")
    ids = [item["marker_id"] for item in observations]
    if ids[0] == ids[1] or any(identifier not in known_map for identifier in ids):
        raise ValueError("Need two different known landmark IDs.")
    timestamps = np.array([item["time_s"] for item in observations], dtype=float)
    if not np.isfinite(timestamps).all() or abs(timestamps[0] - timestamps[1]) > 1e-9:
        raise ValueError("Pair initialization requires simultaneous observations.")
    world = np.array([finite_vector(known_map[i], 2, "Map point") for i in ids])
    local = []
    for item in observations:
        r, b = finite_vector([item["range_m"], item["bearing_rad"]], 2, "Observation")
        if r <= 0:
            raise ValueError("Observed range must be positive.")
        local.append([r * math.cos(b), r * math.sin(b)])
    local = np.asarray(local)
    baseline_world, baseline_local = world[1] - world[0], local[1] - local[0]
    length_world, length_local = np.linalg.norm(baseline_world), np.linalg.norm(baseline_local)
    if min(length_world, length_local) < 1e-6:
        raise ValueError("Coincident map or observed points cannot determine yaw.")
    relative_error = abs(length_local - length_world) / length_world
    if relative_error > max_baseline_relative_error:
        raise ValueError("Observed pair baseline is inconsistent with the known map.")
    yaw = float(wrap_angle(math.atan2(*baseline_world[::-1]) - math.atan2(*baseline_local[::-1])))
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    position = np.mean(world - local @ rotation.T, axis=0)
    return np.array([*position, yaw]), {"baseline_world_m": float(length_world),
                                       "baseline_observed_m": float(length_local),
                                       "baseline_relative_error": float(relative_error)}
