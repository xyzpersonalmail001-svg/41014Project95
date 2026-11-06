"""RGB 图像 -> ArUco 角点 -> PnP -> 机身距离/方位；不导入 MuJoCo。"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


# OpenCV camera: x right, y down, z forward. Base: x forward, y left, z up.
ROTATION_BASE_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])


def camera_matrix(width, height, fovy_degrees):
    if not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0:
        raise ValueError("Camera resolution must contain positive integers.")
    if not math.isfinite(fovy_degrees) or not 0 < fovy_degrees < 180:
        raise ValueError("Vertical field of view must be between 0 and 180 degrees.")
    focal = height / (2 * math.tan(math.radians(fovy_degrees) / 2))
    return np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])


def marker_corners(side_length_m):
    if not math.isfinite(side_length_m) or side_length_m <= 0:
        raise ValueError("Marker black-border side must be positive.")
    half = side_length_m / 2
    return np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)


@dataclass
class VisionObservation:
    time_s: float
    marker_id: int
    corners_px: np.ndarray
    marker_translation_camera_m: np.ndarray
    marker_rotation_vector: np.ndarray
    marker_position_base_m: np.ndarray
    range_m: float
    bearing_rad: float
    reprojection_rmse_px: float


class ArucoVision:
    def __init__(self, config):
        camera, marker = config["camera"], config["marker"]
        self.width, self.height = camera["width"], camera["height"]
        self.K = camera_matrix(self.width, self.height, camera["fovy_degrees"])
        self.distortion = np.zeros(5)
        self.translation_base_camera = np.asarray(camera["position_in_base_m"], dtype=float)
        if self.translation_base_camera.shape != (3,) or not np.isfinite(self.translation_base_camera).all():
            raise ValueError("Camera mounting translation must have three finite values.")
        self.object_points = marker_corners(marker["side_length_m"])
        self.target_id = marker["id"]
        markers = [marker, *config.get("additional_markers", [])]
        if len({item["id"] for item in markers}) != len(markers):
            raise ValueError("Configured marker IDs must be unique.")
        if any(item["dictionary"] != marker["dictionary"] for item in markers):
            raise ValueError("All markers must use the same dictionary.")
        self.points_by_id = {item["id"]: marker_corners(item["side_length_m"]) for item in markers}
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, marker["dictionary"]))
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.quality = dict(config["quality"])

    def estimate_from_corners(self, corners_px, time_s, marker_id=None):
        identifier = self.target_id if marker_id is None else marker_id
        if identifier not in self.points_by_id:
            return None
        object_points = self.points_by_id[identifier]
        points = np.asarray(corners_px, dtype=np.float64).reshape(4, 2)
        if not np.isfinite(points).all() or not math.isfinite(time_s):
            raise ValueError("Corners and timestamp must be finite.")
        side_pixels = np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1)
        if side_pixels.min() < self.quality["min_marker_side_px"]:
            return None
        # 迭代式平面 PnP；对称正视案例也必须通过实际重投影检查。
        result = cv2.solvePnPGeneric(object_points, points, self.K, self.distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        candidates = []
        if result[0]:
            for rotation_vector, translation_vector in zip(result[1], result[2]):
                rotation_vector, translation = rotation_vector.reshape(3), translation_vector.reshape(3)
                if not np.isfinite(rotation_vector).all() or not np.isfinite(translation).all() or translation[2] <= 0:
                    continue
                projected, _ = cv2.projectPoints(object_points, rotation_vector, translation, self.K, self.distortion)
                error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - points) ** 2, axis=1))))
                candidates.append((error, rotation_vector, translation))
        if not candidates:
            return None
        error, rotation_vector, translation = min(candidates, key=lambda item: item[0])
        position_base = ROTATION_BASE_CAMERA @ translation + self.translation_base_camera
        planar_range = float(np.hypot(position_base[0], position_base[1]))
        if error > self.quality["max_reprojection_rmse_px"] or not self.quality["range_min_m"] <= planar_range <= self.quality["range_max_m"]:
            return None
        return VisionObservation(float(time_s), identifier, points.copy(), translation, rotation_vector, position_base,
                                 planar_range, math.atan2(position_base[1], position_base[0]), error)

    def detect(self, rgb, time_s):
        image = np.asarray(rgb)
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError("Expected calibrated-resolution uint8 RGB image.")
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        observations = []
        if ids is not None:
            for points, identifier in zip(corners, ids.ravel()):
                if int(identifier) in self.points_by_id:
                    observation = self.estimate_from_corners(points, time_s, int(identifier))
                    if observation is not None:
                        observations.append(observation)
        return observations
