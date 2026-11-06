"""编码器/已知点距离方位或图像导出位姿 EKF；无仿真器或真实位姿访问。"""

import math
import numpy as np

from evaluation.A_Modelling.code.state_model import (finite_vector, geometry, initialize_from_pair, landmark_model,
                         measurement_difference, wheel_step, wrap_angle)


def covariance(value, size, *, positive_definite=False):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError("Covariance has invalid shape or nonfinite entries.")
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError("Covariance must be symmetric.")
    matrix = (matrix + matrix.T) / 2
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() < -1e-12:
        raise ValueError("Covariance must be positive semidefinite.")
    if positive_definite:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError("Measurement covariance must be positive definite.") from error
    return matrix.copy()


def vision_covariance(count, parameters):
    """同帧独立分量+共享分量，顺序 [range,bearing,...]，单位 m/rad。"""
    if count not in (1, 2):
        raise ValueError("This stage supports one or two known markers per frame.")
    independent = finite_vector(parameters["independent_std_m_rad"], 2, "Vision standard deviations")
    common = finite_vector(parameters["common_std_m_rad"], 2, "Shared vision standard deviations")
    if np.any(independent <= 0) or np.any(common < 0):
        raise ValueError("Vision independent std must be positive; shared std nonnegative.")
    common_mapping = np.tile(np.eye(2), (count, 1))
    return np.diag(np.tile(independent**2, count)) + common_mapping @ np.diag(common**2) @ common_mapping.T


def initialize_pair_covariance(known_map, observations, measurement_covariance, floor_std_m_m_rad):
    """几何初值及一阶协方差传播；R 是声明的设计假设，不能称为标定结果。"""
    R = covariance(measurement_covariance, 4, positive_definite=True)
    floor = finite_vector(floor_std_m_m_rad, 3, "Initialization covariance floor")
    if np.any(floor < 0):
        raise ValueError("Initialization floor std must be nonnegative.")
    pose, diagnostics = initialize_from_pair(known_map, observations)
    z = np.array([v for item in observations for v in (item["range_m"], item["bearing_rad"])])
    J = np.empty((3, 4))
    epsilon = 1e-6
    for j in range(4):
        poses = []
        for sign in (1, -1):
            perturbed = z.copy()
            perturbed[j] += sign * epsilon
            items = [dict(item, range_m=float(perturbed[2*i]), bearing_rad=float(perturbed[2*i+1]))
                     for i, item in enumerate(observations)]
            poses.append(initialize_from_pair(known_map, items)[0])
        difference = poses[0] - poses[1]
        difference[2] = wrap_angle(difference[2])
        J[:, j] = difference / (2 * epsilon)
    P = covariance(J @ R @ J.T + np.diag(floor**2), 3)
    return pose, P, dict(diagnostics, jacobian=J.tolist(), R=R.tolist(),
                          covariance_basis="First-order propagation of design R plus declared floor; not empirical calibration.")


class LocalizationEKF:
    def __init__(self, initial_state, initial_covariance, known_map, wheel_radius_m, track_width_m,
                 encoder_angles_rad, time_s, process_parameters, *, initial_frame_used=False):
        geometry(wheel_radius_m, track_width_m)
        if not math.isfinite(time_s):
            raise ValueError("Initial time must be finite.")
        self.state = finite_vector(initial_state, 3, "Initial state").copy()
        self.P = covariance(initial_covariance, 3)
        self.known_map = {key: finite_vector(point, 2, "Known map point").copy() for key, point in known_map.items()}
        self.radius, self.track = wheel_radius_m, track_width_m
        self.previous_angles = finite_vector(encoder_angles_rad, 2, "Initial encoder angles").copy()
        self.time_s = float(time_s)
        self.last_vision_time_s = float(time_s) if initial_frame_used else -math.inf
        self.last_frame_id = None
        self.motion_coefficient = float(process_parameters["motion_variance_rad2_per_rad"])
        self.idle_coefficient = float(process_parameters["idle_variance_rad2_per_s"])
        self.model_std = finite_vector(process_parameters["model_std_m_m_rad_per_sqrt_s"], 3, "Model std")
        if not all(math.isfinite(v) and v >= 0 for v in [self.motion_coefficient, self.idle_coefficient, *self.model_std]):
            raise ValueError("Process noise parameters must be finite and nonnegative.")

    def predict_encoders(self, angles_rad, time_s):
        angles = finite_vector(angles_rad, 2, "Encoder measurement")
        if not math.isfinite(time_s) or time_s <= self.time_s:
            raise ValueError("Encoder times must increase strictly.")
        dt = time_s - self.time_s
        increments = angles - self.previous_angles
        next_state, F, G = wheel_step(self.state, increments, self.radius, self.track)
        # 方差依据可见的带噪增量近似；不读取理想轮角或真实位姿。
        Q_phi = np.diag(self.motion_coefficient * abs(increments) + self.idle_coefficient * dt)
        Q_model = np.diag(self.model_std**2 * dt)
        next_P = covariance(F @ self.P @ F.T + G @ Q_phi @ G.T + Q_model, 3)
        self.state, self.P = next_state, next_P
        self.previous_angles, self.time_s = angles.copy(), float(time_s)
        return {"dt_s": dt, "wheel_increments_rad": increments.tolist(), "F": F.tolist(), "G": G.tolist(),
                "Q_phi": Q_phi.tolist(), "Q_model": Q_model.tolist()}

    def update_vision(self, observations, time_s, frame_id, measurement_covariance, nis_limit):
        if not math.isfinite(time_s) or abs(time_s - self.time_s) > 1e-7:
            raise ValueError("Vision must match the current encoder time; delayed/future updates are unsupported.")
        if time_s <= self.last_vision_time_s + 1e-9 or frame_id == self.last_frame_id:
            raise ValueError("Vision frame already consumed or older than the last frame.")
        if not isinstance(frame_id, str) or not frame_id:
            raise ValueError("Use a nonempty frame identifier.")
        ids = [item["marker_id"] for item in observations]
        if len(ids) != len(set(ids)) or any(i not in self.known_map for i in ids):
            raise ValueError("Observation IDs must be unique known landmarks.")
        for item in observations:
            if not math.isfinite(item["time_s"]) or abs(item["time_s"] - time_s) > 1e-7:
                raise ValueError("All observations must belong to the current frame.")
            r, _ = finite_vector([item["range_m"], item["bearing_rad"]], 2, "Vision measurement")
            if r <= 0:
                raise ValueError("Observed range must be positive.")
        if not observations:
            self.last_vision_time_s, self.last_frame_id = float(time_s), frame_id
            return {"status": "no_observation", "ids": [], "nis": None}
        if not math.isfinite(nis_limit) or nis_limit <= 0:
            raise ValueError("NIS limit must be positive.")
        R = covariance(measurement_covariance, 2*len(observations), positive_definite=True)
        expected, jacobians = zip(*(landmark_model(self.state, self.known_map[i]) for i in ids))
        H = np.vstack(jacobians)
        z = np.array([v for item in observations for v in [item["range_m"], item["bearing_rad"]]])
        innovation = measurement_difference(z, np.concatenate(expected))
        S = H @ self.P @ H.T + R
        nis = float(innovation @ np.linalg.solve(S, innovation))
        diagnostic = {"ids": ids, "innovation_m_rad": innovation.tolist(), "H": H.tolist(),
                      "R": R.tolist(), "S": S.tolist(), "nis": nis, "nis_limit": nis_limit}
        if nis > nis_limit:
            self.last_vision_time_s, self.last_frame_id = float(time_s), frame_id
            return dict(diagnostic, status="rejected_nis")
        # solve 代替显式求逆，Joseph 形式保留更新协方差的数值稳定性。
        K = np.linalg.solve(S, H @ self.P).T
        next_state = self.state + K @ innovation
        residual_mapping = np.eye(3) - K @ H
        next_P = covariance(residual_mapping @ self.P @ residual_mapping.T + K @ R @ K.T, 3)
        self.state, self.P = next_state, next_P
        self.last_vision_time_s, self.last_frame_id = float(time_s), frame_id
        return dict(diagnostic, status="accepted", K=K.tolist())

    def update_pose(self, measured_pose, time_s, frame_id, measurement_covariance, nis_limit):
        """同一RGB导出的完整平面位姿，h(q)=q、H=I；不得再融合该帧距离/方位。"""
        if not math.isfinite(time_s) or abs(time_s-self.time_s)>1e-7:
            raise ValueError("Pose frame must match current encoder time.")
        if time_s<=self.last_vision_time_s+1e-9 or frame_id==self.last_frame_id:
            raise ValueError("Vision frame already consumed or stale.")
        if not isinstance(frame_id,str) or not frame_id:
            raise ValueError("Use a nonempty frame identifier.")
        if measured_pose is None:
            self.last_vision_time_s,self.last_frame_id=float(time_s),frame_id
            return {"status":"no_observation","measurement_kind":"image_planar_pose","nis":None}
        z=finite_vector(measured_pose,3,"Image-derived pose")
        R=covariance(measurement_covariance,3,positive_definite=True)
        if not math.isfinite(nis_limit) or nis_limit<=0:
            raise ValueError("NIS limit must be positive.")
        innovation=z-self.state
        innovation[2]=wrap_angle(innovation[2])
        S=self.P+R
        nis=float(innovation@np.linalg.solve(S,innovation))
        diagnostic={"measurement_kind":"image_planar_pose","measurement":z.tolist(),"H":np.eye(3).tolist(),
                    "innovation_m_m_rad":innovation.tolist(),"R":R.tolist(),"S":S.tolist(),"nis":nis,"nis_limit":nis_limit}
        self.last_vision_time_s,self.last_frame_id=float(time_s),frame_id
        if nis>nis_limit:
            return dict(diagnostic,status="rejected_nis")
        K=np.linalg.solve(S,self.P).T
        residual=np.eye(3)-K
        self.state=self.state+K@innovation
        self.P=covariance(residual@self.P@residual.T+K@R@K.T,3)
        return dict(diagnostic,status="accepted",K=K.tolist())
