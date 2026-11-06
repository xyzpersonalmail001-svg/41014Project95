"""仿真传感器层：增量独立噪声累计为轮角；估计器只收到 measured 输出。"""

import math
import numpy as np

from evaluation.A_Modelling.code.state_model import finite_vector


class IncrementalEncoder:
    def __init__(self, seed, motion_variance_rad2_per_rad, idle_variance_rad2_per_s):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Use an explicit nonnegative integer seed.")
        if not all(math.isfinite(v) and v >= 0 for v in [motion_variance_rad2_per_rad, idle_variance_rad2_per_s]):
            raise ValueError("Noise coefficients must be finite and nonnegative.")
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.motion, self.idle = motion_variance_rad2_per_rad, idle_variance_rad2_per_s
        self.previous_ideal = None
        self.previous_time = None
        self.accumulated_error = np.zeros(2)

    def sample(self, ideal_angles_rad, time_s):
        angles = finite_vector(ideal_angles_rad, 2, "Ideal wheel angles")
        if not math.isfinite(time_s) or (self.previous_time is not None and time_s <= self.previous_time):
            raise ValueError("Sample times must be finite and increase strictly.")
        delta = np.zeros(2) if self.previous_ideal is None else angles - self.previous_ideal
        dt = 0. if self.previous_time is None else time_s - self.previous_time
        variance = self.motion * abs(delta) + self.idle * dt
        normal = np.zeros(2) if self.previous_ideal is None else self.rng.standard_normal(2)
        error = np.sqrt(variance) * normal
        self.accumulated_error += error
        measured = angles + self.accumulated_error
        self.previous_ideal, self.previous_time = angles.copy(), float(time_s)
        return measured, {"ideal_angles_rad": angles.tolist(), "ideal_increment_rad": delta.tolist(),
                          "increment_variance_rad2": variance.tolist(), "standard_normal": normal.tolist(),
                          "increment_error_rad": error.tolist(), "accumulated_error_rad": self.accumulated_error.tolist()}
