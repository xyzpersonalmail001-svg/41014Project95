"""Bounded, stationary image-pose reanchoring; no simulator or ground-truth input."""

import math
import numpy as np

from evaluation.C_Estimation.code.ekf import covariance
from evaluation.A_Modelling.code.state_model import finite_vector, wrap_angle


class NoPathRecovery:
    def __init__(self, parameters):
        self.p = dict(parameters)
        if not all(math.isfinite(v) and v > 0 for v in self.p.values()):
            raise ValueError("Recovery parameters must be finite and positive.")
        for key in ("consecutive_frames", "max_attempts"):
            if int(self.p[key]) != self.p[key] or (key == "consecutive_frames" and self.p[key] < 2):
                raise ValueError("Invalid recovery count.")
        self.mode = "IDLE"
        self.blocked_since = None
        self.started = None
        self.cooldown_until = -math.inf
        self.attempts = 0
        self.previous = None
        self.previous_time = None
        self.count = 0
        self.full_pose_navigation = False
        self.events = []

    def _clear_candidates(self):
        self.previous, self.previous_time, self.count = None, None, 0

    def supervise(self, command, time_s):
        """Called at the outer control rate. Recovery never issues a moving command."""
        if not math.isfinite(time_s):
            raise ValueError("Finite recovery time required.")
        if self.mode == "COLLECT" and time_s - self.started >= self.p["attempt_timeout_s"]:
            self.events.append({"time_s": time_s, "event": "attempt_timeout", "attempt": self.attempts})
            self.mode = "EXHAUSTED" if self.attempts >= self.p["max_attempts"] else "IDLE"
            self.cooldown_until = time_s + self.p["cooldown_s"]
            self.blocked_since = None
            self._clear_candidates()
        if self.mode == "IDLE":
            if command["state"] == "NO_PATH_STOP":
                self.blocked_since = time_s if self.blocked_since is None else self.blocked_since
                if time_s >= self.cooldown_until and time_s - self.blocked_since >= self.p["persistent_stop_s"]:
                    if self.attempts >= self.p["max_attempts"]:
                        self.mode = "EXHAUSTED"
                    else:
                        self.mode, self.started = "COLLECT", time_s
                        self.attempts += 1
                        self._clear_candidates()
                        self.events.append({"time_s": time_s, "event": "attempt_started", "attempt": self.attempts})
            else:
                self.blocked_since = None
        if self.mode in ("COLLECT", "EXHAUSTED"):
            result = dict(command)
            result.update(state="RECOVERY_WAIT_POSE" if self.mode == "COLLECT" else "RECOVERY_EXHAUSTED_STOP",
                          body_twist=[0., 0.], desired_wheel_rad_s=[0., 0.], stop=True, arrived=False)
        else:
            result = dict(command)
        result["no_path_recovery"] = {"mode": self.mode, "attempts": self.attempts,
                                      "full_pose_navigation": self.full_pose_navigation}
        return result

    def consider(self, pose, R, estimated_pose, measured_rates, time_s, scan_time_s):
        """Previous frames check consistency only; the confirmation frame initializes once."""
        if self.mode != "COLLECT":
            return None, {"status": "inactive"}
        valid = (self.started + self.p["settle_stop_s"] <= time_s < self.started + self.p["attempt_timeout_s"]
                 and scan_time_s is not None and 0 <= time_s - scan_time_s <= self.p["max_scan_age_s"]
                 and np.max(abs(finite_vector(measured_rates, 2, "Measured wheel rates"))) <= self.p["max_stationary_wheel_rad_s"])
        if not valid or pose is None:
            self._clear_candidates()
            return None, {"status": "waiting_stationary_fresh_pose", "count": 0}
        pose = finite_vector(pose, 3, "Oriented image pose")
        R = covariance(R, 3, positive_definite=True)
        delta = pose - finite_vector(estimated_pose, 3, "Estimated pose")
        delta[2] = wrap_angle(delta[2])
        quality = (math.sqrt(np.linalg.eigvalsh(R[:2, :2]).max()) <= self.p["max_pose_position_std_m"]
                   and math.sqrt(R[2, 2]) <= self.p["max_pose_yaw_std_rad"]
                   and np.linalg.norm(delta[:2]) <= self.p["max_reanchor_position_m"]
                   and abs(delta[2]) <= self.p["max_reanchor_yaw_rad"])
        if not quality:
            self._clear_candidates()
            return None, {"status": "candidate_quality_rejected", "count": 0}
        gap = None if self.previous_time is None else time_s - self.previous_time
        if gap is not None and gap <= 0:
            raise ValueError("Recovery frames must be fresh and increasing.")
        consistent = (self.previous is not None and gap <= self.p["max_frame_gap_s"]
                      and np.linalg.norm(pose[:2] - self.previous[:2]) <= self.p["max_candidate_step_m"]
                      and abs(float(wrap_angle(pose[2] - self.previous[2]))) <= self.p["max_candidate_step_rad"])
        self.count = self.count + 1 if consistent else 1
        self.previous, self.previous_time = pose.copy(), time_s
        diagnostic = {"status": "candidate", "count": self.count, "attempt": self.attempts}
        if self.count < self.p["consecutive_frames"]:
            return None, diagnostic
        diagnostic["status"] = "confirmed_oriented_pose_recovery"
        self.events.append({"time_s": time_s, "event": diagnostic["status"], "attempt": self.attempts,
                            "state": pose.tolist(), "P": R.tolist(), "correction": delta.tolist()})
        self.mode, self.blocked_since = "IDLE", None
        self.cooldown_until = time_s + self.p["cooldown_s"]
        self.full_pose_navigation = True
        self._clear_candidates()
        return {"state": pose.copy(), "P": R.copy()}, diagnostic
