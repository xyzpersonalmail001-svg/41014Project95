"""视觉重获监督：连续、几何一致的双标记候选用于显式重初始化；不读真值。"""

import math
import numpy as np

from evaluation.C_Estimation.code.ekf import initialize_pair_covariance
from evaluation.A_Modelling.code.state_model import wrap_angle


class PairRecovery:
    def __init__(self,parameters,initial_time_s):
        self.parameters=dict(parameters)
        if not all(math.isfinite(value) and value>0 for value in parameters.values()):
            raise ValueError("Recovery parameters must be finite and positive.")
        if int(parameters["consecutive_frames"])!=parameters["consecutive_frames"] or parameters["consecutive_frames"]<2:
            raise ValueError("Recovery requires at least two consecutive frames.")
        self.last_accepted=float(initial_time_s)
        self.previous=None
        self.previous_time=None
        self.count=0

    def consider(self,observations,time_s,status,known_map,R,floor):
        if not math.isfinite(time_s):
            raise ValueError("Recovery time must be finite.")
        diagnostic={"status":"not_eligible","consecutive_candidates":0}
        if status=="accepted":
            self.last_accepted=float(time_s)
        eligible=status=="rejected_nis" and len(observations)==2 and time_s-self.last_accepted>=self.parameters["min_seconds_without_accepted_update"]
        if not eligible:
            self.previous,self.previous_time,self.count=None,None,0
            return None,diagnostic
        try:
            pose,P,details=initialize_pair_covariance(known_map,observations,R,floor)
            if details["baseline_relative_error"]>self.parameters["max_baseline_relative_error"]:
                raise ValueError("Pair baseline inconsistent for recovery.")
        except ValueError as error:
            self.previous,self.previous_time,self.count=None,None,0
            return None,{"status":"invalid_candidate","reason":str(error),"consecutive_candidates":0}
        consistent=self.previous is not None and time_s-self.previous_time<=self.parameters["max_frame_gap_s"] and np.linalg.norm(pose[:2]-self.previous[:2])<=self.parameters["max_candidate_step_m"] and abs(float(wrap_angle(pose[2]-self.previous[2])))<=self.parameters["max_candidate_step_rad"]
        self.count=self.count+1 if consistent else 1
        self.previous,self.previous_time=pose.copy(),float(time_s)
        diagnostic={"status":"candidate","consecutive_candidates":self.count,"state":pose.tolist(),"P":P.tolist(),"geometry":details}
        if self.count>=self.parameters["consecutive_frames"]:
            self.last_accepted=float(time_s)
            self.previous,self.previous_time,self.count=None,None,0
            diagnostic["status"]="confirmed_pair_recovery"
            return {"state":pose,"P":P},diagnostic
        return None,diagnostic
