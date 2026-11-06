"""已知朝向标记的图像位姿、相关协方差与估计状态极坐标停靠；不访问MuJoCo。"""

import math
import cv2
import numpy as np

from evaluation.C_Estimation.code.ekf import covariance
from evaluation.E_Integration_Robustness.code.navigation import scan_guard
from evaluation.A_Modelling.code.state_model import finite_vector,twist_to_wheel_rates,wrap_angle
from evaluation.B_Sensing.code.vision import ROTATION_BASE_CAMERA


def station_goal(station):
    origin=finite_vector(station["origin_world_m"],3,"Declared station origin")
    offset=finite_vector(station["goal_offset_m_m_rad"],3,"Station docking offset")
    yaw=float(station["yaw_world_rad"])
    if not math.isfinite(yaw):
        raise ValueError("Station yaw must be finite.")
    c,s=math.cos(yaw),math.sin(yaw)
    return np.r_[origin[:2]+np.array([[c,-s],[s,c]])@offset[:2],float(wrap_angle(yaw+offset[2]))]


def marker_image_pose(observation,station,mount_xyz):
    """T_wb=T_wm inv(T_bm)，声明完整标记姿态先验，输入来自PnP而非世界真值。"""
    if observation.marker_id!=station["marker_id"]:
        raise ValueError("Wrong docking marker ID.")
    Rwm=np.asarray(station["marker_rotation_world"],dtype=float)
    if Rwm.shape!=(3,3) or not np.isfinite(Rwm).all() or not np.allclose(Rwm.T@Rwm,np.eye(3),atol=1e-10) or np.linalg.det(Rwm)<.999999:
        raise ValueError("Declared marker rotation must be in SO(3).")
    rotation,_=cv2.Rodrigues(finite_vector(observation.marker_rotation_vector,3,"PnP rotation vector"))
    Rbm=ROTATION_BASE_CAMERA@rotation
    pbm=ROTATION_BASE_CAMERA@finite_vector(observation.marker_translation_camera_m,3,"PnP translation")+finite_vector(mount_xyz,3,"Camera mount")
    Rwb=Rwm@Rbm.T
    position=finite_vector(station["origin_world_m"],3,"Station marker origin")-Rwb@pbm
    yaw=math.atan2(Rwb[1,0],Rwb[0,0])
    tilt=math.acos(float(np.clip(Rwb[2,2],-1,1)))
    return np.r_[position[:2],yaw],{"base_height_image_m":float(position[2]),"base_tilt_image_rad":tilt}


def pose_measurement(observation,detector,station,parameters):
    """数值传播8个角点坐标的设计像素方差；保留位姿分量之间相关性。"""
    pose,geometry=marker_image_pose(observation,station,detector.translation_base_camera)
    if geometry["base_tilt_image_rad"]>parameters["max_image_tilt_rad"] or abs(geometry["base_height_image_m"])>parameters["max_image_base_height_m"]:
        raise ValueError("Image pose violates declared planar docking envelope.")
    corners=np.asarray(observation.corners_px).reshape(8)
    step=parameters["corner_difference_step_px"]
    J=np.empty((3,8))
    for column in range(8):
        samples=[]
        for sign in (1,-1):
            perturbed=corners.copy()
            perturbed[column]+=sign*step
            item=detector.estimate_from_corners(perturbed.reshape(4,2),observation.time_s,observation.marker_id)
            if item is None:
                raise ValueError("Cannot propagate corner uncertainty through PnP.")
            samples.append(marker_image_pose(item,station,detector.translation_base_camera)[0])
        difference=samples[0]-samples[1]
        difference[2]=wrap_angle(difference[2])
        J[:,column]=difference/(2*step)
    floor=finite_vector(parameters["pose_floor_std_m_m_rad"],3,"Pose model floor")
    R=covariance(J@J.T*parameters["corner_std_px"]**2+np.diag(floor**2),3,positive_definite=True)
    return pose,R,dict(geometry,jacobian_pose_pixels=J.tolist(),covariance_basis="Independent design corner variance propagated through PnP plus design floor; same-frame full correlation retained, not empirical calibration.")


class DockingController:
    def __init__(self,parameters,station,radius_m,track_m,mount_xy):
        self.parameters=dict(parameters)
        for key,value in parameters.items():
            if not math.isfinite(value) or value<=0:
                raise ValueError(f"Invalid docking parameter {key}.")
        if not parameters["k_alpha_per_s"]>parameters["k_rho_per_s"]:
            raise ValueError("Local polar design needs k_alpha > k_rho > 0.")
        # beta gain is stored as a positive magnitude and subtracted in the law.
        self.goal=station_goal(station)
        self.station_xy=np.asarray(station["origin_world_m"][:2],dtype=float)
        self.retreat_goal=self.goal.copy()
        self.retreat_goal[:2]-=parameters["retreat_distance_m"]*np.array([math.cos(self.goal[2]),math.sin(self.goal[2])])
        self.radius,self.track=radius_m,track_m
        self.mount=finite_vector(mount_xy,2,"Lidar mount")
        self.last_pose_time=None
        self.done_since=None
        self.direction=None
        self.route_mode=None

    def accept_pose(self,time_s):
        if not math.isfinite(time_s) or (self.last_pose_time is not None and time_s<=self.last_pose_time):
            raise ValueError("Fresh accepted pose times must increase.")
        self.last_pose_time=float(time_s)

    def command(self,estimated_pose,P,time_s,scan):
        pose=finite_vector(estimated_pose,3,"Estimated docking pose")
        P=covariance(P,3)
        params=self.parameters
        if not math.isfinite(time_s) or (scan is not None and scan.time_s>time_s+1e-7) or (self.last_pose_time is not None and self.last_pose_time>time_s+1e-7):
            raise ValueError("Invalid command/sensor time order.")
        delta=self.goal[:2]-pose[:2]
        rho=float(np.linalg.norm(delta))
        yaw_error=float(wrap_angle(self.goal[2]-pose[2]))
        twist=np.zeros(2)
        alpha,beta=None,None
        active_goal=self.goal.copy()
        if self.last_pose_time is None or time_s-self.last_pose_time>params["max_pose_age_s"]:
            state="VISUAL_POSE_STOP"
        elif scan is None or time_s-scan.time_s>params["max_scan_age_s"]:
            state="STALE_SCAN_STOP"
        elif np.sqrt(np.linalg.eigvalsh(P[:2,:2]).max())>params["max_position_std_m"] or math.sqrt(P[2,2])>params["max_yaw_std_rad"]:
            state="UNCERTAINTY_STOP"
        else:
            lateral=-math.sin(self.goal[2])*(pose[0]-self.goal[0])+math.cos(self.goal[2])*(pose[1]-self.goal[1])
            if self.route_mode is None:
                self.route_mode="RETREAT" if abs(lateral)>params["retreat_lateral_trigger_m"] or abs(yaw_error)>params["retreat_yaw_trigger_rad"] else "FINAL"
            if self.route_mode=="RETREAT" and np.linalg.norm(self.retreat_goal[:2]-pose[:2])<=params["retreat_position_tolerance_m"] and abs(yaw_error)<=params["retreat_yaw_tolerance_rad"]:
                self.route_mode="FINAL"
                self.direction=None
            active_goal=self.retreat_goal if self.route_mode=="RETREAT" else self.goal
            active_delta=active_goal[:2]-pose[:2]
            active_rho=float(np.linalg.norm(active_delta))
            tolerance=params["retreat_position_tolerance_m"] if self.route_mode=="RETREAT" else params["position_tolerance_m"]
            if active_rho<=tolerance:
                if self.route_mode=="RETREAT":
                    state="ALIGN_RETREAT"
                    twist[1]=np.clip(params["alignment_gain_per_s"]*yaw_error,-params["max_omega_rad_s"],params["max_omega_rad_s"])
                elif abs(yaw_error)>params["yaw_tolerance_rad"]:
                    state="ALIGN_DOCK"
                    twist[1]=np.clip(params["alignment_gain_per_s"]*yaw_error,-params["max_omega_rad_s"],params["max_omega_rad_s"])
                else:
                    state="DOCK_HOLD"
                    self.done_since=float(time_s) if self.done_since is None else self.done_since
                    if time_s-self.done_since>=params["hold_seconds"]:
                        state="DOCKED"
            else:
                raw=float(wrap_angle(math.atan2(active_delta[1],active_delta[0])-pose[2]))
                if self.direction is None:
                    self.direction=1 if math.cos(raw)>=0 else -1
                alpha=float(wrap_angle(raw-(math.pi if self.direction<0 else 0)))
                beta=float(wrap_angle(yaw_error-alpha))
                twist[1]=np.clip(params["k_alpha_per_s"]*alpha-params["k_beta_magnitude_per_s"]*beta,-params["max_omega_rad_s"],params["max_omega_rad_s"])
                if abs(alpha)>params["drive_heading_limit_rad"]:
                    state="ALIGN_APPROACH"
                else:
                    state="RETREAT_MOVE" if self.route_mode=="RETREAT" else "APPROACH_DOCK"
                    twist[0]=self.direction*min(params["max_v_m_s"],params["k_rho_per_s"]*active_rho)
        station_delta=self.station_xy-pose[:2]
        bearing=float(wrap_angle(math.atan2(station_delta[1],station_delta[0])-pose[2]))
        raw_omega=float(twist[1])
        if np.any(twist!=0):
            drift=twist[0]*math.sin(bearing)/max(float(np.linalg.norm(station_delta)),1e-6)
            lower=drift+(bearing-params["max_station_bearing_rad"])/params["bearing_horizon_s"]
            upper=drift+(bearing+params["max_station_bearing_rad"])/params["bearing_horizon_s"]
            if lower<=params["max_omega_rad_s"] and upper>=-params["max_omega_rad_s"]:
                twist[1]=np.clip(twist[1],max(-params["max_omega_rad_s"],lower),min(params["max_omega_rad_s"],upper))
            else:
                twist[:]=0.
                state="FOV_LIMIT_STOP"
        minimum=None
        if scan is not None:
            clear,minimum=scan_guard(scan,twist,self.mount,params)
            if not clear and np.any(twist!=0):
                twist[:]=0
                state="SCAN_GUARD_STOP"
        if state not in ("DOCK_HOLD","DOCKED"):
            self.done_since=None
        wheel=twist_to_wheel_rates(twist,self.radius,self.track)
        scale=min(1.,params["max_desired_wheel_rad_s"]/max(float(np.max(abs(wheel))),1e-15))
        return {"state":state,"desired_wheel_rad_s":(wheel*scale).tolist(),"body_twist":(twist*scale).tolist(),
                "rho_estimated_m":rho,"yaw_error_estimated_rad":yaw_error,"alpha_rad":alpha,"beta_rad":beta,
                "direction":self.direction,"last_accepted_pose_time_s":self.last_pose_time,"guard_min_point_distance_m":minimum,
                "route_mode":self.route_mode,"active_goal_world":active_goal.tolist(),"station_bearing_estimated_rad":bearing,"bearing_limited":bool(abs(raw_omega-twist[1])>1e-12),
                "stop":bool(np.all(twist==0)),"docked":state=="DOCKED"}
