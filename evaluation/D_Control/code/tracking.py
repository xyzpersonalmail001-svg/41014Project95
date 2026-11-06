"""平面参考轨迹、估计位姿反馈、轮速PI；均不访问MuJoCo或真实位姿。"""

import math
import numpy as np

from evaluation.A_Modelling.code.state_model import finite_vector, geometry, twist_to_wheel_rates, wrap_angle


def integrate_reference(pose, v0, acceleration, omega, duration):
    """每段角速度常量、线速度线性变化的解析积分，单位m/s、m/s²、rad/s、s。"""
    x, y, yaw = finite_vector(pose, 3, "Reference pose")
    if not all(math.isfinite(v) for v in [v0, acceleration, omega, duration]) or duration < 0:
        raise ValueError("Reference motion must be finite, with nonnegative duration.")
    angle = yaw + omega*duration
    if abs(omega) < 1e-8:
        distance = v0*duration + acceleration*duration**2/2
        dx, dy = distance*math.cos(yaw), distance*math.sin(yaw)
    else:
        dx = v0*(math.sin(angle)-math.sin(yaw))/omega + acceleration*(duration*math.sin(angle)/omega + (math.cos(angle)-math.cos(yaw))/omega**2)
        dy = v0*(math.cos(yaw)-math.cos(angle))/omega + acceleration*(-duration*math.cos(angle)/omega + (math.sin(angle)-math.sin(yaw))/omega**2)
    return np.array([x+dx, y+dy, angle])


class PlanarReference:
    def __init__(self, initial_pose, segments):
        self.initial = finite_vector(initial_pose, 3, "Reference initial pose").copy()
        self.segments, self.starts, self.poses = [], [], []
        pose, time = self.initial.copy(), 0.
        for segment in segments:
            values = [segment[key] for key in ["seconds", "v_start_m_s", "v_end_m_s", "omega_rad_s"]]
            if not all(math.isfinite(v) for v in values) or values[0] <= 0:
                raise ValueError("Reference segments must have finite inputs and positive duration.")
            self.segments.append(dict(segment))
            self.starts.append(time)
            self.poses.append(pose.copy())
            acceleration = (values[2]-values[1])/values[0]
            pose = integrate_reference(pose, values[1], acceleration, values[3], values[0])
            time += values[0]
        if not self.segments:
            raise ValueError("Need at least one reference segment.")
        self.duration_s, self.final_pose = time, pose

    def at(self, time_s):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Reference time must be finite and nonnegative.")
        for segment, start, pose in zip(self.segments, self.starts, self.poses):
            if time_s < start+segment["seconds"]-1e-10:
                elapsed = max(0.,time_s-start)
                acceleration = (segment["v_end_m_s"]-segment["v_start_m_s"])/segment["seconds"]
                return {"state": integrate_reference(pose,segment["v_start_m_s"],acceleration,segment["omega_rad_s"],elapsed),
                        "v_m_s": segment["v_start_m_s"]+acceleration*elapsed,
                        "omega_rad_s": segment["omega_rad_s"], "phase":segment["name"], "stop":segment.get("stop",False)}
        return {"state":self.final_pose.copy(),"v_m_s":0.,"omega_rad_s":0.,"phase":"finished","stop":True}


def body_error(estimated_state, reference_state):
    state = finite_vector(estimated_state,3,"Estimated state")
    reference = finite_vector(reference_state,3,"Reference state")
    dx, dy = reference[:2]-state[:2]
    c,s = math.cos(state[2]),math.sin(state[2])
    return np.array([c*dx+s*dy,-s*dx+c*dy,float(wrap_angle(reference[2]-state[2]))])


def tracking_command(estimated_state, reference, gains, limits, radius_m, track_m, *, feedback=True):
    """前向参考的非线性状态反馈；停车段强制零，未实现最终静态姿态调节。"""
    geometry(radius_m,track_m)
    error = body_error(estimated_state,reference["state"])
    kx,ky,ktheta = finite_vector([gains[key] for key in ["kx_per_s","ky_per_m2","ktheta_per_s"]],3,"Tracking gains")
    max_v,max_omega,max_wheel = finite_vector([limits[key] for key in ["body_v_m_s","body_omega_rad_s","desired_wheel_rad_s"]],3,"Limits")
    if min(kx,ky,ktheta,max_v,max_omega,max_wheel) <= 0:
        raise ValueError("Gains and limits must be positive.")
    v_ref,omega_ref = finite_vector([reference["v_m_s"],reference["omega_rad_s"]],2,"Reference twist")
    if v_ref < 0:
        raise ValueError("This tracking design is restricted to nonnegative reference speed.")
    if reference["stop"]:
        v,omega = 0.,0.
    elif feedback:
        v = v_ref*math.cos(error[2])+kx*error[0]
        omega = omega_ref+ky*v_ref*error[1]+ktheta*math.sin(error[2])
    else:
        v,omega = v_ref,omega_ref
    raw = np.array([v,omega])
    bounded = np.clip(raw,[-max_v,-max_omega],[max_v,max_omega])
    wheels = twist_to_wheel_rates(bounded,radius_m,track_m)
    scale = min(1.,max_wheel/max(float(np.max(abs(wheels))),1e-15))
    wheels *= scale
    return {"error_body_m_m_rad":error.tolist(),"raw_body_twist":raw.tolist(),
            "limited_body_twist":(bounded*scale).tolist(),"desired_wheel_rad_s":wheels.tolist(),
            "body_limited":bool(np.any(raw!=bounded)),"wheel_scale":scale,
            "stop":bool(reference["stop"]),"feedback_enabled":feedback}


class WheelPI:
    def __init__(self,parameters):
        self.kp = float(parameters["kp"])
        self.ki = float(parameters["ki_per_s"])
        self.tau = float(parameters["rate_filter_tau_s"])
        self.limit = float(parameters["actuator_limit_rad_s"])
        if not all(math.isfinite(v) and v > 0 for v in [self.kp,self.ki,self.tau,self.limit]):
            raise ValueError("PI parameters must be finite and positive.")
        self.integral,self.filtered_rate = np.zeros(2),np.zeros(2)

    def update(self,desired_rad_s,encoder_rate_rad_s,dt_s,*,stop=False,enabled=True):
        desired = finite_vector(desired_rad_s,2,"Desired wheel rate")
        rate = finite_vector(encoder_rate_rad_s,2,"Measured encoder rate")
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("Wheel PI dt must be positive.")
        alpha = 1-math.exp(-dt_s/self.tau)
        self.filtered_rate += alpha*(rate-self.filtered_rate)
        error = desired-self.filtered_rate
        frozen = np.zeros(2,dtype=bool)
        if stop:
            self.integral[:] = 0.
            raw,command = np.zeros(2),np.zeros(2)
        elif enabled:
            candidate = self.integral+self.ki*error*dt_s
            proposed = desired+self.kp*error+candidate
            frozen = ((proposed>self.limit)&(error>0))|((proposed<-self.limit)&(error<0))
            self.integral = np.where(frozen,self.integral,candidate)
            raw = desired+self.kp*error+self.integral
            command = np.clip(raw,-self.limit,self.limit)
        else:
            self.integral[:] = 0.
            raw,command = desired.copy(),np.clip(desired,-self.limit,self.limit)
        return {"filtered_rate_rad_s":self.filtered_rate.tolist(),"wheel_rate_error_rad_s":error.tolist(),
                "integral_output_rad_s":self.integral.tolist(),"unsaturated_actuator_rad_s":raw.tolist(),
                "actuator_command_rad_s":command.tolist(),"saturated":(abs(raw)>self.limit).tolist(),
                "integration_frozen":frozen.tolist(),"PI_enabled":enabled,"stop":stop}
