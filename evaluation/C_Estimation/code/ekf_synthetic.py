"""匹配模型下的独立随机估计试验；与真实渲染物理轨迹分开计数。"""

import numpy as np

from evaluation.C_Estimation.code.ekf import LocalizationEKF, vision_covariance
from evaluation.A_Modelling.code.state_model import landmark_model, twist_to_wheel_rates, wheel_step, wrap_angle
from evaluation.B_Sensing.code.wheel_check import write_csv, write_json


def run_synthetic_study(config, known_map, output):
    design = config["synthetic_study"]
    radius, track = config["wheel_radius_m"], config["track_width_m"]
    dt = design["dt_s"]
    increment = twist_to_wheel_rates(design["twist_m_s_rad_s"], radius, track) * dt
    process = {"motion_variance_rad2_per_rad": 0., "idle_variance_rad2_per_s": design["encoder_idle_variance_rad2_per_s"],
               "model_std_m_m_rad_per_sqrt_s": [0, 0, 0]}
    R = vision_covariance(2, {"independent_std_m_rad": design["vision_independent_std_m_rad"],
                               "common_std_m_rad": design["vision_common_std_m_rad"]})
    P0 = np.diag(np.asarray(design["initial_std_m_m_rad"])**2)
    trials = []
    for trial in range(design["trials"]):
        seed = design["seed_start"] + trial
        rng = np.random.Generator(np.random.PCG64(seed))
        truth = np.array(config["start_pose_m_m_rad"], dtype=float)
        initial = truth + rng.multivariate_normal(np.zeros(3), P0)
        estimator = LocalizationEKF(initial, P0, known_map, radius, track, [0, 0], 0., process)
        angles, nis = np.zeros(2), []
        for k in range(1, design["steps"] + 1):
            truth, _, _ = wheel_step(truth, increment, radius, track)
            angles += increment + rng.normal(0, np.sqrt(design["encoder_idle_variance_rad2_per_s"]*dt), 2)
            estimator.predict_encoders(angles, k*dt)
            values = np.concatenate([landmark_model(truth, known_map[i])[0] for i in [17, 18]])
            values += rng.multivariate_normal(np.zeros(4), R)
            observations = [{"marker_id": i, "range_m": float(values[2*j]), "bearing_rad": float(wrap_angle(values[2*j+1])),
                             "time_s": k*dt} for j, i in enumerate([17, 18])]
            diagnostic = estimator.update_vision(observations, k*dt, f"synthetic-{k}", R, 1e12)
            nis.append(diagnostic["nis"])
        error = estimator.state - truth
        error[2] = wrap_angle(error[2])
        nees = float(error @ np.linalg.solve(estimator.P, error))
        trials.append({"seed": seed, "final_nees": nees, "covered_by_95pct_3d_ellipsoid": nees <= design["nees_95_limit_3d"],
                       "mean_nis_4d": float(np.mean(nis)), "error_x_m": error[0], "error_y_m": error[1], "error_yaw_rad": error[2],
                       **{f"P_{i}{j}": estimator.P[i, j] for i in range(3) for j in range(3)}})
    write_csv(output / "synthetic_trials.csv", trials)
    coverage = float(np.mean([trial["covered_by_95pct_3d_ellipsoid"] for trial in trials]))
    mean_nees = float(np.mean([trial["final_nees"] for trial in trials]))
    result = {"trials": design["trials"], "seed_start": design["seed_start"], "seed_end": design["seed_start"]+design["trials"]-1,
              "final_95pct_3d_coverage": coverage, "mean_final_nees_3d": mean_nees,
              "mean_nis_4d": float(np.mean([trial["mean_nis_4d"] for trial in trials])),
              "passed": design["final_coverage_bounds"][0] <= coverage <= design["final_coverage_bounds"][1]
                        and design["mean_final_nees_bounds"][0] <= mean_nees <= design["mean_final_nees_bounds"][1],
              "scope": f"Ideal matched kinematics and declared Gaussian noises; {design['trials']} independent synthetic estimation trials, not physical runs or calibration of actual image R. NIS gate disabled by a large threshold for this Gaussian consistency check."}
    write_json(output / "synthetic_summary.json", result)
    return result
