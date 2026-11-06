"""连续LiDAR导航→图像位姿EKF→末端停靠；真值只用于传感器生成及评价。"""



# Permit both package imports and direct execution of this existing driver.
import sys as _project_sys
from pathlib import Path as _ProjectPath
if __package__ in (None, ""):
    _project_sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[3]))

from evaluation.common.release_support import project_python_sources


import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import uuid

import mujoco
import numpy as np
from PIL import Image

from evaluation.common.camera_scene import load_camera_scene
from evaluation.D_Control.code.docking import DockingController,pose_measurement,station_goal
from evaluation.C_Estimation.code.ekf import LocalizationEKF,initialize_pair_covariance,vision_covariance
from evaluation.C_Estimation.code.ekf_check import append_json,public_observation
from evaluation.B_Sensing.code.lidar import Lidar2D
from evaluation.B_Sensing.code.lidar_check import set_static_pose
from evaluation.common.main import ROOT,load_model,validate_state
from evaluation.E_Integration_Robustness.code.navigation import GridNavigator
from evaluation.E_Integration_Robustness.code.no_path_recovery import NoPathRecovery
from evaluation.B_Sensing.code.online_encoder import IncrementalEncoder
from evaluation.A_Modelling.code.state_model import wrap_angle
from evaluation.D_Control.code.tracking import WheelPI
from evaluation.B_Sensing.code.vision import ArucoVision
from evaluation.E_Integration_Robustness.code.vision_recovery import PairRecovery
from evaluation.B_Sensing.code.wheel_check import read_truth_for_evaluation,read_wheels,wheel_indices,write_json


def evaluation_geometry(model):
    robot={i for i in range(model.ngeom) if model.geom_group[i]==1 and (model.geom_contype[i] or model.geom_conaffinity[i])}
    obstacles={i for i in range(model.ngeom) if i not in robot and model.geom_type[i]!=mujoco.mjtGeom.mjGEOM_PLANE and (model.geom_contype[i] or model.geom_conaffinity[i])}
    return robot,obstacles


def geometry_clearance_for_evaluation(model,data,robot,obstacles):
    minimum=.5
    pair=None
    for first in robot:
        for second in obstacles:
            value=mujoco.mj_geomDistance(model,data,first,second,.5,None)
            if value<minimum:
                minimum=float(value)
                pair=[int(first),int(second)]
    return minimum,pair


def evaluate_case(case,config,rows,truths,commands,nav,contacts,output):
    goal=station_goal(config["station"])
    last=truths[-1]
    actual=np.array([[row[key] for key in ["x_m","y_m","yaw_rad"]] for row in truths])
    estimated=np.array([row["estimated_state"] for row in rows])
    error=estimated-actual
    error[:,2]=wrap_angle(error[:,2])
    tail=[row for row in truths if row["elapsed_s"]>=last["elapsed_s"]-.5-1e-8]
    metrics={"final_position_error_m":float(np.linalg.norm(actual[-1,:2]-goal[:2])),
             "final_yaw_error_degrees":float(np.degrees(wrap_angle(actual[-1,2]-goal[2]))),
             "position_localization_rmse_m":float(np.sqrt(np.mean(np.sum(error[:,:2]**2,axis=1)))),
             "actual_path_length_m":float(np.linalg.norm(np.diff(actual[:,:2],axis=0),axis=1).sum()),
             "elapsed_seconds":last["elapsed_s"],"min_signed_geom_clearance_m":min(row["signed_geom_clearance_m"] for row in truths),
             "stop_tail_max_true_speed_m_s":max(row["body_speed_m_s"] for row in tail),
             "stop_tail_max_true_yaw_rate_rad_s":max(abs(row["yaw_rate_rad_s"]) for row in tail),
             "max_abs_actuator_command_rad_s":max(abs(v) for row in commands for v in row["PI"]["actuator_command_rad_s"]),
             "obstacle_contact_physics_steps":contacts,"planning_events":len(nav.plan_events),
             "successful_plans":sum(event["path"] is not None for event in nav.plan_events),
             "max_true_y_m":float(actual[:,1].max()),"max_estimated_y_m":float(estimated[:,1].max())}
    docking_rows=[row for row in commands if row["task_phase"]=="DOCKING"]
    checks={"docked":commands[-1]["navigation"].get("docked",False),
            "navigation_handoff_occurred":bool(docking_rows),
            "final_position":metrics["final_position_error_m"]<=config["checks"]["max_final_position_error_m"],
            "final_heading":abs(metrics["final_yaw_error_degrees"])<=config["checks"]["max_final_yaw_error_degrees"],
            "no_obstacle_contacts":contacts==0,
            "positive_sampled_clearance":metrics["min_signed_geom_clearance_m"]>=config["checks"]["min_signed_geom_clearance_m"],
            "physically_stopped":metrics["stop_tail_max_true_speed_m_s"]<=config["checks"]["max_stop_true_speed_m_s"] and metrics["stop_tail_max_true_yaw_rate_rad_s"]<=config["checks"]["max_stop_true_yaw_rate_rad_s"],
            "actuator_limits":metrics["max_abs_actuator_command_rad_s"]<=7.88+1e-12}
    if case["obstacle_appears_at_s"] is not None:
        checks["replanned_after_appearance"]=any(event["time_s"]-rows[0]["time_s"]>=case["obstacle_appears_at_s"] and event["path"] is not None for event in nav.plan_events)
        checks["actual_detour"]=metrics["max_true_y_m"]>1.75
    if case["dock_blackout_interval_s"] is not None:
        frames=[json.loads(line) for line in (output/"vision_frames.jsonl").read_text().splitlines()]
        blackout=[row for row in frames if row.get("fixture_blackout")]
        checks["blackout_observed_in_rgb"]=bool(blackout) and all(not row["observations"] for row in blackout)
        checks["visual_timeout_stopped"]=any(row["navigation"]["state"]=="VISUAL_POSE_STOP" and row["navigation"]["stop"] and row["fixture_blackout"] for row in docking_rows)
        checks["fresh_pose_resumed"]=any(row.get("task_phase_at_processing")=="DOCKING" and not row.get("fixture_blackout") and row["elapsed_s"]>blackout[-1]["elapsed_s"] and row["update"]["status"]=="accepted" for row in frames) if blackout else False
    metrics["docking_handoff_elapsed_s"]=None if not docking_rows else docking_rows[0]["elapsed_s"]
    metrics["terminal_seconds"]=None if not docking_rows else last["elapsed_s"]-docking_rows[0]["elapsed_s"]
    result={"case":case["name"],"passed":all(checks.values()),"checks":checks,"metrics":metrics,
            "navigation_state_counts":{state:sum(row["navigation"]["state"]==state for row in commands) for state in sorted({row["navigation"]["state"] for row in commands})},
            "clearance_scope":"Signed physical geom separation sampled at lidar times; robot-obstacle contacts checked at every physics step, floor/self contacts excluded. Neither feeds navigation."}
    write_json(output/"evaluation.json",result)
    return result


def check_close_images(config,camera_config,room,output):
    output.mkdir(parents=True,exist_ok=False)
    model,data,provenance,_,_=load_camera_scene(room,camera_config)
    detector=ArucoVision(camera_config)
    results=[]
    with mujoco.Renderer(model,height=detector.height,width=detector.width) as renderer:
        for case in config["image_validation_cases"]:
            set_static_pose(model,data,case["base_xy_yaw"])
            renderer.update_scene(data,camera=camera_config["camera"]["name"])
            rgb=renderer.render().copy()
            Image.fromarray(rgb).save(output/f"{case['name']}.png")
            observations=detector.detect(rgb,0.)
            primary=next((item for item in observations if item.marker_id==config["station"]["marker_id"]),None)
            row={"case":case,"observations":[public_observation(item) for item in observations],"passed":False}
            if primary is not None:
                try:
                    pose,R,details=pose_measurement(primary,detector,config["station"],config["pose_measurement"])
                    error=pose-np.array(case["base_xy_yaw"])
                    error[2]=wrap_angle(error[2])
                    row.update(image_pose=pose.tolist(),R=R.tolist(),details=details,position_error_m=float(np.linalg.norm(error[:2])),yaw_error_degrees=float(np.degrees(error[2])))
                    row["passed"]=row["position_error_m"]<=config["checks"]["static_image_position_m"] and abs(row["yaw_error_degrees"])<=config["checks"]["static_image_yaw_degrees"]
                except ValueError as error:
                    row["error"]=str(error)
            results.append(row)
    write_json(output/"results.json",{"cases":results,"passed":all(row["passed"] for row in results),"scope":"Static perception fixtures only; not a docking trajectory or controller truth input."})
    if not all(row["passed"] for row in results):
        raise RuntimeError("Actual close-up image/pose precheck failed; no truth fallback.")
    return results


def run_case(case,config,ekf_config,tracking_config,camera_config,room,known_map,output):
    output.mkdir(parents=True,exist_ok=False)
    model,data,provenance,robot_xml,scene_xml=load_camera_scene(room,camera_config)
    write_json(output/"scene_provenance.json",provenance)
    (output/"derived_robot.xml").write_bytes(robot_xml)
    (output/"derived_scene.xml").write_bytes(scene_xml)
    original,_,_,_=load_model()
    for name in ["base","wheel_left","wheel_right"]:
        np.testing.assert_array_equal(model.body(name).mass,original.body(name).mass)
        np.testing.assert_array_equal(model.body(name).inertia,original.body(name).inertia)
    set_static_pose(model,data,config["initial_true_pose_m_m_rad"])
    dt=float(model.opt.timestep)
    settle_steps=round(ekf_config["settle_seconds"]/dt)
    for _ in range(settle_steps):
        mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    validate_state(model,data)
    start=float(data.time)
    strides={"encoder":round(1/(ekf_config["encoder_rate_hz"]*dt)),"camera":round(1/(ekf_config["camera_rate_hz"]*dt)),
             "lidar":round(1/(config["lidar_rate_hz"]*dt)),"control":round(1/(config["control_rate_hz"]*dt))}
    assert all(value%strides["encoder"]==0 for value in strides.values())
    sensor=IncrementalEncoder(config["encoder_seed"],ekf_config["encoder_noise"]["motion_variance_rad2_per_rad"],ekf_config["encoder_noise"]["idle_variance_rad2_per_s"])
    indices=wheel_indices(model)
    detector=ArucoVision(camera_config)
    scanner=Lidar2D(model,room["site_name"],**room["lidar"],robot_geom_group=room["robot_geom_group"])
    nav=GridNavigator(config["navigator"],config["grid"],ekf_config["wheel_radius_m"],ekf_config["track_width_m"],config.get("frontier_recovery"))
    no_path=NoPathRecovery(config["no_path_recovery"]) if "no_path_recovery" in config else None
    dock=DockingController(config["controller"],config["station"],ekf_config["wheel_radius_m"],ekf_config["track_width_m"],config["grid"]["lidar_mount_xy_m"])
    pi=WheelPI(tracking_config["wheel_PI"])
    task_phase="NAVIGATION"
    docking_start=None
    latest_scan=None
    fixture_blackout=False
    transitions=[{"elapsed_s":0.,"phase":"NAVIGATION"}]
    robot,obstacles=evaluation_geometry(model)
    mocap_id=int(model.body_mocapid[model.body("vision_occluder").id])
    rgb_dir=output/"rgb"
    rgb_dir.mkdir()
    (output/"maps").mkdir()
    write_json(output/"lidar_calibration.json",{"angles_rad":scanner.angles_rad.tolist(),"range_min_m":scanner.range_min_m,"range_max_m":scanner.range_max_m,
                                                "mount_xy_m":config["grid"]["lidar_mount_xy_m"],"frame_id":scanner.frame_id})
    write_json(output/"camera_calibration.json",{"K":detector.K.tolist(),"distortion":detector.distortion.tolist()})
    rows,truths,commands,frames=[],[],[],[]
    contact_steps=0
    contact_pairs=set()
    steps=round(config["max_seconds_per_case"]/dt)
    actual_steps=0
    scan_count=0
    with (output/"measurements.jsonl").open("x",encoding="utf-8") as measure_stream, \
         (output/"encoder_sensor_audit.jsonl").open("x",encoding="utf-8") as audit_stream, \
         (output/"estimates.jsonl").open("x",encoding="utf-8") as estimate_stream, \
         (output/"controls.jsonl").open("x",encoding="utf-8") as command_stream, \
         (output/"scans.jsonl").open("x",encoding="utf-8") as scan_stream, \
         (output/"plans.jsonl").open("x",encoding="utf-8") as plan_stream, \
         (output/"vision_frames.jsonl").open("x",encoding="utf-8") as frame_stream, \
         (output/"ground_truth_evaluation.csv").open("x",encoding="utf-8",newline="") as truth_stream, \
         mujoco.Renderer(model,height=detector.height,width=detector.width) as renderer:
        wheels=read_wheels(data,indices)
        measured,audit=sensor.sample([wheels["left_angle_rad"],wheels["right_angle_rad"]],start)
        renderer.update_scene(data,camera=camera_config["camera"]["name"])
        rgb=renderer.render().copy()
        Image.fromarray(rgb).save(rgb_dir/"frame0000.png")
        public=[public_observation(item) for item in sorted(detector.detect(rgb,start),key=lambda item:item.marker_id)]
        if [item["marker_id"] for item in public]!=[17,18]:
            raise RuntimeError("Initial image requires both known markers; no truth fallback.")
        state,P,details=initialize_pair_covariance(known_map,public,vision_covariance(2,ekf_config["vision_R"]),ekf_config["initialization_floor_std_m_m_rad"])
        estimator=LocalizationEKF(state,P,known_map,ekf_config["wheel_radius_m"],ekf_config["track_width_m"],measured,start,ekf_config["process"],initial_frame_used=True)
        recovery=PairRecovery(config["pair_recovery"],start)
        write_json(output/"initialization.json",{"time_s":start,"state":state.tolist(),"P":P.tolist(),"details":details,"image_used_once":True})
        frame={"frame_id":"frame0000","time_s":start,"elapsed_s":0.,"observations":public,"update":{"status":"initialization_only"}}
        frames.append(frame)
        append_json(frame_stream,frame)
        previous=measured.copy()
        truth_writer=None
        clearance,pair=geometry_clearance_for_evaluation(model,data,robot,obstacles)
        navigation_command=None

        def scan_update(elapsed):
            nonlocal scan_count,clearance,pair,latest_scan
            scan=scanner.scan(data)
            latest_scan=scan
            statistics,event=nav.update_scan(scan,estimator.state) if task_phase=="NAVIGATION" else (None,None)
            append_json(scan_stream,{"time_s":float(data.time),"elapsed_s":elapsed,"task_phase":task_phase,"mapping_epoch":nav.mapping_epoch,"ranges_m":scan.ranges_m.tolist(),"status":scan.status.tolist(),
                                     "estimated_pose_input":estimator.state.tolist(),"map_statistics":statistics})
            scan_count+=1
            if event is not None:
                append_json(plan_stream,event)
                np.savez_compressed(output/"maps"/f"plan{len(nav.plan_events):04d}.npz",cells=nav.grid.cells,blocked=nav.grid.blocked(),
                                    path=np.empty((0,2)) if nav.path is None else nav.path,estimated_pose=estimator.state)
            # 这里读取几何分离距离只供下面的独立真值CSV，绝不传回nav。
            clearance,pair=geometry_clearance_for_evaluation(model,data,robot,obstacles)

        scan_update(0.)
        navigation_command=nav.command(estimator.state,estimator.P,start)
        if no_path is not None:
            navigation_command=no_path.supervise(navigation_command,start)
        inner=pi.update(navigation_command["desired_wheel_rad_s"],[0,0],1/ekf_config["encoder_rate_hz"],stop=navigation_command["stop"])

        def record(elapsed,prediction,vision_status,nav_updated):
            nonlocal truth_writer
            estimate={"time_s":float(data.time),"elapsed_s":elapsed,"estimated_state":estimator.state.tolist(),"P":estimator.P.tolist(),"prediction":prediction,"vision_status":vision_status}
            rows.append(estimate)
            append_json(estimate_stream,estimate)
            append_json(measure_stream,{"time_s":float(data.time),"elapsed_s":elapsed,"measured_wheel_angles_rad":measured.tolist()})
            append_json(audit_stream,dict(time_s=float(data.time),**audit))
            command={"time_s":float(data.time),"elapsed_s":elapsed,"task_phase":task_phase,"fixture_blackout":fixture_blackout,"estimated_state_available":estimator.state.tolist(),"navigation_updated":nav_updated,"navigation":navigation_command,"PI":inner}
            commands.append(command)
            append_json(command_stream,command)
            truth=dict(time_s=float(data.time),elapsed_s=elapsed,**read_truth_for_evaluation(model,data),signed_geom_clearance_m=clearance,
                       closest_geom_1=-1 if pair is None else pair[0],closest_geom_2=-1 if pair is None else pair[1])
            if truth_writer is None:
                truth_writer=csv.DictWriter(truth_stream,fieldnames=list(truth))
                truth_writer.writeheader()
            truth_writer.writerow(truth)
            truth_stream.flush()
            truths.append(truth)

        record(0.,None,"initialization_only",True)
        for step in range(1,steps+1):
            elapsed=step*dt
            appears=case["obstacle_appears_at_s"]
            data.mocap_pos[mocap_id]=case["obstacle_position_world_m"] if appears is not None and (step-1)*dt>=appears-1e-9 else [0,0,-10]
            interval=case["dock_blackout_interval_s"]
            fixture_blackout=interval is not None and docking_start is not None and interval[0]<=elapsed-docking_start<interval[1]
            if fixture_blackout:
                data.mocap_pos[mocap_id]=case["blackout_board_world_m"]
            for index,name in enumerate(["wheel_left","wheel_right"]):
                data.ctrl[indices[name]["actuator"]]=inner["actuator_command_rad_s"][index]
            mujoco.mj_step(model,data)
            actual_steps=step
            pairs=[(int(contact.geom1),int(contact.geom2)) for contact in data.contact]
            bad=[(a,b) for a,b in pairs if (a in robot and b in obstacles) or (b in robot and a in obstacles)]
            if bad:
                contact_steps+=1
                contact_pairs.update(bad)
            if step%strides["encoder"]:
                continue
            mujoco.mj_forward(model,data)
            validate_state(model,data)
            wheels=read_wheels(data,indices)
            measured,audit=sensor.sample([wheels["left_angle_rad"],wheels["right_angle_rad"]],float(data.time))
            prediction=estimator.predict_encoders(measured,float(data.time))
            status="none"
            if step%strides["camera"]==0:
                frame_id=f"frame{step//strides['camera']:04d}"
                renderer.update_scene(data,camera=camera_config["camera"]["name"])
                rgb=renderer.render().copy()
                Image.fromarray(rgb).save(rgb_dir/f"{frame_id}.png")
                detected=sorted(detector.detect(rgb,float(data.time)),key=lambda item:item.marker_id)
                public=[public_observation(item) for item in detected]
                prior_state,prior_P=estimator.state.copy(),estimator.P.copy()
                image_pose,image_R,pose_details=None,None,None
                recovery_diagnostic=None
                oriented_branch=task_phase=="DOCKING" or (no_path is not None and (no_path.mode=="COLLECT" or no_path.full_pose_navigation))
                no_path_diagnostic=None
                if not oriented_branch:
                    R=vision_covariance(len(public),ekf_config["vision_R"]) if public else np.empty((0,0))
                    update=estimator.update_vision(public,float(data.time),frame_id,R,ekf_config["nis_limits"].get(str(2*len(public)),1))
                    candidate,recovery_diagnostic=recovery.consider(public,float(data.time),update["status"],known_map,R,ekf_config["initialization_floor_std_m_m_rad"])
                    if candidate is not None:
                        estimator=LocalizationEKF(candidate["state"],candidate["P"],known_map,ekf_config["wheel_radius_m"],ekf_config["track_width_m"],measured,float(data.time),ekf_config["process"],initial_frame_used=True)
                        nav.reset_mapping()
                        update={"status":"reinitialized_from_consistent_pair","original_rejected_update":update}
                else:
                    primary=next((item for item in detected if item.marker_id==config["station"]["marker_id"]),None)
                    if primary is not None:
                        try:
                            image_pose,image_R,pose_details=pose_measurement(primary,detector,config["station"],config["pose_measurement"])
                        except ValueError as error:
                            pose_details={"invalid_image_pose":str(error)}
                    if task_phase=="NAVIGATION" and no_path.mode=="COLLECT":
                        # These frames are consistency checks, not repeated independent EKF information.
                        update=estimator.update_pose(None,float(data.time),frame_id,None,config["pose_measurement"]["nis_limit_3d"])
                        candidate,no_path_diagnostic=no_path.consider(image_pose,image_R,estimator.state,(measured-previous)/(strides["encoder"]*dt),float(data.time),None if latest_scan is None else latest_scan.time_s)
                        if candidate is not None:
                            estimator=LocalizationEKF(candidate["state"],candidate["P"],known_map,ekf_config["wheel_radius_m"],ekf_config["track_width_m"],measured,float(data.time),ekf_config["process"],initial_frame_used=True)
                            estimator.last_frame_id=frame_id
                            nav.reset_mapping()
                            nav.frontier_enabled=True
                            update={"status":"reinitialized_from_oriented_marker","measurement_kind":"image_planar_pose","confirmation_frame_used_once":True}
                    else:
                        update=estimator.update_pose(image_pose,float(data.time),frame_id,image_R,config["pose_measurement"]["nis_limit_3d"])
                    if task_phase=="DOCKING" and update["status"]=="accepted":
                        dock.accept_pose(float(data.time))
                status=update["status"]
                frame={"frame_id":frame_id,"time_s":float(data.time),"elapsed_s":elapsed,"observations":public,"prior_state":prior_state.tolist(),"prior_P":prior_P.tolist(),
                       "posterior_state":estimator.state.tolist(),"posterior_P":estimator.P.tolist(),"update":update,"pair_recovery":recovery_diagnostic,
                       "measurement_branch":"oriented_pose" if oriented_branch else "range_bearing","no_path_recovery":no_path_diagnostic,
                       "task_phase_at_processing":task_phase,"fixture_blackout":fixture_blackout,"image_pose":None if image_pose is None else image_pose.tolist(),"image_pose_R":None if image_R is None else image_R.tolist(),"pose_details":pose_details}
                frames.append(frame)
                append_json(frame_stream,frame)
            if step%strides["lidar"]==0:
                scan_update(elapsed)
            nav_updated=step%strides["control"]==0
            if nav_updated:
                if task_phase=="NAVIGATION":
                    navigation_command=nav.command(estimator.state,estimator.P,float(data.time))
                    if no_path is not None:
                        navigation_command=no_path.supervise(navigation_command,float(data.time))
                    if navigation_command["arrived"]:
                        task_phase="DOCKING"
                        docking_start=elapsed
                        transitions.append({"elapsed_s":elapsed,"phase":"DOCKING","estimated_pose":estimator.state.tolist(),"reason":"navigator_arrived; current RGB already consumed, wait for next accepted full pose"})
                        navigation_command=dock.command(estimator.state,estimator.P,float(data.time),latest_scan)
                else:
                    navigation_command=dock.command(estimator.state,estimator.P,float(data.time),latest_scan)
            rate=(measured-previous)/(strides["encoder"]*dt)
            inner=pi.update(navigation_command["desired_wheel_rad_s"],rate,strides["encoder"]*dt,stop=navigation_command["stop"])
            previous=measured.copy()
            record(elapsed,prediction,status,nav_updated)
            if navigation_command.get("docked",False):
                break
    np.savez_compressed(output/"final_map.npz",cells=nav.grid.cells,blocked=nav.grid.blocked(),origin_xy_m=nav.grid.origin,resolution_m=nav.grid.resolution)
    result=evaluate_case(case,config,rows,truths,commands,nav,contact_steps,output)
    summary={"case":case,"evaluation":result,"physics_steps":actual_steps+settle_steps,"motion_steps":actual_steps,"settle_steps":settle_steps,
             "encoder_samples":len(rows),"camera_frames":len(frames),"lidar_scans":scan_count,"obstacle_contact_pairs":sorted(contact_pairs),
             "transitions":transitions,"vision_status_counts":{state:sum(frame["update"]["status"]==state for frame in frames) for state in ["initialization_only","accepted","rejected_nis","no_observation","reinitialized_from_consistent_pair"]},
             "no_path_recovery_events":[] if no_path is None else no_path.events,
             "warning_counts":validate_state(model,data),"robot_mass_and_inertia_unchanged":True}
    write_json(output/"summary.json",summary)
    return summary


def make_plot(summaries,output,config):
    os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".tmp/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(summaries),figsize=(6*len(summaries),5),layout="constrained",squeeze=False)
    for axis,summary in zip(axes[0],summaries):
        folder=output/summary["case"]["name"]
        with (folder/"ground_truth_evaluation.csv").open(encoding="utf-8",newline="") as stream:
            actual=list(csv.DictReader(stream))
        estimates=[json.loads(line) for line in (folder/"estimates.jsonl").read_text().splitlines()]
        grid=np.load(folder/"final_map.npz")
        cells=grid["cells"]
        image=np.zeros((*cells.shape,3))
        image[cells==0]=[.65,.65,.65]
        image[cells==1]=[1,1,1]
        image[cells==2]=[.1,.1,.1]
        bounds=config["grid"]
        extent=[bounds["origin_xy_m"][0],bounds["origin_xy_m"][0]+bounds["size_xy_m"][0],bounds["origin_xy_m"][1],bounds["origin_xy_m"][1]+bounds["size_xy_m"][1]]
        axis.imshow(image,origin="lower",extent=extent)
        axis.contour(grid["blocked"],levels=[.5],colors=["#dfad65"],linewidths=.6,origin="lower",extent=extent)
        axis.plot([float(row["x_m"]) for row in actual],[float(row["y_m"]) for row in actual],label="Truth (evaluation)",color="#138ca0")
        axis.plot([row["estimated_state"][0] for row in estimates],[row["estimated_state"][1] for row in estimates],label="EKF",color="#e2802c",linestyle="--")
        events=[json.loads(line) for line in (folder/"plans.jsonl").read_text().splitlines()]
        paths=[np.array(event["path"]) for event in events if event["path"] is not None]
        if paths:
            path=max(paths,key=lambda value:len(value))
            axis.plot(path[:,0],path[:,1],"b:",label="One planned route")
        goal=station_goal(config["station"])
        axis.scatter([goal[0]],[goal[1]],marker="*",color="red",s=90,label="Docking pose goal")
        axis.set(xlim=(.3,2.7),ylim=(.6,2.35),xlabel="World x (m)",ylabel="World y (m)",title=f"{summary['case']['name']}: {summary['evaluation']['metrics']['elapsed_seconds']:.1f} s")
        axis.legend(fontsize=7)
    fig.savefig(output/"task_paths.png",dpi=150)
    plt.close(fig)


def main():
    docking_config=json.loads((ROOT/"configs/docking_check.json").read_text(encoding="utf-8"))
    navigation_config=json.loads((ROOT/docking_config["navigation_config"]).read_text(encoding="utf-8"))
    config=dict(navigation_config,**docking_config)
    ekf_config=json.loads((ROOT/config["ekf_config"]).read_text(encoding="utf-8"))
    tracking_config=json.loads((ROOT/config["tracking_config"]).read_text(encoding="utf-8"))
    camera_config=json.loads((ROOT/ekf_config["camera_config"]).read_text(encoding="utf-8"))
    camera_config["camera"]["position_in_base_m"]=config["camera_mount_override_m"]
    model_config=json.loads((ROOT/ekf_config["model_config"]).read_text(encoding="utf-8"))
    camera_config["additional_markers"]=model_config["additional_markers"]
    room=json.loads((ROOT/camera_config["lidar_config"]).read_text(encoding="utf-8"))
    known_map={item["id"]:item["position_world_m"][:2] for item in [camera_config["marker"],*camera_config["additional_markers"]]}
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"-"+uuid.uuid4().hex[:8]
    output=ROOT/"results/stage5_docking"/run_id
    output.mkdir(parents=True,exist_ok=False)
    git=shutil.which("git") or str(Path.home()/".cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe")
    revision=subprocess.run([git,"-C",str(ROOT),"rev-parse","--verify","HEAD"],capture_output=True,text=True)
    metadata={"run_id":run_id,"status":"started","created_at_utc":datetime.now(timezone.utc).isoformat(),"command":subprocess.list2cmdline([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]]),
              "python":sys.version,"platform":platform.platform(),"git_commit":revision.stdout.strip() if revision.returncode==0 else None,"encoder_seed":config["encoder_seed"],
              "packages":{name:importlib.metadata.version(name) for name in ["mujoco","numpy","matplotlib","pillow","opencv-python-headless"]},"scope":config["scope"]}
    sources=[*project_python_sources(ROOT),*ROOT.glob("tests/*.py"),*ROOT.glob("configs/*.json"),ROOT/"scenes/room.xml",ROOT/"requirements.txt",ROOT/"resources.lock.json",ROOT/"launch_docking_check.cmd"]
    for source in sources:
        target=output/"source"/source.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for name,value in [("config.json",config),("ekf_config.json",ekf_config),("tracking_config.json",tracking_config),("camera_config.json",camera_config),("known_map.json",known_map),("run.json",metadata)]:
        write_json(output/name,value)
    started,exit_code=time.perf_counter(),1
    try:
        tests=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],cwd=ROOT,capture_output=True,text=True)
        (output/"tests.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
        if tests.returncode:
            raise RuntimeError("Module tests failed.")
        dependency=subprocess.run([sys.executable,"-m","pip","check"],capture_output=True,text=True)
        (output/"dependency_check.txt").write_text(dependency.stdout+dependency.stderr,encoding="utf-8")
        if dependency.returncode:
            raise RuntimeError("Dependency check failed.")
        if not np.allclose(camera_config["marker"]["position_world_m"],config["station"]["origin_world_m"]) or camera_config["marker"]["id"]!=config["station"]["marker_id"]:
            raise ValueError("Declared station prior must match the configured fixed marker map.")
        check_close_images(config,camera_config,room,output/"close_images")
        summaries=[]
        for case in config["cases"]:
            result=run_case(case,config,ekf_config,tracking_config,camera_config,room,known_map,output/case["name"])
            summaries.append(result)
            write_json(output/"partial_results.json",summaries)
            print(f"{case['name']}: {result['evaluation']['passed']}, t={result['evaluation']['metrics']['elapsed_seconds']:.1f}s, final={result['evaluation']['metrics']['final_position_error_m']:.3f}m",flush=True)
        make_plot(summaries,output,config)
        passed=all(result["evaluation"]["passed"] for result in summaries)
        summary={"status":"completed" if passed else "failed","cases":summaries,"module_tests_passed":True,
                 "truth_boundary":"Navigation and docking control receive estimates/P and local scans; full image pose uses RGB PnP plus declared oriented marker prior. True poses/obstacles only generate sensors or evaluate. No robot/encoder reset at handoff, no double fusion of one RGB frame.",
                 "scope":config["scope"]}
        write_json(output/"summary.json",summary)
        metadata["status"],exit_code=summary["status"],0 if passed else 1
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as error:
        metadata.update(status="error",error=f"{type(error).__name__}: {error}")
        (output/"error.txt").write_text(traceback.format_exc(),encoding="utf-8")
        print(metadata["error"],file=sys.stderr,flush=True)
    finally:
        metadata["completed_at_utc"],metadata["wall_elapsed_seconds"]=datetime.now(timezone.utc).isoformat(),time.perf_counter()-started
        write_json(output/"run.json",metadata)
        write_json(output/"manifest.json",{"files_sha256":{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file()}})
    print(f"DOCKING_{metadata['status'].upper()}: {output.relative_to(ROOT).as_posix()}",flush=True)
    return exit_code


if __name__=="__main__":
    raise SystemExit(main())
