"""扫描+估计位姿建图、保守网格A*、航点反馈和局部扫描停车；不访问仿真器。"""

import heapq
import math
import numpy as np

from evaluation.A_Modelling.code.state_model import finite_vector,twist_to_wheel_rates,wrap_angle

UNKNOWN,FREE,OCCUPIED=0,1,2


def grid_line(first,last):
    """包含两端的Bresenham整数网格线，输入/输出均(row,col)。"""
    y0,x0=map(int,first)
    y1,x1=map(int,last)
    dx,dy=abs(x1-x0),-abs(y1-y0)
    sx,sy=1 if x0<x1 else -1,1 if y0<y1 else -1
    error=dx+dy
    result=[]
    while True:
        result.append((y0,x0))
        if (x0,y0)==(x1,y1):
            return result
        doubled=2*error
        if doubled>=dy:
            error+=dy
            x0+=sx
        if doubled<=dx:
            error+=dx
            y0+=sy


def scan_points_base(scan,mount_xy):
    angles=np.asarray(scan.angles_rad,dtype=float)
    ranges=np.asarray(scan.ranges_m,dtype=float)
    status=np.asarray(scan.status)
    if angles.ndim!=1 or not angles.size or angles.shape!=ranges.shape or angles.shape!=status.shape or not np.isfinite([angles,ranges]).all():
        raise ValueError("Scan arrays must be matching finite vectors.")
    if np.any(ranges<0) or not np.isin(status,[0,1,2]).all() or not math.isfinite(scan.time_s):
        raise ValueError("Invalid scan range, status or timestamp.")
    mount=finite_vector(mount_xy,2,"Declared lidar mount")
    return np.column_stack([ranges*np.cos(angles),ranges*np.sin(angles)])+mount


class ScanGrid:
    def __init__(self,parameters):
        self.origin=finite_vector(parameters["origin_xy_m"],2,"Grid origin")
        self.resolution=float(parameters["resolution_m"])
        extent=finite_vector(parameters["size_xy_m"],2,"Grid size")
        if not math.isfinite(self.resolution) or self.resolution<=0 or np.any(extent<=0):
            raise ValueError("Grid resolution and extent must be positive.")
        self.shape=tuple(np.ceil(extent[::-1]/self.resolution).astype(int))
        self.cells=np.zeros(self.shape,dtype=np.uint8)
        self.mount=finite_vector(parameters["lidar_mount_xy_m"],2,"Lidar mount")
        self.inflation=float(parameters["robot_radius_m"])+float(parameters["clearance_margin_m"])+self.resolution/math.sqrt(2)
        if not math.isfinite(self.inflation) or self.inflation<=0:
            raise ValueError("Inflation radius must be positive.")
        self.last_time=-math.inf

    def index(self,point):
        xy=np.floor((finite_vector(point,2,"Grid point")-self.origin)/self.resolution).astype(int)
        return int(xy[1]),int(xy[0])

    def centre(self,index):
        return self.origin+(np.asarray(index)[::-1]+.5)*self.resolution

    def inside(self,index):
        return 0<=index[0]<self.shape[0] and 0<=index[1]<self.shape[1]

    def update(self,scan,estimated_pose):
        pose=finite_vector(estimated_pose,3,"Estimated pose")
        points=scan_points_base(scan,self.mount)
        if scan.time_s<=self.last_time:
            raise ValueError("Map must not reuse old scans.")
        c,s=np.cos(pose[2]),np.sin(pose[2])
        rotation=np.array([[c,-s],[s,c]])
        origin=pose[:2]+rotation@self.mount
        start=self.index(origin)
        if not self.inside(start):
            raise ValueError("Estimated lidar origin lies outside the declared grid.")
        world=points@rotation.T+pose[:2]
        free,occupied=set(),set()
        for endpoint,status in zip(world,scan.status):
            ray=grid_line(start,self.index(endpoint))
            if status in (1,2):
                free.update(cell for cell in ray[:-1] if self.inside(cell))
                if self.inside(ray[-1]):
                    occupied.add(ray[-1])
            else:
                free.update(cell for cell in ray if self.inside(cell))
        if free:
            rows,cols=zip(*free)
            self.cells[rows,cols]=FREE
        if occupied:
            rows,cols=zip(*occupied)
            self.cells[rows,cols]=OCCUPIED
        self.last_time=float(scan.time_s)
        return {"free_cells_observed":len(free),"occupied_cells_observed":len(occupied),
                "map_free_count":int(np.count_nonzero(self.cells==FREE)),"map_occupied_count":int(np.count_nonzero(self.cells==OCCUPIED))}

    def blocked(self):
        # 未知和障碍都不可走；按保守圆形足迹和栅格半对角扩张。
        source=self.cells!=FREE
        count=math.ceil(self.inflation/self.resolution)
        padded=np.pad(source,count,constant_values=True)
        blocked=np.zeros(self.shape,dtype=bool)
        for dr in range(-count,count+1):
            for dc in range(-count,count+1):
                if math.hypot(dr,dc)*self.resolution<=self.inflation+1e-12:
                    blocked|=padded[count+dr:count+dr+self.shape[0],count+dc:count+dc+self.shape[1]]
        return blocked

    def segment_clear(self,first,last,blocked):
        cells=grid_line(self.index(first),self.index(last))
        for i,cell in enumerate(cells):
            if not self.inside(cell) or blocked[cell]:
                return False
            if i and cell[0]!=cells[i-1][0] and cell[1]!=cells[i-1][1]:
                if blocked[cell[0],cells[i-1][1]] or blocked[cells[i-1][0],cell[1]]:
                    return False
        return True

    def plan(self,start_xy,goal_xy,blocked=None):
        blocked=self.blocked() if blocked is None else blocked
        start,goal=self.index(start_xy),self.index(goal_xy)
        if not self.inside(start) or not self.inside(goal) or blocked[start] or blocked[goal]:
            return None,{"reason":"start_or_goal_blocked_or_unknown","expanded":0}
        frontier=[(math.dist(start,goal),0.,start)]
        costs={start:0.}
        previous={}
        expanded=0
        while frontier:
            _,cost,current=heapq.heappop(frontier)
            if cost>costs[current]+1e-12:
                continue
            expanded+=1
            if current==goal:
                route=[current]
                while current!=start:
                    current=previous[current]
                    route.append(current)
                route.reverse()
                points=[np.asarray(start_xy,dtype=float),*[self.centre(cell) for cell in route[1:-1]],np.asarray(goal_xy,dtype=float)]
                smoothed=[points[0]]
                index=0
                while index<len(points)-1:
                    candidate=len(points)-1
                    while candidate>index+1 and not self.segment_clear(points[index],points[candidate],blocked):
                        candidate-=1
                    if not self.segment_clear(points[index],points[candidate],blocked):
                        return None,{"reason":"endpoint_connection_blocked","expanded":expanded}
                    smoothed.append(points[candidate])
                    index=candidate
                return np.asarray(smoothed),{"reason":"planned","expanded":expanded,"grid_cost_m":cost*self.resolution,"raw_cells":len(route)}
            for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nxt=(current[0]+dr,current[1]+dc)
                if not self.inside(nxt) or blocked[nxt]:
                    continue
                if dr and dc and (blocked[current[0]+dr,current[1]] or blocked[current[0],current[1]+dc]):
                    continue
                new_cost=cost+math.hypot(dr,dc)
                if new_cost<costs.get(nxt,math.inf)-1e-12:
                    costs[nxt],previous[nxt]=new_cost,current
                    heapq.heappush(frontier,(new_cost+math.dist(nxt,goal),new_cost,nxt))
        return None,{"reason":"no_connected_known_free_route","expanded":expanded}


def scan_guard(scan,twist,mount_xy,parameters):
    points=scan_points_base(scan,mount_xy)[np.isin(scan.status,[1,2])]
    if np.any(np.asarray(scan.status)==2):
        return False,0.
    if not len(points):
        return True,None
    v,omega=finite_vector(twist,2,"Guard twist")
    times=np.linspace(0,parameters["guard_horizon_s"],8)
    turn=omega*times
    distance=v*times*np.sinc(turn/(2*np.pi))
    centres=np.column_stack([distance*np.cos(turn/2),distance*np.sin(turn/2)])
    minimum=float(np.linalg.norm(points[:,None,:]-centres[None,:,:],axis=2).min())
    return minimum>=parameters["guard_radius_m"],minimum


class GridNavigator:
    def __init__(self,parameters,grid_parameters,radius_m,track_m,frontier_parameters=None):
        self.parameters=dict(parameters)
        self.grid_parameters=dict(grid_parameters)
        self.grid=ScanGrid(grid_parameters)
        self.goal=finite_vector(parameters["goal_pose_m_m_rad"],3,"Navigation goal")
        if not all(math.isfinite(value) and value>0 for key,value in parameters.items() if key!="goal_pose_m_m_rad"):
            raise ValueError("Navigator scalar parameters must be finite and positive.")
        self.radius,self.track=radius_m,track_m
        self.path=None
        self.index=1
        self.last_check=-math.inf
        self.scan=None
        self.plan_events=[]
        self.done_since=None
        self.mapping_epoch=0
        self.frontier_parameters=None if frontier_parameters is None else dict(frontier_parameters)
        if self.frontier_parameters is not None and not all(math.isfinite(v) and v>0 for v in self.frontier_parameters.values()):
            raise ValueError("Frontier recovery bounds must be positive and finite.")
        self.frontier_enabled=False
        self.route_kind="global_goal"

    def reset_mapping(self):
        """全局位姿重定位后旧扫描地图不能继续当作同一坐标配准。"""
        self.grid=ScanGrid(self.grid_parameters)
        self.path=None
        self.index=1
        self.scan=None
        self.last_check=-math.inf
        self.done_since=None
        self.mapping_epoch+=1

    def frontier_plan(self,pose,blocked):
        """Choose a bounded, reachable known-free subgoal; unknown cells remain blocked."""
        p=self.frontier_parameters
        start=self.grid.index(pose[:2])
        if p is None or not self.grid.inside(start) or blocked[start]:
            return None,None,None
        current_distance=float(np.linalg.norm(self.goal[:2]-pose[:2]))
        candidates=[]
        for row,col in np.argwhere(~blocked):
            point=self.grid.centre((row,col))
            distance=float(np.linalg.norm(point-pose[:2]))
            remaining=float(np.linalg.norm(point-self.goal[:2]))
            if distance<=p["max_subgoal_step_m"] and remaining<=current_distance-p["min_goal_progress_m"]:
                candidates.append((remaining,distance,int(row),int(col),point))
        for _,_,_,_,point in sorted(candidates,key=lambda x:x[:4])[:int(p["max_candidates"])]:
            path,diagnostic=self.grid.plan(pose[:2],point,blocked)
            if path is not None and diagnostic["grid_cost_m"]<=p["max_subgoal_path_m"]:
                return path,diagnostic,point
        return None,None,None

    def update_scan(self,scan,estimated_pose):
        statistics=self.grid.update(scan,estimated_pose)
        self.scan=scan
        if scan.time_s-self.last_check<self.parameters["plan_check_period_s"]-1e-9:
            return statistics,None
        self.last_check=float(scan.time_s)
        blocked=self.grid.blocked()
        route=[np.asarray(estimated_pose)[:2],*self.path[self.index:]] if self.path is not None else []
        valid=bool(route) and all(self.grid.segment_clear(a,b,blocked) for a,b in zip(route,route[1:]))
        if valid and self.route_kind!="known_free_subgoal":
            return statistics,None
        self.path,diagnostics=self.grid.plan(np.asarray(estimated_pose)[:2],self.goal[:2],blocked)
        self.route_kind="global_goal"
        planning_goal=self.goal[:2]
        if self.path is None and self.frontier_enabled:
            path,detail,subgoal=self.frontier_plan(np.asarray(estimated_pose),blocked)
            if path is not None:
                self.path,diagnostics,planning_goal=path,detail,subgoal
                self.route_kind="known_free_subgoal"
        self.index=1
        event={"time_s":float(scan.time_s),"mapping_epoch":self.mapping_epoch,"estimated_pose":list(map(float,estimated_pose)),
               "path":None if self.path is None else self.path.tolist(),"diagnostics":diagnostics}
        if self.frontier_enabled:
            event.update(route_kind=self.route_kind,planning_goal_xy=list(map(float,planning_goal)))
        self.plan_events.append(event)
        return statistics,event

    def command(self,estimated_pose,P,time_s):
        pose=finite_vector(estimated_pose,3,"Estimated pose")
        P=np.asarray(P,dtype=float)
        if P.shape!=(3,3) or not np.isfinite(P).all() or np.linalg.eigvalsh((P+P.T)/2).min()<-1e-12 or not math.isfinite(time_s):
            raise ValueError("Navigation requires a valid pose covariance and finite time.")
        if self.scan is not None and time_s<self.scan.time_s-1e-7:
            raise ValueError("A future scan cannot guard a past command.")
        params=self.parameters
        state="WAIT_SCAN"
        twist=np.zeros(2)
        target=None
        distance=float(np.linalg.norm(self.goal[:2]-pose[:2]))
        if self.scan is None or time_s-self.scan.time_s>params["max_scan_age_s"]:
            state="STALE_SCAN_STOP"
            self.done_since=None
        elif np.sqrt(np.linalg.eigvalsh(np.asarray(P)[:2,:2]).max())>params["max_position_std_m"] or math.sqrt(P[2,2])>params["max_yaw_std_rad"]:
            state="UNCERTAINTY_STOP"
            self.done_since=None
        elif distance<=params["goal_tolerance_m"]:
            heading=float(wrap_angle(self.goal[2]-pose[2]))
            if abs(heading)>params["goal_yaw_tolerance_rad"]:
                state="ALIGN_GOAL"
                twist[1]=np.clip(params["heading_gain_per_s"]*heading,-params["max_omega_rad_s"],params["max_omega_rad_s"])
                self.done_since=None
            else:
                state="GOAL_HOLD"
                if self.done_since is None:
                    self.done_since=float(time_s)
                if time_s-self.done_since>=params["goal_hold_s"]:
                    state="ARRIVED"
        elif self.path is None:
            state="NO_PATH_STOP"
            self.done_since=None
        else:
            self.done_since=None
            while self.index<len(self.path)-1 and np.linalg.norm(self.path[self.index]-pose[:2])<params["waypoint_tolerance_m"]:
                self.index+=1
            target=self.path[self.index]
            relative=target-pose[:2]
            heading=float(wrap_angle(math.atan2(relative[1],relative[0])-pose[2]))
            twist[1]=np.clip(params["heading_gain_per_s"]*heading,-params["max_omega_rad_s"],params["max_omega_rad_s"])
            if abs(heading)>params["drive_heading_limit_rad"]:
                state="ALIGN_PATH"
            else:
                state="FOLLOW_PATH"
                twist[0]=min(params["cruise_m_s"],params["distance_gain_per_s"]*np.linalg.norm(relative))*max(0.,math.cos(heading))
        guard_clear,minimum=(True,None) if self.scan is None else scan_guard(self.scan,twist,self.grid.mount,params)
        if not guard_clear and np.any(abs(twist)>0):
            twist[:]=0
            state="SCAN_GUARD_STOP"
            self.done_since=None
        wheels=twist_to_wheel_rates(twist,self.radius,self.track)
        scale=min(1.,params["max_desired_wheel_rad_s"]/max(float(np.max(abs(wheels))),1e-15))
        return {"state":state,"desired_wheel_rad_s":(wheels*scale).tolist(),"body_twist":(twist*scale).tolist(),
                "goal_distance_estimated_m":distance,"target_xy":None if target is None else target.tolist(),
                "waypoint_index":self.index,"guard_min_point_distance_m":minimum,"wheel_scale":scale,
                "stop":bool(np.all(twist==0)),"arrived":state=="ARRIVED"}
