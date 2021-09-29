import numpy as np
import NGSIM_env.utils as utils
from NGSIM_env.vehicle.behavior import IDMVehicle
from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.vehicle.planner import planner


class NGSIMVehicle(IDMVehicle):
    """
    Use NGSIM human driving trajectories
    """
    # Longitudinal policy parameters
    ACC_MAX = 5.0  # [m/s2] Maximum acceleration
    COMFORT_ACC_MAX = 3.0  # [m/s2] Desired maximum acceleration
    COMFORT_ACC_MIN = -3.0  # [m/s2] Desired minimum acceleration
    DISTANCE_WANTED = 1.0  # [m] Desired jam distance to the front vehicle
    TIME_WANTED = 0.5  # [s] Desired time gap to the front vehicle
    DELTA = 4.0  # [] Exponent of the velocity term

    # Lateral policy parameters [MOBIL]
    POLITENESS = 0.1  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BREAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    # Driving scenario
    SCENE = "us-101"

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, target_velocity=None, route=None,
                 enable_lane_change=False, timer=None, vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None):
        super(NGSIMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route,
                                           enable_lane_change, timer)
        self.ngsim_traj = ngsim_traj
        self.traj = np.array(self.position)
        self.vehicle_ID = vehicle_ID
        self.sim_steps = 0
        self.overtaken = False
        self.appear = True if self.position[0] != 0 else False
        self.velocity_history = []
        self.heading_history = []
        self.crash_history = []
        self.overtaken_history = []

        # vehicle length [m]
        self.LENGTH = v_length
        # Vehicle width [m]
        self.WIDTH = v_width

    @classmethod
    def create(cls, road, vehicle_ID, position, v_length, v_width, ngsim_traj, heading=0, velocity=15):
        """
        Create a new NGSIM vehicle.

        :param road: the road where the vehicle is driving
        :param vehicle_ID: NGSIM vehicle ID
        :param position: the position here the vehicle start on the road
        :param v_length: vehicle length
        :param v_width: vehicle width
        :param ngsim_traj: NGSIM trajectory
        :param velocity: initial velocity in [m/s]. If None, will be chosen randomly.
        :param heading: initial heading

        :return: A vehicle with NGSIM position and velocity
        """

        v = cls(road, position, heading, velocity, vehicle_ID=vehicle_ID, v_length=v_length, v_width=v_width,
                ngsim_traj=ngsim_traj)

        return v

    def act(self, action=None):
        """
        Execute an action when NGSIM vehicle is overridden.

        :param action: the action
        """
        if not self.overtaken:
            return
        if self.crashed:
            return

        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicle(self)

        # Lateral: MOBIL
        self.follow_road()
        if self.enable_lane_change:
            self.change_lane_policy()
        action["steering"] = self.steering_control(self.target_lane_index)

        # Longitudinal: IDM
        action["acceleration"] = self.acceleration(ego_vehicle=self, front_vehicle=front_vehicle,
                                                   rear_vehicle=rear_vehicle)
        action["acceleration"] = np.clip(action["acceleration"], -self.ACC_MAX, self.ACC_MAX)
        self.check_collision()  # should be a method
        self.action = action

    def step(self, dt):
        """
        Update the state of a NGSIM vehicle.
        If the front vehicle is too close, use IDM model to override the NGSIM vehicle.
        """
        self.appear = True if self.ngsim_traj[self.sim_steps][0] != 0 else False
        self.timer += dt
        self.sim_steps += 1
        self.heading_history.append(self.heading)
        self.velocity_history.append(self.velocity)
        self.crash_history.append(self.crashed)
        self.overtaken_history.append(self.overtaken)

        # Check if need to overtake
        front_vehicle, rear_vehicle = self.road.neighbour_vehicle(self)
        if front_vehicle is not None and isinstance(front_vehicle, NGSIMVehicle) and front_vehicle.overtaken:
            gap = self.lane_distance_to(front_vehicle)
            desired_gap = self.desired_gap(self, front_vehicle)
        elif front_vehicle is not None and isinstance(front_vehicle, HumanLikeVehicle)


class HumanLikeVehicle(IDMVehicle):
    """
    Create a human-like (IRL) driving agent.
    """
    TAU_A = 0.2  # [s]
    TAU_DS = 0.1  # [s]
    PURSUIT_TAU = 1.5*TAU_DS  # [s]
    KP_A = 1/TAU_A
    KP_HEADING = 1/TAU_DS
    KP_LATERAL = 1 / 0.2  # [1/s]
    MAX_STEERING_ANGLE = np.pi/3  # [rad]
    MAX_VELOCITY = 30  # [m/s]

    def __init__(self, road, position, heading=0, velocity=0, acc=0, target_lane_index=None, target_velocity=15,
                 route=None, timer=None, vehicle_ID=None, v_length=None, v_width=None, ngsim_traj=None, human=False,
                 IDM=False):
        pass


