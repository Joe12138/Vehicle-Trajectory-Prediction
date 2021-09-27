import numpy as np
from NGSIM_env.vehicle.control import ControlledVehicle
import NGSIM_env.utils as utils


class IDMVehicle(ControlledVehicle):
    """
    A vehicle using a longitudinal and a lateral decision policies.

    - Longitudinal: the IDM model compute an acceleration given the preceding (前面的) vehicle's distance and velocity.
    - Lateral: the MOBIL model decide when to change lane by maximizing the acceleration of nearby vehicles.
    """

    # Longitudinal policy parameters
    ACC_MAX = 1.5  # [m/s2]
    COMFORT_ACC_MAX = 0.7  # [m/s2]
    COMFORT_AAC_MIN = -0.7  # [m/s2]
    DISTANCE_WANTED = 1.5  # [m]
    TIME_WANTED = 1.2  # [s]
    DELTA = 4.0  # []

    # lateral policy parameters
    POLITENESS = 0.01  # in [0, 1]
    LANE_CHANGE_MIN_ACC_GAIN = 0.2  # [m/s2]
    LANE_CHANGE_MAX_BREAKING_IMPOSED = 2.0  # [m/s2]
    LANE_CHANGE_DELAY = 1.0  # [s]

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, target_velocity=None, route=None,
                 enable_lane_change=True, timer=None):
        super(IDMVehicle, self).__init__(road, position, heading, velocity, target_lane_index, target_velocity, route)
        self.enable_lane_change = enable_lane_change
        self.timer = timer or (np.sum(self.position)*np.pi) % self.LANE_CHANGE_DELAY

    def randomize_behavior(self):
        pass

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.targer_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route, timer=getattr(vehicle, "timer", None))
        return v

    def act(self, action=None):
        """
        Execute an action.

        For now, no action is supported because the vehicle takes all decisions of acceleration and lane changes on its
        own, based on the IDM abd MOBIL model.

        :param action: the action
        """
        if self.crashed:
            return
        action = {}
        front_vehicle, rear_vehicle = self.road.neighbour_vehicles(self)

