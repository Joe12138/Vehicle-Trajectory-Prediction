import numpy as np
import copy
import NGSIM_env.utils as utils
from NGSIM_env.vehicle.dynamics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane change.

    - The longitudinal controller is a velocity controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    TAU_LATERAL = 3  # [s]

    PURSUIT_TAU = 0.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / TAU_LATERAL  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]
    DELTA_VELOCITY = 2  # [m/s]

    def __init__(self, road, position, heading=0, velocity=0, target_lane_index=None, target_velocity=None, route=None):
        super(ControlledVehicle, self).__init__(road, position, heading, velocity)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_velocity = target_velocity or self.velocity
        self.route = route

    @classmethod
    def create_from(cls, vehicle):
        """
        Create a new vehicle from an existing one.
        The vehicle dynamics and target dynamics are copied, other properties are default.

        :param vehicle: a vehicle
        :return: a new vehicle at the same dynamical state
        """
        v = cls(vehicle.road, vehicle.position, heading=vehicle.heading, velocity=vehicle.velocity,
                target_lane_index=vehicle.target_lane_index, target_velocity=vehicle.target_velocity,
                route=vehicle.route)
        return v

    def plan_route_to(self, destination):
        """
        Plan a route to a destination in the road network

        :param destination: a node in the road network
        """
        path = self.road.network.shortest_path(self.lane_index[1], destination)
        if path:
            self.route = [self.lane_index]+[(path[i], path[i+1], None) for i in range(len(path)-1)]
        else:
            self.route = [self.lane_index]
        return self

    def act(self, action=None):
        """
        Perform a high-level action to change the desired lane or velocity.

        - If a high-level action is provided, update the target velocity and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        pass
