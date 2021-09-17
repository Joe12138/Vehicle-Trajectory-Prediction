from NGSIM_env.logger import Loggable
import numpy as np
from collections import deque

class Vehicle(Loggable):
    """
    A moving vehicle on a road, and its dynamics.

    The vehicle is represented by a dynamics system: a modified bicycle model.
    It's state is propagated depending on its steering and acceleration actions.
    """

    # Enable collision detection between vehicles 
    COLLISIONS_ENABLED = True
    # vehicle length [m]
    LENGTH = 5.0
    # vehicle width [m]
    WIDTH = 2.0
    # Range for random innitial velocities [m/s]
    DEFAULT_VELOCITIES = [23, 25]
    # Maximum reachable velocity [m/s]
    MAX_VELOCITY = 40

    def __init__(self, road, position, heading=0, velocity=0):
        self.road = road
        self.position = np.array(position).astype("float")
        self.heading = heading
        self.velocity = velocity
        self.lane_index = self.road.network.get_closest_lane_index(self.position) if self.road else np.nan
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None
        self.action = {"steering": 0, "acceleration": 0}
        self.carshed = False
        self.history = deque(maxlen=50)


    @classmethod
    def make_on_lane(cls, road, lane_index, longitudinal, velocity=0):
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: the road where the vehicle is driving
        :param lane_index: index of the lane where the vehicle is located
        :param longitudinal: longitudinal position along the lane
        :param velocity: initial velocity in [m/s]
        :return A vehicle with at the specified position
        """

        lane = road.network.get_lane(lane_index)

        if velocity is None:
            velocity = lane.speed_limit

        return cls(road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), velocity)

    @classmethod
    def create_random(cls, road, velocity=None, spacing=1):
        """
        """
        pass