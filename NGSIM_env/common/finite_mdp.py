import importlib
from functools import partial
import numpy as np
from NGSIM_env import utils


def finite_mdp(env, time_quantization=1., horizon=10.):
    """
        Time-To-Collision (TTC) representation of the state.

        The state reward is defined from a occupancy grid over different TTCs and lanes. The grid cells encode the
        probability that the ego-vehicle will collide with another vehicle if it is located on a given lane in a given
        duration, under the hypothesis that every vehicles observed will maintain a constant velocity (including the
        ego-vehicle) and not change lane (excluding the ego-vehicle).

        For instance, in a three-lane road with a vehicle on the left lane with collision predicted in 5s the grid will
        be:
        [0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0]
        The TTC-state is a coordinate (lane, time) within this grid.

        If the ego-vehicle has the ability to change its velocity, an additional layer is added to the occupancy grid to
        iterate over the different velocity choice available.

        Finally, this states is flattened for compatibility with the FiniteMDPEnv environment.

    :param AbstractEnv env: an environment
    :param time_quantization: the time quantization used in the states representation [s]
    :param horizon: the horizon on which the collisions are predicted [s]
    """
    pass


def compute_ttc_grid(env, time_quantization, horizon, considered_lanes="all"):
    """
    For each ego-velocity and lane, compute the predicted time-to-collision to each vehicle within the lane and store
    the results in an occupancy grid.
    """
    road_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)  # 所有的lane index in this road
    grid = np.zeros((env.vehicle.SPEED_COUNT, len(road_lanes), int(horizon/time_quantization)))  # 初始化grid矩阵

    for velocity_index in range(grid.shape[0]):
        ego_velocity = env.vehicle.index_to_speed(velocity_index)

        for other in env.road.vehicles:
            if (other is env.vehicle) or (ego_velocity == other.velocity):  # 速度相同不会发生碰撞
                continue
            margin = other.LENGTH/2+env.LENGTH/2
            collision_points = [(0, 1), (-margin, 0.5), (margin, 0.5)]

            for m, cost in collision_points:
                distance = env.vehicle.lane_distance_to(other)+m
                other_projected_velocity = other.velocity*np.dot(other.direction, env.vehicle.direction)


