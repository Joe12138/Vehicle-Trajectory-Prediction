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
                time_to_collision = distance/utils.not_zero(ego_velocity-other_projected_velocity)
                if time_to_collision < 0:
                    continue
                if env.road.network.is_connected_road(env.vehicle.lane_index, other.lane_index,
                                                      route=env.vehicle.route, depth=3):
                    # Same road, or connected road with same number of lanes
                    if len(env.road.network.all_side_lanes(other.lane_index)) == len(env.road.network.all_side_lanes(
                            env.vehicle.lane_index)):
                        lane = [other.lane_index[2]]
                    # Different road of different number of lanes: uncertainty on future lane, use all
                    else:
                        lane = range(grid.shape[1])

                    # Quantize time-to-collision to both upper and lower values
                    for time in [int(time_to_collision/time_quantization),
                                 int(np.ceil(time_to_collision/time_quantization))]:
                        if 0 <= time < grid.shape[2]:
                            grid[velocity_index, lane, time] = np.maximum(grid[velocity_index, lane, time], cost)

    return grid


def transition_model(h, i, j, a, grid):
    """
    Deterministic transition from a position in the grid to the next.

    :param h: velocity index
    :param i: lane index
    :param j: time index
    :param a action index
    :param grid: ttc grid specifying the limits of velocities, lanes, time and actions
    """
    # Idle action (1) as default transition
    next_state = clip_position(h, i, j+1, grid)
    left = a == 0
    right = a == 2
    faster = (a == 3) & (j == 0)
    slower = (a == 4) & (j == 0)
    next_state[left] = clip_position(h[left], i[left] - 1, j[left] + 1, grid)
    next_state[right] = clip_position(h[right], i[right] + 1, j[right] + 1, grid)
    next_state[faster] = clip_position(h[faster] + 1, i[faster], j[faster] + 1, grid)
    next_state[slower] = clip_position(h[slower] - 1, i[slower], j[slower] + 1, grid)
    return next_state


def clip_position(h, i, j, grid):
    """
    Clip a position in the TTC grid, so that it stays within bounds.

    :param h: velocity index
    :param i: lane index
    :param j: time index
    :param grid: the ttc grid
    :return: The raveled index of the clipped position
    """
    h = np.clip(h, 0, grid.shape[0]-1)
    i = np.clip(i, 0, grid.shape[1]-1)
    j = np.clip(j, 0, grid.shape[2]-1)
    indexes = np.ravel_multi_index((h, i, j), grid.shape)

    return indexes


