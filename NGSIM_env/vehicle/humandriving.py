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
