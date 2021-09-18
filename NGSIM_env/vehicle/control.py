import numpy as np
import copy
import NGSIM_env.utils as utils
from NGSIM_env.vehicle.dynamics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actionbs such as 
    """