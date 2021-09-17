import gym

class AbstractEnv(gym.Env):
	"""
	A generic environment for various tasks involving a vehicle driving on a road.

	Tne environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
	velocity. The action space is fixed, but the observation space and reward function must be defined in the 
	environment implementations.
	"""
	metadata = {'render.modes': ['human', 'rgb_array']}

	# A mapping of action indexed to acrtion labels
	ACTIONS = {0: "LANE_LEFT", 
			   1: "IDLE",
			   2: "LANE_RIGHT", 
			   3: "FASTER",
			   4: "SLOWER"}

	# A mapping of action labels to action indexed
	ACTION_INDEXES = {v: k for k, v in ACTIONS.items()}

	# The frequency at which the system dunamics are simulated [Hz]
	SIMULATION_FREQUENCY = 10

	# The maximum distance of any vehicle present in the observation [m]
	PERCEPTION_DISTANCE = 6.0 * 
