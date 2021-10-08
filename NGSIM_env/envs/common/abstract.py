import gym
import copy
from gym import spaces
from gym.utils import seeding

from NGSIM_env.vehicle.control import MDPVehicle
from NGSIM_env.envs.common.observation import observation_factory
from NGSIM_env.envs.common.graphics import EnvViewer


class AbstractEnv(gym.Env):
	"""
	A generic environment for various tasks involving a vehicle driving on a road.

	Tne environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
	velocity. The action space is fixed, but the observation space and reward function must be defined in the 
	environment implementations.
	"""
	metadata = {'render.modes': ['human', 'rgb_array']}

	# A mapping of action indexed to action labels
	ACTIONS = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}

	# A mapping of action labels to action indexed
	ACTION_INDEXES = {v: k for k, v in ACTIONS.items()}

	# The frequency at which the system dynamics are simulated [Hz]
	SIMULATION_FREQUENCY = 10

	# The maximum distance of any vehicle present in the observation [m]
	PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX

	def __init__(self, config=None):
		# Configuration
		self.config = self.default_config()
		if config:
			self.config.update(config)

		# Seeding
		self.np_random = None
		self.seed()

		# Scene
		self.road = None
		self.vehicle = None

		# Spaces
		self.observation = None
		self.action_space = None
		self.observation_space = None
		self.define_spaces()

		# Running
		self.time = 0  # Simulation time
		self.steps = 0  # Actions performed
		self.done = False

		# Rendering
		self.viewer = None
		self.automatic_rendering_callback = None
		self.should_update_rendering = True
		self.rendering_mode = "human"
		self.offscreen = self.config.get("offscreen_rendering", False)
		self.enable_auto_render = False

		self.reset()

	@classmethod
	def default_config(cls):
		"""
		Default environment configuration.

		Can be overloaded in environment implementations, or by calling configure().
		:return: a configuration dict
		"""
		return {
			"observation": {"type": "TimeToCollision"},
			"policy_frequency": 1,  # [Hz]
			"other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
			"screen_width": 640,  # [px]
			"screen_height": 320,  # [px]
			"centering_position": [0.5, 0.5],
			"show_trajectories": False
		}

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def configure(self, config):
		if config:
			self.config.update(config)

	def define_spaces(self):
		self.action_space = spaces.Discrete(len(self.ACTIONS))

		if "observation" not in self.config:
			raise ValueError("The observation configuration must be defined")
		self.observation = observation_factory(self, self.config["observation"])
		self.observation_space = self.observation.space()

	def _reward(self, action):
		"""
		Return the reward associated with performing a given action and ending up in the current state.

		:param action: the last action performed
		:return: the reward
		"""

		raise NotImplementedError

	def _is_terminal(self):
		"""
		Check whether the current state is a terminal state
		:return: is the state terminal
		"""
		raise NotImplementedError

	def _cost(self, action):
		"""
		A constraint metric, for budgeted MDP.

		If a constraint id defined, it must be used with an alternate reward that doesn't contain it as a penalty.
		:param action: the last action performed
		:return: the constraint signal, the alternate (constraint-free) reward
		"""
		raise NotImplementedError

	def reset(self):
		"""
		Reset the environment to it's initial configuration
		:return: the observation of the reset state
		"""
		self.time = 0
		self.done = False
		self.define_spaces()

	def step(self, action):
		"""
		Perform an action and step the environment dynamics.

		The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
		for several simulation timesteps until the next decision making step.

		:param int action: the action performed by the ego-vehicle
		:return: a tuple (observation, reward, terminal, info)
		"""
		if self.road is None or self.vehicle is None:
			raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

	def _simulate(self, action=None):
		"""
		Perform several steps of simulation with constant action
		"""
		for _ in range(int(self.SIMULATION_FREQUENCY//self.config["policy_frequency"])):
			if action is not None and self.time % int(self.SIMULATION_FREQUENCY//self.config["policy_frequency"]) == 0:
				# Forward action to the vehicle
				self.vehicle.act(self.ACTIONS[action])

			self.road.act()
			self.road.step(1/self.SIMULATION_FREQUENCY)
			self.time += 1

	def render(self, mode='human'):
		"""
		Render the environment.

		Create a viewer if none exists, and use it to render an image.
		:param mode: the rendering mode
		"""
		self.rendering_mode = mode

		if self.viewer is None:
			self.viewer = EnvViewer(self, offscreen=self.offscreen)

		self.enable_auto_render = not self.offscreen

		# If the frame has already been rendered, do nothing
		if self.should_update_rendering:
			self.viewer.display()

		if mode == "rgb_array":
			image = self.viewer.get_image()
			return image

		self.should_update_rendering = False

	def close(self):
		"""
		Close the environment.

		Will close the environment viewer if it exists.
		"""
		self.done = True
		if self.viewer is not None:
			self.viewer.close()

		self.viewer = None

	def get_available_actions(self):
		"""
		Get the list of currently available actions.

		Lane changes are not available on the boundary of the road, the velocity changes are not available at maximal or
		minimal velocity.

		:return : the list of available actions
		"""
		actions = [self.ACTION_INDEXES["IDLE"]]

		for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
			if l_index[2] < self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(
					self.vehicle.position):
				actions.append(self.ACTION_INDEXES["LANE_LEFT"])
			if l_index[2] > self.vehicle.lane_index[2] and self.road.network.get_lane(l_index).is_reachable_from(
					self.vehicle.position):
				actions.append(self.ACTION_INDEXES["LANE_RIGHT"])

		if self.vehicle.velocity_index < self.vehicle.SPEED_COUNT-1:
			actions.append(self.ACTION_INDEXES["FASTER"])
		if self.vehicle.velocity_index > 0:
			actions.append(self.ACTION_INDEXES["SLOWER"])

		return actions

	def _automatic_rendering(self):
		"""
		Automatically render the intermediate frames while an action is still ongoing.
		This allows to render the whole video and not only single steps corresponding to agent decision-making.

		If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers such as
		video-recording monitor that need to access these intermediate renderings.
		"""

		if self.viewer is not None and self.enable_auto_render:
			self.should_update_rendering = True

			if self.automatic_rendering_callback:
				self.automatic_rendering_callback()
			else:
				self.render(self.rendering_mode)

	def simplify(self):
		"""
		Return a simplified copy of the environment where distant vehicles have been removed from the road.

		This is meant to lower the policy computational load while preserving the optimal actions set

		:return: a simplified environment state.
		"""
		state_copy = copy.deepcopy(self)
		state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicle_to(
			state_copy.vehicle, self.PERCEPTION_DISTANCE)
		return state_copy

