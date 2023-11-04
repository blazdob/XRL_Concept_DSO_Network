from voltage_barrier.voltage_main import VoltageBarrier
from pandapower import ppException
import pandapower as pp
from gym import spaces
import numpy as np
import copy
import gym
# from plotly_animation import pf_res_plotly_animate
from pandapower.plotting.plotly import pf_res_plotly
from plotly_animation import pf_res_plotly_animate

N_CHANNELS = 3

class VoltageControlEnv1(gym.Env):
	"""Custom Environment that follows gym interface."""

	metadata = {"render.modes": ["human"]}

	def __init__(self, net=None, 
					loads=None, 
					generation=None,
					active_consumers=None,
					voltage_threshold=None,
					episode_limit=None,
					use_forecast=False,):
		super().__init__()
		self.base_net = net
		self.current_net = copy.deepcopy(net)
		self.load = loads
		# self.generation = generation
		self.v_lower = 0.95
		self.v_upper = 1.05

		self.voltage_threshold = voltage_threshold # the allowed teoretical limit is between 0.8 and 1.2 p.u.
		self.episode_limit = episode_limit
		self.steps_beyond_done = None
		self.active_consumers = active_consumers

		self.use_forecast = use_forecast
		#voltage barrier function
		# self.voltage_barrier_type = voltage_barrier_type
		# self.voltage_barrier = VoltageBarrier(self.voltage_barrier_type)
		
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using continous actions:
		self.action_space = spaces.Box(low=0, high=1,
											shape=(len(active_consumers), ), dtype=np.float32)
		# Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=(1 - self.voltage_threshold), high=(1 + self.voltage_threshold),
											shape=(len(self.base_net.bus), ), dtype=np.float32)
		# run powerflow to get initial state
		self.state = self.reset()
		self.steps = 0
		self.error_counter = 0
	

	def step(self, action):
		err_msg = f"{action!r} ({type(action)}) invalid"
		assert self.action_space.contains(action), err_msg
		assert self.state is not None, "Call reset before using step method."

		# self.state --> action --> updated_state --> reward

		############# TAKING THE ACTION ################
		solvable, new_state = self._take_action(action)

		############# CALCULATE THE REWARD ################
		if solvable:
			reward, info = self._calc_reward(new_state, action)
		else:
			print("not succeded running loadflow")
			info = {}
			reward = 0
			# print("not succeded running loadflow")
			info["destroy"] = 1
			info["totally_controllable_ratio"] = 0
		# self.sum_rewards += reward
		# calculating if we are finished
		done = bool(
			np.any(self.current_net.res_bus.vm_pu < (1 - self.voltage_threshold))
			and np.any(self.current_net.res_bus.vm_pu > (1 + self.voltage_threshold))
		)

		if self.steps >= self.episode_limit or not solvable:
			done = True
		else:
			done = False
			self.steps += 1

		return new_state, float(reward), done, info

	def get_random_action(self):
		"""return the action according to a uniform distribution over [action_lower, action_upper)
		"""
		return self.action_space.sample()

	def _calc_reward(self, new_state, action, info={}):
		

		# # percentage of voltage out of control
		v = self.current_net.res_bus["vm_pu"].sort_index().to_numpy(copy=True)
		info["percentage_of_higher_limit"] = np.sum(v > self.v_upper) / v.shape[0]
		percent_of_v_out_of_control = ( np.sum(v < self.v_lower) + np.sum(v > self.v_upper) ) / v.shape[0]
		info["percentage_of_v_out_of_control"] = percent_of_v_out_of_control
		info["percentage_of_lower_than_v_lower"] = np.sum(v < self.v_lower) / v.shape[0]
		info["percentage_of_higher_than_v_upper"] = np.sum(v > self.v_upper) / v.shape[0]
		info["totally_controllable_ratio"] = 0. if percent_of_v_out_of_control > 1e-3 else 1.


		# get indexes of the new unchanged state where the voltages are over the limit
		new_states_where_old_exceeded = np.where(self.new_unchanged_state > self.v_upper, new_state, 0)
		
		new_states_where_old_not_exceed = np.where(self.new_unchanged_state < self.v_upper, new_state, 0)
		alpha = 1
		intermediate_reward = 0
		# check voltages on busses of all active consumers and calculate the distance?

		for el in new_states_where_old_exceeded:
			# print(el)
			if el != 0:
				intermediate_reward -= max(0, (el - self.v_upper))**2*1000
		
		if self.new_unchanged_state.max() < self.v_upper:
			intermediate_reward -= np.sum(action)

		reward = 0
		# give a punishment for the number of violations
		reward += alpha * intermediate_reward

		info["number_of_violations"] = np.sum(new_state > self.v_upper)
		
		# print(self.new_unchanged_state - state, max(state), max(self.new_unchanged_state))
		
		# punish for eccessive actions
		# reward -= np.sum(action)
		print(self.new_unchanged_state.max(), new_state.max(), reward)

		# # voltage violations
		v_ref = 0.5 * (self.v_lower + self.v_upper)
		info["average_voltage_deviation"] = np.mean( np.abs( new_state - v_ref ) )
		info["average_voltage"] = np.mean(new_state)
		info["max_voltage_drop_deviation"] = np.max( (new_state < self.v_lower) * (self.v_lower - new_state) )
		info["max_voltage_rise_deviation"] = np.max( (new_state > self.v_upper) * (new_state - self.v_upper) )
		
		# v_loss = np.sum(self.voltage_barrier.step(new_state))
		# line loss
		line_loss = np.sum(self.current_net.res_line["pl_mw"])
		# povprečna izguba linij
		# avg_line_loss = np.mean(self.current_net.res_line["pl_mw"])
		info["total_line_loss"] = line_loss
		info["percentage_of_higher_than_1"] = np.sum(new_state > 1.0) / new_state.shape[0]

		
		return reward, info

	def _take_action(self, action):
		# # setting next states of generation and load based on an actions
		load = self.load.curr() * 10

		# empty vector of length of the number of loads
		next_updated_loads = load.copy()
		# fill the vectors with the next values
		for i, active_i in enumerate(self.active_consumers):
			if load[active_i] <= 0.0:
				next_updated_loads[active_i] = load[active_i] * (1 - action[i])

		self.current_net.load.p_mw = next_updated_loads
		self.load.next()
		try:
			pp.runpp(self.current_net)
			self.state = self.current_net.res_bus.vm_pu.values.astype(np.float32)
			return True, self.state
		except:
			#if loadflow could not be calculated the reward should be negative, since the state is not valid
			print("The power flow cannot be solved, the cutof was too big.")
			self.error_counter +=1
			# setting the old network back
			self.state = self.current_net.res_bus.vm_pu.values.astype(np.float32)
			return False, self.state

	def reset(self, starting=None):
		self.steps = 0
		self.current_net = copy.deepcopy(self.base_net)
		solvable = False
		while not solvable:
			# set the random starting data
			if starting != None:
				self.load.current = starting
				# self.generation.current = starting
			else:
				self.load.set_random_starting_pos(self.episode_limit)

			# curr_gen = self.generation.curr()
			curr_load = self.load.curr()

			# self.current_net.sgen.p_mw = curr_gen
			self.current_net.load.p_mw = curr_load
			# if random reset, then set the random action to be appended to the sgens
			try:
				pp.runpp(self.current_net)
				solvable = True
			except ppException:
				print ("The power flow for the initialisation of demand and PV cannot be solved.")
				print (f"This is the active demand: \n{self.current_net.load['p_mw']}")
				print (f"This is the res_bus: \n{self.current_net.res_bus}")
				solvable = False
		self.state = self.current_net.res_bus.vm_pu.values.astype(np.float32)
		return self.state

	def render(self, mode="human"):
		"""
		Function renders plotly pandapower image using 
		:param mode: "human" or "rgb_array"
		:return: rgb_array
		"""
		pf_res_plotly_animate(self.current_net,fig=None, active_consumers=self.active_consumers)


