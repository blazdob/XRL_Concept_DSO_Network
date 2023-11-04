
import pandapower as pp
import numpy as np
import argparse
import pickle
import os

from stable_baselines3 import PPO, SAC
from env import VoltageControlEnv
from utils import LoadStream


###########################################################
#                                                         #
#################### PARSING ARGUMENTS ####################
#                                                         #
###########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="test the model", default="PPO")
parser.add_argument("--network_path", help="path to the network", default="src/data/env_data/Srakovlje.json")
parser.add_argument("--load_test_file", help="path to the load data", default="src/data/env_data/loads_and_generation_test.csv")
parser.add_argument("--voltage_threshold", help="threshold for voltage barier", default=0.2)
parser.add_argument("--episode_limit", help="limit of the episode", default=288)

parser.add_argument("--model_path", help="path to the model", default="src/saved_models/")
parser.add_argument("--model_name", help="name of the model", default="PPO_voltage_control1")
parser.add_argument("--results_path", help="path to the results", default="src/results_data/")
parser.add_argument("--results_file", help="name of the results file", default="results_PPO_voltage_control_29032023.pkl")
parser.add_argument("--device", help="device to use", default="cpu", type=str)

args = parser.parse_args()

print()
print("ARGUMENTS: \n", args)
print()

###########################################################
#                                                         #
#################### ENVIRONMENT SETUP ####################
#                                                         #
###########################################################
active_consumers = [1, 2, 5, 10, 11, 15, 16]
net = pp.from_json(args.network_path)
net.ext_grid.vm_pu = 1.0
loads_and_generation = LoadStream(args.load_test_file, index_col=[0])

env = VoltageControlEnv(net=net,
                        loads=loads_and_generation,
                        # generation=generation,
                        active_consumers=active_consumers,
                        voltage_threshold=args.voltage_threshold,
                        episode_limit=args.episode_limit)

ModelType = dict(
    PPO = PPO,
    SAC = SAC
)

###########################################################
#                                                         #
#################### MODEL SETUP #########################
#                                                         #
###########################################################
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym import spaces

# LRP class for QNetwork using linear layers in lrp folder and lienar file
from lrp import Sequential, Linear

import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomNNLRP(BaseFeaturesExtractor):
    """ Actor (Policy) Model."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, seed: int = 0, fc1_unit: int = 64, fc2_unit: int = 64):
        """
        Initialize parameters and build model.
        Params
        =======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.seed = torch.manual_seed(seed)
        self.model = Sequential(
            Linear(n_input_channels, fc1_unit),
            nn.ReLU(),
            Linear(fc1_unit, fc2_unit),
            nn.ReLU(),
            Linear(fc2_unit, features_dim)
        )

    def forward(self, x, explain=False, rule="epsilon", pattern=None):
        # x = state
        """
        Build a network that maps state -> action values.
        """
        return self.model(x, explain=explain, rule=rule, pattern=pattern)

###########################################################
#                                                         #
#################### UTILITY SETUP ########################
#                                                         #
###########################################################

def calculate_effect(env, action, diff):
    # for each action by changing it by diff calculate the effect on the state
    net = env.current_net.deepcopy()
    # locate the index of the bus with max voltage
    max_voltage_index = np.argmax(env.current_net.res_bus.vm_pu)
    diff_voltages = []
    for i, consumer_id in enumerate(env.active_consumers):
        # change the generation on i-th bus
        voltage_before = net.res_bus.loc[max_voltage_index, "vm_pu"]
        net.load.loc[consumer_id, "p_mw"] = net.load.loc[i, "p_mw"] * (1 - (action[i] + diff))
        # run power flow
        pp.runpp(net)
        voltage_after = net.res_bus.loc[max_voltage_index, "vm_pu"]
        v_diff = (voltage_before - voltage_after)*10000
        diff_voltages.append(v_diff)
    # get argmax of the diff_voltages
    argmax = np.argmax(diff_voltages)

    # convert diff_voltages to percentage
    diff_voltages_percentage = [x / sum(diff_voltages) for x in diff_voltages]
    # print(np.round(diff_voltages_percentage, 2))
    # print(diff_voltages)
    # get the bus of the env.active_consumers[argmax]
    bus = net.load.loc[env.active_consumers[argmax], "bus"]

    return diff_voltages, bus

###########################################################
#                                                         #
######################### TESTING #########################
#                                                         #
###########################################################

model = ModelType[args.model].load(os.path.join(args.model_path, args.model_name))

obs = env.reset()
diff = 0.1
results_dict = {}
with open(os.path.join(args.results_path, args.results_file), "w") as f:
    while True:
        action, _states = model.predict(obs)
        correlation, max_effect_consumer = calculate_effect(env, action, diff)
        obs, rewards, done, info = env.step(action)
        # print(correlation, max_effect_consumer)
        # print(max(obs), max(env.new_unchanged_state), action)
        results_dict[env.steps] = [correlation, max_effect_consumer, max(obs), max(env.new_unchanged_state), action]
        if done:
            break


# save results_dict to json
with open(os.path.join(args.results_path, args.results_file), "wb") as f:
    pickle.dump(results_dict, f)
