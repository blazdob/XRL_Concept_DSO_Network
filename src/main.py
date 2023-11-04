import pandapower as pp
import numpy as np
import argparse
import pickle
import os

import torch.utils.tensorboard
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, SAC

from env import VoltageControlEnv
from utils import LoadStream


###########################################################
#                                                         #
#################### PARSING ARGUMENTS ####################
#                                                         #
###########################################################
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train the model", default=True)
parser.add_argument("--model", help="test the model", default="PPO")
parser.add_argument("--network_path", help="path to the network", default="src/data/env_data/Srakovlje.json")
parser.add_argument("--load_train_file", help="path to the load data", default="src/data/env_data/loads_and_generation_train.csv")
parser.add_argument("--load_test_file", help="path to the load data", default="src/data/env_data/loads_and_generation_test.csv")
parser.add_argument("--voltage_threshold", help="threshold for voltage barier", default=0.2)
parser.add_argument("--episode_limit", help="limit of the episode", default=33623)
parser.add_argument("--num_episodes", help="number of episodes", default=1000)

parser.add_argument("--tensorboard_log", help="use tensorboard logger or not?", default=True)
parser.add_argument("--tensorboard_log_path", help="path to tensorboard log", default="src/voltage_control_tensorboard/")
parser.add_argument("--model_path", help="path to the model", default="src/saved_models/")
parser.add_argument("--model_name", help="name of the model", default="PPO_voltage_control1")
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

# loads = Load("src/data/env_data/loads.csv", index_col=[0])
# generation = Generation("src/data/env_data/generation.csv", index_col=False)
if args.train:
    loads_and_generation = LoadStream(args.load_train_file, index_col=[0])
else:
    loads_and_generation = LoadStream(args.load_test_file, index_col=[0])

env = VoltageControlEnv(net=net,
                        loads=loads_and_generation,
                        # generation=generation,
                        active_consumers=active_consumers,
                        voltage_threshold=args.voltage_threshold,
                        episode_limit=args.episode_limit,
                        use_forecast=True,)

check_env(env, skip_render_check=True)
env.reset()

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, epidose_limit=384):
        super().__init__()
        self.episode_limit = epidose_limit
    
    def _on_training_start(self):
        self._log_freq = 1000  # log every 1000 calls

        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("number_of_violations", self.locals["infos"][0]["number_of_violations"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("percentage_of_higher_limit", self.locals["infos"][0]["percentage_of_higher_limit"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("percentage_of_v_out_of_control", self.locals["infos"][0]["percentage_of_v_out_of_control"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("percentage_of_lower_than_v_lower", self.locals["infos"][0]["percentage_of_lower_than_v_lower"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("percentage_of_higher_than_v_upper", self.locals["infos"][0]["percentage_of_higher_than_v_upper"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("totally_controllable_ratio", self.locals["infos"][0]["totally_controllable_ratio"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("average_voltage_deviation", self.locals["infos"][0]["average_voltage_deviation"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("average_voltage", self.locals["infos"][0]["average_voltage"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("max_voltage_drop_deviation", self.locals["infos"][0]["max_voltage_drop_deviation"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("max_voltage_rise_deviation", self.locals["infos"][0]["max_voltage_rise_deviation"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("total_line_loss", self.locals["infos"][0]["total_line_loss"], self.num_timesteps)
            self.tb_formatter.writer.add_scalar("percentage_of_higher_than_1", self.locals["infos"][0]["percentage_of_higher_than_1"], self.num_timesteps)

            self.tb_formatter.writer.flush()

ModelType = dict(
    PPO = PPO,
    SAC = SAC
)

##################################################
#                                                #
#################### TRAINING ####################
#                                                #
##################################################

args.train = False

if args.train:
    print("Training model for {} episodes".format(args.num_episodes))
    if args.tensorboard_log:
        model = ModelType[args.model]("MlpPolicy",
                                        env,
                                        verbose=1,
                                        device=args.device,
                                        tensorboard_log=args.tensorboard_log_path)
    else:
        model = ModelType[args.model]("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.num_episodes*args.episode_limit,
                callback=TensorboardCallback(epidose_limit=args.episode_limit),
                reset_num_timesteps=True)
    
    
    model.save(os.path.join(args.model_path, args.model_name))
    
    del model # remove to demonstrate saving and loading

#################### TESTING ####################

model = ModelType[args.model].load(os.path.join(args.model_path, args.model_name))

obs = env.reset()
error = 0
traj = {"obs": [], "unchanged_state": [], "action": [], "rewards": []}
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.plot_scenario_explanation(obs, env.new_unchanged_state, action, rewards)
    error += rewards
    # env.render()
    if done:
        break
    traj["obs"].append(obs)
    traj["unchanged_state"].append(env.new_unchanged_state)
    traj["action"].append(action)
    traj["rewards"].append(rewards)
#save the results to pickle file
with open("results.pkl", "wb") as f:
    pickle.dump(traj, f)