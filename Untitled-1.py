
import gymnasium
from stable_baselines3.common.utils import set_random_seed
import numpy as np
import dr_envs

task_num = 40
set_random_seed(42)

env = gymnasium.make("RandomHumanoid-v0")

default_task = env.unwrapped.get_task()
print("Default task:", default_task)