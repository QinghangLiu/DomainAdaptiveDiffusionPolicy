# from OpenGL.GL import glGetError
# print(glGetError())
import gym
import gymnasium
import dr_envs
import argparse
import numpy as np
from policy.policy import Policy
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from utils.utils import *
from utils.gym_utils import *
import wandb
import torch
import imageio
from stable_baselines3 import SAC
import time
import minari
from torch.utils.data import DataLoader

if __name__ == "__main__":
    wandb.init(project="your_project_name",mode = "disabled")

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',           default="sac", type=str, help='Algorithm to use')
    parser.add_argument('--env',            default="RandomHalfCheetah-v0", type=str, help='Train gym env')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=16, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--gradient_steps', default=4, type=int, help='Number of gradient steps per update')
    parser.add_argument('--model_path',     default="./best_model/best_model_heavy.zip", help='Train from scratch')
    parser.add_argument('--mode',           default="data", help='Evaluate the policy')
    parser.add_argument('--task',           default=np.array([2,7,7,7,7,7,7,1.8]), help='Use wandb for logging')
    parser.add_argument('--video_save_path',default="./video", help='Path to save video')
    parser.add_argument('--data_collect_episode',            default=800, type=int, help='Number of cpus for parallelization')
    args = parser.parse_args()
    env = gym.make(args.env)
    task = args.task

    # print(env.get_task())
    env.set_task(*task)
    par_env = make_vec_env(args.env, n_envs=args.now, seed=None, vec_env_cls=RandomSubprocVecEnv)

    par_env.set_task(np.tile(task, (args.now,1)))
    eff_lr = get_learning_rate(args, par_env)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.path.exists(args.model_path):
        # agent = Policy(algo=args.algo,
        #         env=par_env,
        #         lr=eff_lr,
        #         gamma=args.gamma,
        #         load_from_pathname=args.model_path,
        #         gradient_steps= args.gradient_steps,
        #         device="cuda"if  torch.cuda.is_available() else "cpu",     
        #                     )
        model = SAC.load(args.model_path, env=par_env, device="cuda" if torch.cuda.is_available() else "cpu")
    else:
        # agent = Policy(algo=args.algo,
        #         env=par_env,
        #         lr=eff_lr,
        #         gamma=args.gamma,
        #         gradient_steps= args.gradient_steps,
        #         device="cuda"if  torch.cuda.is_available() else "cpu",)     
        model = SAC("MlpPolicy", par_env, learning_rate=eff_lr, gamma=args.gamma, gradient_steps=args.gradient_steps, verbose=1, device="cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    if args.mode == "eval":

        accu_reward = 0
        frames = []
        print(f"=============== Parameter of Dynamics ====================================")
        print(env.get_task())
        obs = env.reset()

        done = False
        random_action = 0
        for _ in range(16):
            obs,_,_,_ = env.step(np.random.uniform(low = -1,high = 1,size = (par_env.action_space.shape[0])))
            frame = env.render(mode = "rgb_array")
        while not done:
            if np.random.rand() < 0.02 and random_action == 0:
                random_action = 8
            if random_action > 0:
                action = np.random.uniform(low = -1,high = 1,size = (par_env.action_space.shape[0]))
                random_action -= 1
            else:
                action = model.predict(obs, deterministic=True)[0]

            obs, reward, done, info = env.step(action)
            if done:
                print("Episode finished due to:", info.get('terminated_due_to_fall', 'unknown reason'))
                print(obs)
            frame = env.render(mode = "rgb_array")  # This returns an RGB array
            frames.append(frame)
            accu_reward += reward
        # Save video
        print(f"Total reward: {accu_reward}")
        video_path = f"{args.video_save_path}/{args.env}"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        task_str = "_".join(map(str, np.array(args.task).flatten()))
        imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=30)

    elif args.mode == "train":
        print(par_env.get_task())
        model.learn(
                    total_timesteps=1200000,
                    log_interval=10,
                    )
        model.save("./best_model/best_model_heavy.zip")
        accu_reward = 0
        frames = []
        env = gym.make(args.env)
        obs = env.reset()
        done = False
        while not done:
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, info = env.step(action)
            frame = env.render(mode = "rgb_array")  # This returns an RGB array
            frames.append(frame)
            accu_reward += reward
        # Save video
        print(f"Total reward: {accu_reward}")
        video_path = f"{args.video_save_path}/{args.env}"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        task_str = "_".join(map(str, np.array(args.task).flatten()))
        imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=30)
    elif args.mode == "data":

        # env = DataCollector(env)
        episodes = []
        while len(episodes) < args.data_collect_episode:
            obss = np.empty((args.now,0, par_env.observation_space.shape[0]))
            actions = np.empty((args.now,0, par_env.action_space.shape[0]))
            rewards = np.empty((args.now,0))

            done = np.zeros(args.now, dtype=bool)
            obs = par_env.reset()
            random_action = 0
            for _ in range(16):
                action = np.random.uniform(low = -1,high = 1,size = (args.now,par_env.action_space.shape[0]))
                obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
                obs,reward,_,_ = par_env.step(action)
                actions = np.append(actions,action[:,np.newaxis,:], axis = 1)
                rewards = np.append(rewards, np.zeros(args.now)[:, np.newaxis], axis=1)
                
            for _ in range(500):
                
                # if np.random.rand() < 0.02 and random_action == 0:
                #     random_action = 4
                # if random_action > 0:
                #     action = np.random.uniform(low = -1,high = 1,size = (args.now,par_env.action_space.shape[0]))
                #     random_action -= 1
                # else:
                action = model.predict(obs, deterministic=True)[0]
                # action = model.predict(obs, deterministic=True)[0]
                obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
                actions = np.append(actions, action[:, np.newaxis, :], axis=1)
                obs, reward, done, info = par_env.step(action)
                rewards = np.append(rewards, reward[:, np.newaxis], axis=1)
            accu_rewards = rewards.sum(axis = 1)
            print(accu_rewards)
            print(accu_rewards.mean())

            print(accu_rewards.std())
            for j in range(args.now):

                terminations = np.zeros(obss.shape[1])
                truncations = np.zeros(obss.shape[1])

                truncations[-1] = 1
                infos = {}

                episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
                                                observations=obss[j],
                                                actions=actions[j],
                                                rewards=rewards[j],
                                                terminations = terminations,
                                                truncations = truncations,
                                                infos = infos))
            print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
        dataset = minari.create_dataset_from_buffers(f'heavy_{args.env}',
                                                     buffer = episodes,
                                                     observation_space=gymnasium.spaces.Box(low=-10, high=10, shape=(17,)),
                                                     action_space=gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,)))


