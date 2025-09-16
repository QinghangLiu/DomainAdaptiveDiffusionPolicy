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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer
class EpisodeLengthCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Check if an episode finished
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_lengths.append(info["episode"]["l"])
        return True
    
if __name__ == "__main__":
    wandb.init(project="your_project_name",mode = "disabled")

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo',           default="sac", type=str, help='Algorithm to use')
    parser.add_argument('--env',            default="RandomWalker2d-v0", type=str, help='Train gym env')
    parser.add_argument('--lr',             default=None, type=float, help='Learning rate')
    parser.add_argument('--gamma',          default=0.99, type=float, help='gamma discount factor')
    parser.add_argument('--now',            default=20, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--gradient_steps', default=4, type=int, help='Number of gradient steps per update')
    parser.add_argument('--model_path',     default="./best_model/", help='Train from scratch')
    parser.add_argument('--mode',           default="data", help='Evaluate the policy')
    parser.add_argument('--task',           default=np.array([8,7,7,7,7,7,7,0.9,0.78,0.2,0.9,2.4,0.8]), help='Use wandb for logging')
    parser.add_argument('--video_save_path',default="./video", help='Path to save video')
    parser.add_argument('--data_collect_episode',            default=300, type=int, help='Number of cpus for parallelization')
    parser.add_argument('--gpu',           default=0, type=int, help='Random seed')
    args = parser.parse_args()
    env = gym.make(args.env)
    default_task = env.get_task()
    bound = np.zeros((env.dyn_ind_to_name.__len__(),2))
    for i in range(env.dyn_ind_to_name.__len__()):
        bound[i,:] = np.array(env.get_search_bounds_mean(i))
    series_task = np.linspace(bound[:,0],bound[:,1], 5)

    model_save_path = f"./best_model/{args.env}/"
    # series_task = default_task.copy()

  

    # series_task = np.vstack([series_task,np.linspace(default_task,args.task, 10)])



    # print(env.get_task())
    # env.set_task(*task)
    par_env = make_vec_env(args.env, n_envs=args.now,  seed=None, vec_env_cls=RandomSubprocVecEnv)

    # par_env.set_task(np.tile(task, (args.now,1)))
    eff_lr = get_learning_rate(args, par_env)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model = SAC("MlpPolicy", par_env, learning_rate=eff_lr, gamma=args.gamma, gradient_steps=args.gradient_steps, verbose=1, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # if os.path.exists(args.model_path):
    #     # agent = Policy(algo=args.algo,
    #     #         env=par_env,
    #     #         lr=eff_lr,
    #     #         gamma=args.gamma,
    #     #         load_from_pathname=args.model_path,
    #     #         gradient_steps= args.gradient_steps,
    #     #         device="cuda"if  torch.cuda.is_available() else "cpu",     
    #     #                     )
    #     model = SAC.load(args.model_path, env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # else:
    #     # agent = Policy(algo=args.algo,
    #     #         env=par_env,
    #     #         lr=eff_lr,
    #     #         gamma=args.gamma,
    #     #         gradient_steps= args.gradient_steps,
    #     #         device="cuda"if  torch.cuda.is_available() else "cpu",)     
    #     model = SAC("MlpPolicy", par_env, learning_rate=eff_lr, gamma=args.gamma, gradient_steps=args.gradient_steps, verbose=1, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")



    if args.mode == "eval":
        
        model = SAC.load(f"./best_model/best_model_{args.env}_{np.zeros((len(default_task),))}.zip", env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

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
        accu_reward = 0.01
        cnt_accureward = 0
        while not (0 < (cnt_accureward - accu_reward)/accu_reward < 1e-2):
            accu_reward = cnt_accureward
            print(par_env.get_task())
            model.learn(
                        total_timesteps=400000,
                        log_interval=10,
                        )
            # model.save("./best_model/best_model_heavy.zip")
            cnt_accureward = 0
            frames = []
            env = gym.make(args.env)
            obs = env.reset()
            done = False
            while not done:
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = env.step(action)
                cnt_accureward += reward
        frame = env.render(mode = "rgb_array")  # This returns an RGB array
        frames.append(frame)
        # Save video
        print(f"Total reward: {accu_reward}")
        video_path = f"{args.video_save_path}/{args.env}"
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        task_str = "_".join(map(str, np.array(args.task).flatten()))
        imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames, fps=30)
    elif args.mode == "data":
        model = SAC.load(f"./best_model/best_model_{args.env}_{default_task[:6]}.zip", env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        # env = DataCollector(env)
        episodes = []
        for _ in range(args.data_collect_episode // args.now):
            obss = np.empty((args.now,0, par_env.observation_space.shape[0]))
            actions = np.empty((args.now,0, par_env.action_space.shape[0]))
            rewards = np.empty((args.now,0))
            terminations = np.empty((args.now,0))
            cum_done = 0
            accu_rewards = np.zeros((args.now,))
            obs = par_env.reset()
            # random_action = 0
            for _ in range(16):
                action = np.zeros((args.now,par_env.action_space.shape[0]))
                obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
                # obs,reward,done,_ = par_env.step(action)
                actions = np.append(actions,action[:,np.newaxis,:], axis = 1)
                rewards = np.append(rewards, np.zeros(args.now)[:, np.newaxis], axis=1)
                terminations = np.append(terminations, np.zeros(args.now)[:, np.newaxis], axis=1)
            for _ in range(1000):
                
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
                terminations = np.append(terminations,done[:, np.newaxis], axis=1)
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                accu_rewards += reward * (1 - cum_done)

            print(accu_rewards)
            print(accu_rewards.mean())

            print(accu_rewards.std())
            for j in range(args.now):

                
                truncations = np.zeros(obss.shape[1])

                truncations[-1] = 1
                infos = {}

                episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
                                                observations=obss[j],
                                                actions=actions[j],
                                                rewards=rewards[j],
                                                terminations = terminations[j],
                                                truncations = truncations,
                                                infos = infos))
            print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
        dataset = minari.create_dataset_from_buffers(f'defaultdyna_{args.env}',
                                                     buffer = episodes,
                                                     observation_space=gymnasium.spaces.Box(low=-10, high=10, shape=(17,)),
                                                     action_space=gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,)))
    elif args.mode == "train_and_data":
        episodes = []
        task_num = 1
        trained_task = []
        
        for i in range(task_num):
            
            if not os.exists(f"./best_model/best_model_{args.env}_{np.zeros((len(default_task),))}.zip"):
                task = default_task
                task_index = np.zeros((len(default_task),))
            else:
                task = series_task[task_index,np.arange(len(default_task))]
                task_index = np.random.choice(np.arange(1,series_task.shape[0]-1), size=len(default_task), replace=True)
                
            trained_task.append(task)

            

            par_env.set_task(np.tile(task, (args.now,1)))
            print(f"Training on task: {task}")
            model.learn(
                            total_timesteps=200000,
                            log_interval=10,

                            )
            max_reward = 0.01

            cnt_accureward = 0
            obs = par_env.reset()
            dones = np.zeros(args.now, dtype=bool)
            while not np.all(dones):
                action = model.predict(obs, deterministic=True)[0]
                obs, reward, done, info = par_env.step(action)
                dones = np.logical_or(dones, done)
                cnt_accureward += ((1 - dones) * reward).mean()


            while not (0 < (cnt_accureward - max_reward)/max_reward < 1e-2):
                max_reward = max(max_reward,cnt_accureward)
                model.learn(
                            total_timesteps=100000,
                            log_interval=10,

                            )

                cnt_accureward = 0
                obs = par_env.reset()
                dones = np.zeros(args.now, dtype=bool)

                while not np.all(dones):
                    action = model.predict(obs, deterministic=True)[0]
                    obs, reward, done, info = par_env.step(action)
                    dones = np.logical_or(dones, done)
                    cnt_accureward += ((1 - dones) * reward).mean()

                

                
                print(cnt_accureward)


                model.save(f"./best_model/best_model_{args.env}_{str(task_index)}.zip")
            # clear the replay buffer
            model = SAC.load(f"./best_model/best_model_{args.env}_{str(task_index)}.zip", env=par_env, device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            print(f"Collecting data on task: {task}")
            for _ in range(args.data_collect_episode // args.now):
                obss = np.empty((args.now,0, par_env.observation_space.shape[0]))
                actions = np.empty((args.now,0, par_env.action_space.shape[0]))
                rewards = np.empty((args.now,0))
                terminations = np.empty((args.now,0))

                obs = par_env.reset()
                # random_action = 0
                for _ in range(16):
                    action = np.random.uniform(low = -1,high = 1,size = (args.now,par_env.action_space.shape[0]))
                    obss = np.append(obss, obs[:, np.newaxis, :], axis=1)
                    obs,reward,done,_ = par_env.step(action)
                    actions = np.append(actions,action[:,np.newaxis,:], axis = 1)
                    rewards = np.append(rewards, np.zeros(args.now)[:, np.newaxis], axis=1)
                    terminations = np.append(terminations, done[:, np.newaxis], axis=1)
                for _ in range(1000):
                    
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
                    terminations = np.append(terminations,done[:, np.newaxis], axis=1)
                accu_rewards = rewards.sum(axis = 1)
                print(accu_rewards)
                print(accu_rewards.mean())

                print(accu_rewards.std())
                for j in range(args.now):

                    
                    truncations = np.zeros(obss.shape[1])

                    truncations[-1] = 1
                    infos = {}

                    episodes.append(minari.data_collector.EpisodeBuffer(len(episodes),
                                                    observations=obss[j],
                                                    actions=actions[j],
                                                    rewards=rewards[j],
                                                    terminations = terminations[j],
                                                    truncations = truncations,
                                                    infos = infos))
                print(f"Episode {len(episodes)} collected with {obss[j].shape[0]} timesteps")
                np.save(f"./best_model/trained_tasks_{args.env}_{timestamp}.npy", np.array(trained_task))
        dataset = minari.create_dataset_from_buffers(f'defaultDyna_{args.env}',
                                                     buffer = episodes,
                                                     observation_space=gymnasium.spaces.Box(low=-10, high=10, shape=(17,)),
                                                     action_space=gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(6,)))
        model.save(args.model_path)
        #save the trained tasks
        



