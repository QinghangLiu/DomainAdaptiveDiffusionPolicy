import os

import d4rl
import gym
import argparse
import hydra, wandb, uuid
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dr_envs
import minari
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.classifier import CumRewClassifier
from cleandiffuser.dataset.d4rl_mujoco_dataset import  D4RLMuJoCoTDDataset, RandomMuJoCoSeqDataset
from customwrappers.RandomVecEnv import RandomSubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from cleandiffuser.dataset.dataset_utils import loop_dataloader, loop_two_dataloaders
from cleandiffuser.diffusion import ContinuousDiffusionSDE, DiscreteDiffusionSDE
from cleandiffuser.invdynamic import MlpInvDynamic
from cleandiffuser.nn_condition import MLPCondition, IdentityCondition
from cleandiffuser.nn_diffusion import DiT1d, DVInvMlp
from cleandiffuser.nn_diffusion.dvinvdit import DVInvDiT
from cleandiffuser.nn_classifier import HalfJannerUNet1d
from cleandiffuser.nn_diffusion import JannerUNet1d
from cleandiffuser.utils import report_parameters, DD_RETURN_SCALE, DVHorizonCritic, IDQLVNet
from pipelines.utils import set_seed
from tqdm import tqdm
from omegaconf import OmegaConf
import time 
import imageio

def pipeline(args):
    args.device = args.device if torch.cuda.is_available() else "cpu"
    args_dict = vars(args).copy()
    if "task" in args_dict:
        args_dict["task"] = vars(args.task)
    if args.enable_wandb and args.mode in ["inference", "train"]:
        wandb.require("core")
        print(args)
        wandb.init(
            reinit=True,
            id=str(uuid.uuid4()),
            project=str(args.project),
            group=str(args.group),
            name=str(args.name),
            config=OmegaConf.to_container(OmegaConf.create(args_dict), resolve=True)
        )

    set_seed(args.seed)
    
    # base config
    base_path = f"{args.pipeline_name}_H{args.task.planner_horizon}_Jump{args.task.stride}_History{args.task.history}"
    base_path += f"_next{args.planner_next_obs_loss_weight}"
    # guidance type
    base_path += f"_{args.guidance_type}"
    # For Planner
    base_path += f"_{args.planner_net}"
    if args.planner_net == "transformer":
        base_path += f"_d{args.planner_depth}"
        base_path += f"_width{args.planner_d_model}"
    elif args.planner_net == "unet":
        base_path += f"_width{args.unet_dim}"
    
    if not args.planner_predict_noise:
        base_path += f"_pred_x0"
    
    # pipeline_type
    base_path += f"_{args.pipeline_type}"
    base_path += f"_dp{args.use_diffusion_invdyn}"
    base_path += f"_penalty{args.terminal_penalty}"
    base_path += f"_bonus{args.full_traj_bonus}"
    base_path += f"_gamma{args.discount}"
    base_path += f"_adv{args.use_weighted_regression}"
    base_path += f"_weight{args.weight_factor}_guide{args.planner_guide_noise_scale}"
    # task name
    base_path += f"/{args.task.env_name}/{args.task.dataset}/"
    
    
    save_path = f"{args.save_dir}/" + base_path
    video_path = "video_outputs/" + base_path
    
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    if os.path.exists(video_path) is False:
        os.makedirs(video_path)
    dataset = minari.load_dataset(f"{args.task.dataset}")
    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    planner_dataset = RandomMuJoCoSeqDataset(
        dataset, horizon=args.task.planner_horizon, discount=args.discount, 
        stride=args.task.stride, center_mapping=(args.guidance_type!="cfg"),
        terminal_penalty=args.terminal_penalty,max_path_length=args.task.max_path_length,
        full_traj_bonus=args.full_traj_bonus
    )
    policy_dataset = RandomMuJoCoSeqDataset(
        dataset, horizon=args.task.planner_horizon, discount=args.discount, 
        stride=args.task.stride, center_mapping=(args.guidance_type!="cfg"),
        terminal_penalty=args.terminal_penalty,max_path_length=args.task.max_path_length,
        full_traj_bonus=args.full_traj_bonus
    )
    planner_dataloader = DataLoader(
        planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim
    
    policy_dataloader = DataLoader(
        policy_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = planner_dataset.o_dim, planner_dataset.a_dim

    planner_dim = obs_dim + act_dim

    # --------------- Network Architecture -----------------
    if args.planner_net == "transformer":
        nn_diffusion_planner = DiT1d(
            planner_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//32, depth=args.planner_depth, timestep_emb_type="fourier", dropout=0.1)
    elif args.planner_net == "unet":
        nn_diffusion_planner = JannerUNet1d(
            planner_dim, model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", attention=False, kernel_size=5)
    
    nn_condition_planner = MLPCondition(
                in_dim=obs_dim, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim, ], act=nn.SiLU(), dropout=0)
    classifier = None
        
    if args.guidance_type == "MCSS":
        # --------------- Horizon Critic -----------------
        if args.pipeline_type=="separate":
            critic_input_dim = obs_dim
        else:
            critic_input_dim = obs_dim + act_dim
        critic = DVHorizonCritic(
            critic_input_dim, emb_dim=args.planner_emb_dim,
            d_model=args.planner_d_model, n_heads=args.planner_d_model//32, depth=2, norm_type="pre").to(args.device)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)
        print(f"=============== Parameter Report of Value ====================================")
        report_parameters(critic)
        print(f"==============================================================================")
        
    elif args.guidance_type=="cfg":
        if args.planner_net == "transformer":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim, ], act=nn.SiLU(), dropout=0.25)
        elif args.planner_net == "unet":
            nn_condition_planner = MLPCondition(
                in_dim=1, out_dim=args.unet_dim, hidden_dims=[args.unet_dim, ], act=nn.SiLU(), dropout=0.25)
    
    elif args.guidance_type=="cg":
        nn_classifier = HalfJannerUNet1d(
            args.task.planner_horizon, planner_dim, out_dim=1,
            model_dim=args.unet_dim, emb_dim=args.unet_dim,
            timestep_emb_type="positional", kernel_size=3)
        classifier = CumRewClassifier(nn_classifier, device=args.device)
        print(f"=============== Parameter Report of Classifier ===============================")
        report_parameters(nn_classifier)
        print(f"==============================================================================")

    print(f"=============== Parameter Report of Planner ==================================")
    report_parameters(nn_diffusion_planner)
    print(f"==============================================================================")

    # ----------------- Masking -------------------
    if args.pipeline_type=="joint":
        fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
        fix_mask[:args.task.history, :] = 1.
        fix_mask[args.task.history, :obs_dim] = 1.
    elif args.pipeline_type=="separate":
        fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
        fix_mask[:args.task.history+1, :] = 1.
        fix_mask[args.task.history+1:, obs_dim:] = 1. # only predict future states

    loss_weight = torch.ones((args.task.planner_horizon, planner_dim))
    loss_weight[1] = args.planner_next_obs_loss_weight

    # --------------- Diffusion Model with Classifier-Free Guidance --------------------
    planner = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
        device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear",task_num = 40,guide_noise_scale=args.planner_guide_noise_scale)
    fix_mask = torch.zeros((args.task.planner_horizon, planner_dim))
    fix_mask[:args.task.history, :] = 1.
    fix_mask[args.task.history, :obs_dim] = 1.
    planner2 = ContinuousDiffusionSDE(
        nn_diffusion_planner, nn_condition=nn_condition_planner,
        fix_mask=fix_mask, loss_weight=loss_weight, classifier=classifier, ema_rate=args.planner_ema_rate,
        device=args.device, predict_noise=args.planner_predict_noise, noise_schedule="linear")
    # --------------- Inverse Dynamic (Policy) -------------------
    if args.pipeline_type=="separate":
        if args.use_diffusion_invdyn:
            if args.policy_net == "mlp":
                nn_diffusion_invdyn = DVInvMlp(obs_dim, act_dim, emb_dim=64, hidden_dim=args.policy_hidden_dim, timestep_emb_type="positional").to(args.device)
                nn_condition_invdyn = IdentityCondition(dropout=0.0).to(args.device)
                print(f"=============== Parameter Report of Policy ===================================")
                report_parameters(nn_diffusion_invdyn)
                print(f"==============================================================================")
                # --------------- Diffusion Model Actor --------------------
                policy = DiscreteDiffusionSDE(
                    nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, optim_params={"lr": args.policy_learning_rate},
                    x_max=+1. * torch.ones((1, act_dim), device=args.device),
                    x_min=-1. * torch.ones((1, act_dim), device=args.device),
                    diffusion_steps=args.policy_diffusion_steps, ema_rate=args.policy_ema_rate, device=args.device)
            elif args.policy_net == "transformer":
                nn_diffusion_invdyn = DVInvDiT(obs_dim+act_dim, act_dim, emb_dim=args.planner_emb_dim,  
                                            d_model=args.planner_d_model, n_heads=args.planner_d_model//32, depth=args.planner_depth, timestep_emb_type="fourier",dropout = 0.1).to(args.device)
                nn_condition_invdyn = MLPCondition(
                    in_dim=2*obs_dim, out_dim=args.planner_emb_dim, hidden_dims=[args.planner_emb_dim, ], act=nn.SiLU(), dropout=0)
                print(f"=============== Parameter Report of Policy ===================================")
                report_parameters(nn_diffusion_invdyn)
                print(f"==============================================================================")
                fix_mask = torch.zeros((args.task.history+1, planner_dim))
                fix_mask[:args.task.history, :] = 1.
                fix_mask[args.task.history, :obs_dim] = 1.
                # --------------- Diffusion Model Actor --------------------
                policy = ContinuousDiffusionSDE(
                    nn_diffusion_invdyn, nn_condition_invdyn, predict_noise=args.policy_predict_noise, noise_schedule="linear",

                     ema_rate=args.policy_ema_rate, device=args.device,fix_mask=fix_mask,task_num =  40,guide_noise_scale=args.planner_guide_noise_scale)
        else:
            invdyn = MlpInvDynamic(obs_dim, act_dim, 512, nn.Tanh(), {"lr": 2e-4}, device=args.device)

    # ---------------------- Training ----------------------
    if args.mode == "train":
        # Planner
        planner_lr_scheduler = CosineAnnealingLR(planner.optimizer, args.planner_diffusion_gradient_steps)
        if os.path.exists(save_path + f"planner_ckpt_latest.pt"):
            planner.load(save_path + f"planner_ckpt_latest.pt")
            print(f"Load planner from {save_path + f'planner_ckpt_latest.pt'}")
        planner.train()
        
        # Critic or classifier
        if args.guidance_type=="MCSS":
            critic_lr_scheduler = CosineAnnealingLR(critic_optim, args.planner_diffusion_gradient_steps)
            if os.path.exists(save_path + f"critic_ckpt_latest.pt"):
                critic_ckpt = torch.load(save_path + f"critic_ckpt_latest.pt")
                critic.load_state_dict(critic_ckpt["critic"])
                print(f"Load critic from {save_path + f'critic_ckpt_latest.pt'}")
            critic.train()
        elif args.guidance_type=="cg":
            classifier_lr_scheduler = CosineAnnealingLR(planner.classifier.optim, args.planner_diffusion_gradient_steps)
            classifier.train()
        
        # Policy
        if args.pipeline_type=="separate":

                
            if args.use_diffusion_invdyn:
                policy_lr_scheduler = CosineAnnealingLR(policy.optimizer, args.policy_diffusion_gradient_steps)
                if os.path.exists(save_path + f"policy_ckpt_latest.pt"):
                    policy.load(save_path + f"policy_ckpt_latest.pt")
                    print(f"Load policy from {save_path + f'policy_ckpt_latest.pt'}")
                policy.train()
            else:
                invdyn_lr_scheduler = CosineAnnealingLR(invdyn.optim, args.invdyn_gradient_steps)
                if os.path.exists(save_path + f"invdyn_ckpt_latest.pt"):
                    invdyn.load(save_path + f"invdyn_ckpt_latest.pt")
                    print(f"Load inverse dynamics from {save_path + f'invdyn_ckpt_latest.pt'}")
                invdyn.train()

        n_gradient_step = 0
        log = {
            "val_pred": 0,
            "val_loss": 0,
            "avg_loss_planner": 0, 
            "bc_loss_policy": 0,
            "avg_loss_classifier": 0,


        }
        
        pbar = tqdm(total=max(args.planner_diffusion_gradient_steps, args.policy_diffusion_gradient_steps,args.critic_gradient_steps)/args.log_interval)
        for planner_batch, policy_batch in loop_two_dataloaders(planner_dataloader, policy_dataloader):

            planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
            planner_horizon_action = planner_batch["act"].to(args.device)
            planner_horizon_obs_action = torch.cat([planner_horizon_obs, planner_horizon_action], -1)
            
            planner_horizon_data = planner_horizon_obs_action
            if args.pipeline_type == "separate":
                planner_horizon_data[:,args.task.history:,obs_dim:] = 0
            planner_td_val = planner_batch["val"].to(args.device)
            planner_task = planner_batch["task_id"].to(args.device) if "task_id" in planner_batch else None
            planner_horizon_rew = planner_batch["rew"].to(args.device)
            policy_horizon_obs = policy_batch["obs"]["state"].to(args.device)
            policy_horizon_action = policy_batch["act"].to(args.device)
            if args.policy_net == "mlp":

                policy_td_obs, policy_td_next_obs, policy_td_act = policy_horizon_obs[:,args.task.history,:], policy_horizon_obs[:,args.task.history+1,:], policy_horizon_action[:,args.task.history,:]
            else:
                policy_horizon_obs_action = torch.cat([policy_horizon_obs[:,:args.task.history+1,:], policy_horizon_action[:,:args.task.history+1,:]], -1)
                policy_td_obs, policy_td_next_obs, policy_td_act = policy_horizon_obs[:,args.task.history,:], policy_horizon_obs[:,args.task.history+1,:], policy_horizon_action[:,args.task.history,:]

            # ----------- Planner Gradient Step ------------
            if n_gradient_step <= args.planner_diffusion_gradient_steps:
                if args.guidance_type == "cfg":
                    log["avg_loss_planner"] += planner.update(planner_horizon_data, planner_td_val)['loss']
                else:
                    if args.use_weighted_regression:
                        weighted_regression_tensor = torch.exp( (planner_td_val - 1) * args.weight_factor)
                        log["avg_loss_planner"] += planner.update(planner_horizon_data, 
                                                                  weighted_regression_tensor=weighted_regression_tensor,
                                                                  condition = planner_horizon_obs[:,args.task.history,:],task_id = planner_task)['loss']
                    else:
                        log["avg_loss_planner"] += planner.update(planner_horizon_data)['loss']
                planner_lr_scheduler.step()
            
            if args.guidance_type=="MCSS":
                # ----------- Horizon Critic Gradient Step ------------    
                if n_gradient_step <= args.critic_gradient_steps:
                    if args.pipeline_type=="separate":
                        planner_horizon_data_critic = planner_horizon_obs
                        planner_horizon_data_critic_0 = planner_horizon_obs
                    else:  
                        planner_horizon_data_critic = planner_horizon_obs_action
                    planner_td_val = planner_td_val
                    # rew = planner_horizon_rew[:,args.task.history]
                    val_pred = critic(planner_horizon_data_critic)
                    # val_pred_h_0 = critic(planner_horizon_data_critic_0)
                    # val_pred = rew + args.discount * val_pred_h_1
                    assert val_pred.shape == planner_td_val.shape
                    critic_loss = F.mse_loss(val_pred, planner_td_val)
                    # critic_loss = F.mse_loss(val_pred, val_pred_h_0)
                    log["val_pred"] += val_pred.mean().item()
                    log["val_loss"] += critic_loss.item()
                    critic_optim.zero_grad()
                    critic_loss.backward()
                    critic_optim.step()
                    critic_lr_scheduler.step()
      
            elif args.guidance_type=="cg":
                if n_gradient_step <= args.planner_diffusion_gradient_steps:
                    log["avg_loss_classifier"] += planner.update_classifier(planner_horizon_data, planner_td_val)['loss']
                    classifier_lr_scheduler.step()
            
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    # ----------- Policy Gradient Step ------------
                    if n_gradient_step <= args.policy_diffusion_gradient_steps:
                        if args.policy_net == "mlp":
                            log["bc_loss_policy"] += policy.update(policy_td_act, torch.cat([policy_td_obs, policy_td_next_obs], dim=-1))['loss']
                        elif args.policy_net == "transformer":
                            log["bc_loss_policy"] += policy.update(policy_horizon_obs_action, torch.cat([policy_td_obs, policy_td_next_obs], dim=-1),task_id = planner_task)['loss']
                        policy_lr_scheduler.step()
                else:    
                    if n_gradient_step <= args.invdyn_gradient_steps:
                        log["bc_loss_policy"] += invdyn.update(policy_td_obs, policy_td_act, policy_td_next_obs)['loss']
                        invdyn_lr_scheduler.step()

            # ----------- Logging ------------
            if (n_gradient_step + 1) % args.log_interval == 0:
                log["gradient_steps"] = n_gradient_step + 1
                log["val_pred"] /= args.log_interval
                log["val_loss"] /= args.log_interval
                log["avg_loss_planner"] /= args.log_interval
                log["bc_loss_policy"] /= args.log_interval
                log["avg_loss_classifier"] /= args.log_interval
                print(log)
                if args.enable_wandb:
                    wandb.log(log, step=n_gradient_step + 1)
                pbar.update(1)
                log = {
                    "val_pred": 0,
                    "val_loss": 0,
                    "avg_loss_planner": 0, 
                    "bc_loss_policy": 0,
                    "avg_loss_classifier": 0
                }

            # ----------- Saving ------------
            if (n_gradient_step + 1) % args.save_interval == 0:
                if n_gradient_step <= args.planner_diffusion_gradient_steps:
                    planner.save(save_path + f"planner_ckpt_{n_gradient_step + 1}.pt")
                    planner.save(save_path + f"planner_ckpt_latest.pt")
                if args.guidance_type=="MCSS":
                    if n_gradient_step <= args.critic_gradient_steps:
                        torch.save({"critic": critic.state_dict(),}, save_path + f"critic_ckpt_{n_gradient_step + 1}.pt")
                        torch.save({"critic": critic.state_dict(),}, save_path + f"critic_ckpt_latest.pt")
                elif args.guidance_type=="cg":
                    planner.classifier.save(save_path + f"classifier_ckpt_{n_gradient_step + 1}.pt")
                    planner.classifier.save(save_path + f"classifier_ckpt_latest.pt")
                
                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        if n_gradient_step <= args.policy_diffusion_gradient_steps:
                            policy.save(save_path + f"policy_ckpt_{n_gradient_step + 1}.pt")
                            policy.save(save_path + f"policy_ckpt_latest.pt")
                    else:
                        invdyn.save(save_path + f"invdyn_ckpt_{n_gradient_step + 1}.pt")
                        invdyn.save(save_path + f"invdyn_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step >= args.planner_diffusion_gradient_steps and n_gradient_step >= args.policy_diffusion_gradient_steps and n_gradient_step >= args.critic_gradient_steps:
                break

    elif args.mode == "train_expected_value":
        from copy import deepcopy
        MAX_STEPS = 1_000_000
        
        dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env))
        td_dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4, persistent_workers=True)
        obs_dim, act_dim = dataset.o_dim, dataset.a_dim
        
        EV = IDQLVNet(obs_dim, hidden_dim=256).to(args.device)
        EV_target = deepcopy(EV)
        v_optim = torch.optim.Adam(EV.parameters(), lr=3e-4)

        n_gradient_step = 0
        log = dict.fromkeys(["loss_v", "v_mean"], 0.)
        pbar = tqdm(total=MAX_STEPS/args.log_interval)
        for batch in loop_dataloader(td_dataloader):
            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            current_v = EV(obs)
            next_v = EV_target(next_obs).detach()
            target_v = (rew + (1 - tml) * args.discount * next_v).detach()
            
            v_loss = F.mse_loss(current_v, target_v)
            v_optim.zero_grad()
            v_loss.backward()
            v_optim.step()
            
            mu = 0.995
            for p, p_targ in zip(EV.parameters(), EV_target.parameters()):
                p_targ.data = mu * p_targ.data + (1 - mu) * p.data
            
            log["loss_v"] += v_loss.mean().item()
            log["v_mean"] += current_v.mean().item()

            if (n_gradient_step + 1) % args.log_interval == 0:
                log = {k: v / args.log_interval for k, v in log.items()}
                log["gradient_steps"] = n_gradient_step + 1
                print(log)
                pbar.update(1)
                log = dict.fromkeys(["loss_v", "v_mean"], 0.)

            if (n_gradient_step + 1) % args.save_interval == 0:
                torch.save({"ev": EV.state_dict()}, save_path + f"EV_ckpt_{n_gradient_step + 1}.pt")
                torch.save({"ev": EV.state_dict()}, save_path + f"EV_ckpt_latest.pt")

            n_gradient_step += 1
            if n_gradient_step > 1_000_000:
                break
    
    # ---------------------- Inference ----------------------
    elif args.mode == "inference":
        
        if args.guidance_type=="MCSS":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            # planner2.load(save_path + f"planner_ckpt_{args.planner_ckpt} copy.pt")
            planner.eval()
            # planner2.eval()
            # load critic
            critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt")
            critic.load_state_dict(critic_ckpt["critic"])
            critic.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
        
        elif args.guidance_type=="cfg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
            
        elif args.guidance_type=="cg":
            # load planner
            planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
            # load classifier
            planner.classifier.load(save_path + f"classifier_ckpt_{args.planner_ckpt}.pt")
            planner.eval()
            # load policy
            if args.pipeline_type == "separate":
                if args.use_diffusion_invdyn:
                    policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                    policy.eval()
                else:
                    invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                    invdyn.eval()
                    
        

        env_eval = make_vec_env(args.task.env_name, n_envs=args.num_envs, seed=None, vec_env_cls=RandomSubprocVecEnv)
        # env_eval.set_task(np.tile(args.domain, (args.num_envs,1)))
        print(env_eval.get_task())
        frames = []
            # env_eval = gym.vector.make(args.task.env_name, args.num_envs)

        normalizer = planner_dataset.get_normalizer()
        episode_rewards = []
        history_length = args.task.history  # should be 4
        
        for i in range(args.num_episodes):
            obs_history = []
            obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0
            for _ in range(args.task.history):
                obs_history.append(torch.concatenate(
                    (
                        torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32),
                        torch.zeros((args.num_envs,act_dim),device=args.device)
                        ),
                    dim = -1
                     ))
            while not np.all(cum_done) and t < args.task.max_path_length + 1:
                
                # 1) generate plan
                if args.guidance_type == "MCSS":
                    planner_prior = torch.zeros((args.num_envs * args.planner_num_candidates, args.task.planner_horizon, planner_dim), device=args.device)

                    for h in range(len(obs_history)):
                        history = torch.tensor(obs_history[h], device=args.device, dtype=torch.float32)
                        history_repeat = history.unsqueeze(1).repeat(1,args.planner_num_candidates, 1).view(-1, planner_dim)
                        planner_prior[:, h, :planner_dim] = history_repeat


                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs_repeat = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1).view(-1, obs_dim)

                    # sample trajectories
                    planner_prior[:, len(obs_history), :obs_dim] = obs_repeat
                    # print(planner_prior)
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs * args.planner_num_candidates, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=obs_repeat, w_cfg=1.0, temperature=args.task.planner_temperature,)
                    
                    # traj2, log2 = planner2.sample(
                    #     planner_prior, solver=args.planner_solver,
                    #     n_samples=args.num_envs * args.planner_num_candidates, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                    #     condition_cfg=obs_repeat, w_cfg=1.0, temperature=args.task.planner_temperature,)
                    
                    
                    # resample
                    with torch.no_grad():
                        if args.pipeline_type=="separate":
                            traj_critic = traj[:, args.task.history+1:, :obs_dim]
                            # traj2_critic = traj2[:, :, :obs_dim]
                        else:
                            traj_critic = traj
                        value = critic(traj_critic)
                        
                        value = value.view(args.num_envs, args.planner_num_candidates)
                        idx = torch.argmax(value, -1)
                        traj = traj.reshape(args.num_envs, args.planner_num_candidates, args.task.planner_horizon, planner_dim)
                        traj = traj[torch.arange(args.num_envs), idx]

                        # value2 = critic(traj2_critic)
                        
                        # value2 = value2.view(args.num_envs, args.planner_num_candidates)
                        # idx2 = torch.argmax(value2, -1)
                        # traj2 = traj2.reshape(args.num_envs, args.planner_num_candidates, args.task.planner_horizon, planner_dim)
                        # traj2 = traj2[torch.arange(args.num_envs), idx]
                
                elif args.guidance_type == "cfg":
                    planner_prior = torch.zeros((args.num_envs, args.task.planner_horizon, planner_dim), device=args.device)
                    condition = torch.ones((args.num_envs, 1), device=args.device) * args.task.planner_target_return
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)

                    # sample trajectories
                    planner_prior[:, 0, :obs_dim] = obs
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=condition, w_cfg=args.task.planner_w_cfg, temperature=args.task.planner_temperature)
                
                elif args.guidance_type == "cg":
                    planner_prior = torch.zeros((args.num_envs * args.planner_num_candidates, args.task.planner_horizon, planner_dim), device=args.device)
                    
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs_repeat = obs.unsqueeze(1).repeat(1, args.planner_num_candidates, 1).view(-1, obs_dim)
                    
                    planner_prior[:, 0, :obs_dim] = obs_repeat
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,
                        n_samples=args.num_envs * args.planner_num_candidates, sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        w_cg=args.task.planner_w_cfg, temperature=args.task.planner_temperature)
                    
                    # resample
                    with torch.no_grad():
                        logp = log["log_p"].view(args.num_envs, args.planner_num_candidates)
                        idx = torch.argmax(logp, -1)
                        traj = traj.reshape(args.num_envs, args.planner_num_candidates, args.task.planner_horizon, planner_dim)
                        traj = traj[torch.arange(args.num_envs), idx]

                # 2) generate action
                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        if args.policy_net == "mlp":
                            policy_prior = torch.zeros((args.num_envs, act_dim), device=args.device)
                        elif args.policy_net == "transformer":
                            policy_prior = traj[:, :args.task.history+1, :]
                        # print(policy_prior[:,args.task.history,obs_dim:])
                        with torch.no_grad():
                            next_obs_plan = traj[:, args.task.history+1, :obs_dim]
                            # next_obs_plan2 = traj2[:, args.task.history+1, :obs_dim]
                            #error
                            # error = torch.norm(next_obs_plan - next_obs_plan2, dim=-1) / (torch.norm(next_obs_plan2, dim=-1) + 1e-5)
                            # print(f'Error between two planners: {error.mean().item():.4f}')

                            obs_policy = obs.clone()
                            next_obs_policy = next_obs_plan.clone()
                           
                        
                            if args.rebase_policy:
                                next_obs_policy[:, :2] -= obs_policy[:, :2]
                                obs_policy[:, :2] = 0
                            
                            act, log = policy.sample(
                                policy_prior,
                                solver=args.policy_solver,
                                n_samples=args.num_envs,
                                sample_steps=args.policy_sampling_steps,
                                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1), w_cfg=1.0,
                                use_ema=args.policy_use_ema, temperature=args.policy_temperature)
                            if args.policy_net == "transformer":
                                # print(act[0,args.task.history,:])
                                act = act[:, args.task.history, obs_dim:]
                            # act2 = traj2[:, len(obs_history), obs_dim:]
                            # err_act = torch.norm(act - act2, dim=-1) / (torch.norm(act2, dim=-1) + 1e-5)
                            # print(f'Action difference between diffusion policy and planner: {err_act.mean().item():.4f}')
                            obs_history.append(torch.concatenate((obs,act), dim=-1))
                            act = act.cpu().numpy()
                    else:
                        # inverse dynamic
                        with torch.no_grad():
                            act = invdyn.predict(obs, traj[:, 1, :]).cpu().numpy()

                else:                                                                                                                            
                    act = traj[:, len(obs_history), obs_dim:]
                    obs_history.append(torch.concatenate((obs,act), dim=-1))
                    act = act.cpu().numpy()


                if len(obs_history) > history_length:
                    obs_history.pop(0)
                # step
                if args.plot:
                    images = env_eval.get_images()
                    frame = np.concatenate([
                    np.concatenate(images[:5], axis=1),   # first row (5 images side by side)
                    np.concatenate(images[5:], axis=1)    # second row (5 images side by side)
                ], axis=0)  
                    frames.append(frame)
                # print(obs,act)
                obs, rew, done, info = env_eval.step(act)
                


                t += 1
                cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                ep_reward += (rew * (1 - cum_done)) if t < args.task.max_path_length else rew
                # print(f'[t={t}] xy: {np.around(obs[:, :2], 2)}')
                print(f'[t={t}] rew: {np.around((rew * (1 - cum_done)), 2)}')

            episode_rewards.append(ep_reward)
        episode_rewards = np.array(episode_rewards).reshape(-1)
        mean = np.mean(episode_rewards)
        err = np.std(episode_rewards) / np.sqrt(len(episode_rewards))
        print(mean, err)

        if args.enable_wandb:
            wandb.log({'Mean Reward': mean, 'Error': err})
            wandb.finish()
        if args.plot:
        # Save video

            video_path = f"{args.video_save_path}/{args.task.env_name}"
            if not os.path.exists(video_path):
                os.makedirs(video_path)
            task_str = "_".join(map(str, np.array(args.task).flatten()))
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            frames_resized = [frames[i][::2, ::2] for i in range(0,len(frames),2)]  # downscale by 2
            imageio.mimsave(f"{video_path}/evaluation{timestamp}_task{task_str}.gif", frames_resized, fps=30)
    elif args.mode == "test":#test critic,policy,planner on dataset
        # load planner
        planner.load(save_path + f"planner_ckpt_{args.planner_ckpt}.pt")
        planner.eval()
        # load critic
        if args.guidance_type=="MCSS":
            critic_ckpt = torch.load(save_path + f"critic_ckpt_{args.critic_ckpt}.pt")
            critic.load_state_dict(critic_ckpt["critic"])
            critic.eval()
        elif args.guidance_type=="cg":
            planner.classifier.load(save_path + f"classifier_ckpt_{args.planner_ckpt}.pt")
            planner.classifier.eval()
        # load policy
        if args.pipeline_type == "separate":
            if args.use_diffusion_invdyn:
                policy.load(save_path + f"policy_ckpt_{args.policy_ckpt}.pt")
                policy.eval()
            else:
                invdyn.load(save_path + f"invdyn_ckpt_{args.invdyn_ckpt}.pt")
                invdyn.eval()
        
        eval_planner_dataloader = DataLoader(planner_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
        
        planner_loss = 0.
        policy_loss = 0.
        classifier_loss = 0.
        val_loss = 0.
        val_pred = 0.
        n_batch = 0
        #test for 1 epoch
        with torch.no_grad():
            for _,planner_batch in enumerate(eval_planner_dataloader):
                planner_horizon_obs = planner_batch["obs"]["state"].to(args.device)
                planner_horizon_action = planner_batch["act"].to(args.device)
                planner_horizon_obs_action = torch.cat([planner_horizon_obs, planner_horizon_action], -1)
                planner_horizon_data = planner_horizon_obs_action
                
                planner_td_val = planner_batch["val"].to(args.device)
                
                if args.guidance_type == "cfg":
                    eval_log = planner.evaluate(planner_horizon_data, condition=planner_horizon_obs[:,args.task.history,:], target_val=torch.ones((planner_horizon_data.shape[0],1), device=args.device)*args.task.planner_target_return)
                    planner_loss += eval_log['loss'].item()
                else:
                    planner_prior = torch.zeros((planner_horizon_data.shape[0], args.task.planner_horizon, planner_dim), device=args.device)
                    planner_prior[:, :args.task.history, :] = planner_horizon_data[:, :args.task.history, :]
                    planner_prior[:, args.task.history, :obs_dim] = planner_horizon_obs[:,args.task.history,:]#current obs
                    # planner_prior[:, args.task.history:, obs_dim:] = planner_horizon_data[:, args.task.history:, obs_dim:]
                    obs_repeat = planner_horizon_obs[:,args.task.history,:]
                    traj, log = planner.sample(
                        planner_prior, solver=args.planner_solver,n_samples=args.batch_size,
                        sample_steps=args.planner_sampling_steps, use_ema=args.planner_use_ema,
                        condition_cfg=obs_repeat, w_cfg=1.0, temperature=args.task.planner_temperature,)
                    loss = F.mse_loss(traj[:,args.task.history+1:,:obs_dim], planner_horizon_data[:,args.task.history+1:,:obs_dim])
                    planner_loss += loss.item()
                    n_batch += 1

                if args.pipeline_type == "separate":
                    if args.use_diffusion_invdyn:
                        if args.policy_net == "mlp":
                            policy_prior = torch.zeros((args.batch_size, act_dim), device=args.device)
                        elif args.policy_net == "transformer":
                            policy_prior = planner_horizon_data[:, :args.task.history+1, :]
                            policy_prior[:, args.task.history, obs_dim:] = 0.
                        # print(policy_prior[:,args.task.history,obs_dim:])
                        with torch.no_grad():
                            next_obs_plan = planner_horizon_data[:, args.task.history+1, :obs_dim]
                            # next_obs_plan2 = traj2[:, args.task.history+1, :obs_dim]
                            #error
                            # error = torch.norm(next_obs_plan - next_obs_plan2, dim=-1) / (torch.norm(next_obs_plan2, dim=-1) + 1e-5)
                            # print(f'Error between two planners: {error.mean().item():.4f}')

                            obs_policy = planner_horizon_data[:,args.task.history,:obs_dim].clone()
                            next_obs_policy = next_obs_plan.clone()
                           
                        
                            if args.rebase_policy:
                                next_obs_policy[:, :2] -= obs_policy[:, :2]
                                obs_policy[:, :2] = 0
                            
                            act, log = policy.sample(
                                policy_prior,
                                solver=args.policy_solver,
                                n_samples=args.batch_size,
                                sample_steps=args.policy_sampling_steps,
                                condition_cfg=torch.cat([obs_policy, next_obs_policy], dim=-1), w_cfg=1.0,
                                use_ema=args.policy_use_ema, temperature=args.policy_temperature)
                            if args.policy_net == "transformer":
                                # print(act[0,args.task.history,:])
                                act = act[:, args.task.history, obs_dim:]
                            # print(act)
                            # print(planner_horizon_action[:,args.task.history,:])
                            policy_loss += F.mse_loss(act, planner_horizon_action[:,args.task.history,:]).item()
                    else:
                        #inverse dynamics
                        invdyn_horizon_obs = planner_batch["obs"]["state"].to(args.device)
                        invdyn_horizon_action = planner_batch["act"].to(args.device)
                        invdyn_td_obs, invdyn_td_next_obs, invdyn_td_act = invdyn_horizon_obs[:,args.task.history,:], invdyn_horizon_obs[:,args.task.history+1,:], invdyn_horizon_action[:,args.task.history,:]
                        eval_log = invdyn.evaluate(invdyn_td_obs, invdyn_td_act, invdyn_td_next_obs)
                        policy_loss += eval_log['loss'].item()
                if n_batch == 100:
                    break
                if args.guidance_type=="MCSS":
                    if args.pipeline_type=="separate":
                        planner_horizon_data_critic = planner_horizon_obs
                    else:  
                        planner_horizon_data_critic = planner_horizon_obs_action
                    val_pred_batch = critic(planner_horizon_data_critic)
                    val_pred_batch = val_pred_batch.view(-1)
                    val_loss += F.mse_loss(val_pred_batch, planner_td_val).item()
                    val_pred += val_pred_batch.mean().item()
                elif args.guidance_type=="cg":
                    eval_log = planner.evaluate_classifier(planner_horizon_data, planner_td_val)

        print(f"Planner Loss: {planner_loss / (n_batch):.4f}")
        print(f"Policy Loss: {policy_loss / (n_batch):.4f}")
        if args.guidance_type=="MCSS":
            print(f"Critic Loss: {val_loss / (n_batch):.4f}, Val Pred: {val_pred / (n_batch):.4f}")
    else:

        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    #check dataset, env, device, pipeline_type, pipeline_name
    #rectify save path when pipeline name is changed
    # All arguments in one parser
    parser = argparse.ArgumentParser()
    # Task arguments
    parser.add_argument('--env_name', type=str, default="RandomWalker2d-v0", help='Environment name')
    parser.add_argument('--planner_horizon', type=int, default=20, help='Planner horizon')
    parser.add_argument('--history', type=int, default=16, help='History trajectory')
    parser.add_argument('--dataset', type=str, default="taggedmix40dynamics_RandomWalker2d-v0", help="dataset name")
    parser.add_argument('--stride', type=int, default=1, help='Stride for the dataset')
    parser.add_argument('--max_path_length', type=int, default=1016, help='Maximum path length')
    parser.add_argument('--planner_temperature', type=int, default=1, help='Planner temperature')
    parser.add_argument('--planner_target_return', type=int, default=1, help='Planner target return')
    parser.add_argument('--planner_w_cfg', default=1.0, type=float, help='Planner w_cfg')

    # Main arguments
    parser.add_argument('--pipeline_name', default="veteran_random_mujoco_test_deeper_network_with_noise_guidance", type=str, help='Pipeline name')
    parser.add_argument('--mode', default="train", type=str, help='Mode: train/inference/test/etc')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', default="cuda:1", type=str, help='Device to use')
    parser.add_argument('--project', default="dadp", type=str, help='Log path')
    parser.add_argument('--group', default="transformer inverse dynamics", type=str, help='Log path')
    parser.add_argument('--name', default="train with 40 dynamics with pre assigned noise", type=str, help='Log path')
    parser.add_argument('--enable_wandb', default=True, type=bool, help='Enable wandb logging')
    parser.add_argument('--save_dir', default="results", type=str, help='Directory to save results')

    # Guidance
    parser.add_argument('--guidance_type', default="MCSS", type=str, help='Guidance type: MCSS/cfg/cg')
    parser.add_argument('--planner_net', default="transformer", type=str, help='Planner network type')
    parser.add_argument('--policy_net', default="transformer", type=str, help='Policy network type')
    parser.add_argument('--pipeline_type', default="separate", type=str, help='Pipeline type: separate/joint')
    parser.add_argument('--rebase_policy', default=False, type=bool, help='Rebase policy position')

    # Environment
    parser.add_argument('--terminal_penalty', default=0, type=float, help='Terminal penalty')
    parser.add_argument('--full_traj_bonus', default=0, type=float, help='Full trajectory bonus')
    parser.add_argument('--discount', default=0.997, type=float, help='Discount factor')
    parser.add_argument('--domain', default=[7.525, 5.05 , 5.05 , 2.575, 2.575, 2.575, 2.575, 0.775, 0.775, 0.325, 0.325, 1.55 , 0.825], help='Dynamics')
    # Planner Config
    parser.add_argument('--planner_solver', default="ddim", type=str, help='Planner solver')
    parser.add_argument('--planner_emb_dim', default=256, type=int, help='Planner embedding dimension')
    parser.add_argument('--planner_d_model', default=256, type=int, help='Planner model dimension')
    parser.add_argument('--planner_depth', default=6, type=int, help='Planner depth')
    parser.add_argument('--planner_sampling_steps', default=20, type=int, help='Planner sampling steps')
    parser.add_argument('--planner_predict_noise', default=True, type=bool, help='Planner predict noise')
    parser.add_argument('--planner_next_obs_loss_weight', default=1, type=float, help='Planner next obs loss weight')
    parser.add_argument('--planner_ema_rate', default=0.9999, type=float, help='Planner EMA rate')
    parser.add_argument('--planner_guide_noise_scale', default=0.001, type=float, help='Planner learning rate')
    parser.add_argument('--unet_dim', default=32, type=int, help='UNet dimension')
    parser.add_argument('--use_weighted_regression', default=1, type=int, help='Use weighted regression')
    parser.add_argument('--weight_factor', default=2, type=int, help='Weight factor')

    # Policy Config
    parser.add_argument('--policy_solver', default="ddim", type=str, help='Policy solver')
    parser.add_argument('--policy_hidden_dim', default=256, type=int, help='Policy hidden dimension')
    parser.add_argument('--policy_diffusion_steps', default=10, type=int, help='Policy diffusion steps')
    parser.add_argument('--policy_sampling_steps', default=20, type=int, help='Policy sampling steps')
    parser.add_argument('--policy_predict_noise', default=True, type=bool, help='Policy predict noise')

    parser.add_argument('--policy_ema_rate', default=0.995, type=float, help='Policy EMA rate')
    parser.add_argument('--policy_learning_rate', default=0.0003, type=float, help='Policy learning rate')
    parser.add_argument('--critic_learning_rate', default=0.0003, type=float, help='Critic learning rate')

    # Training
    parser.add_argument('--use_diffusion_invdyn', default=1, type=int, help='Use diffusion inverse dynamics')
    parser.add_argument('--invdyn_gradient_steps', default=200000, type=int, help='Inverse dynamics gradient steps')
    parser.add_argument('--policy_diffusion_gradient_steps', default=1000000, type=int, help='Policy diffusion gradient steps')
    parser.add_argument('--planner_diffusion_gradient_steps', default=1000000, type=int, help='Planner diffusion gradient steps')
    parser.add_argument('--critic_gradient_steps', default=1, type=int, help='Critic gradient steps')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--log_interval', default=1000, type=int, help='Log interval')
    parser.add_argument('--save_interval', default=100000, type=int, help='Save interval')

    # Inference
    parser.add_argument('--num_envs', default=10, type=int, help='Number of environments')
    parser.add_argument('--num_episodes', default=1, type=int, help='Number of episodes')
    parser.add_argument('--planner_num_candidates', default=50, type=int, help='Planner number of candidates')
    parser.add_argument('--planner_ckpt', default=700000, type=int, help='Planner checkpoint')
    parser.add_argument('--critic_ckpt', default=200000, type=int, help='Critic checkpoint')
    parser.add_argument('--policy_ckpt', default=1000000, type=int, help='Policy checkpoint')
    parser.add_argument('--invdyn_ckpt', default=200000, type=int, help='Inverse dynamics checkpoint')
    parser.add_argument('--planner_use_ema', default=True, type=bool, help='Planner use EMA')
    parser.add_argument('--policy_temperature', default=0.5, type=float, help='Policy temperature')
    parser.add_argument('--policy_use_ema', default=True, type=bool, help='Policy use EMA')
    parser.add_argument('--plot', default=False, type=bool, help='Planner use true reward')
    parser.add_argument('--video_save_path', default="./video", help='Path to save video')
    # Value
    parser.add_argument('--value_type', default="ev", type=str, help='Value type')
    parser.add_argument('--value_mode', default="current", type=str, help='Value mode')
    #logger

    args = parser.parse_args()

    # Build a task namespace
    class Task:
        pass
    task = Task()
    task.env_name = args.env_name
    task.planner_horizon = args.planner_horizon
    task.history = args.history
    task.dataset = args.dataset
    task.stride = args.stride
    task.max_path_length = args.max_path_length
    task.planner_temperature = args.planner_temperature
    task.planner_target_return = args.planner_target_return
    task.planner_w_cfg = args.planner_w_cfg
    args.task = task

    pipeline(args)
