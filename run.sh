#!/bin/bash
SESSION=ddap



tmux send-keys -t $SESSION:0.0 "conda activate dv; python veteran_random_mujoco.py \\
--device cuda:1 \\
--policy_diffusion_gradient_steps 1000000 \\
--planner_diffusion_gradient_steps 2 \\
--critic_gradient_steps  2" C-m

tmux send-keys -t $SESSION:0.1 "conda activate dv; python veteran_random_mujoco.py \\
--device cuda:2 \\
--policy_diffusion_gradient_steps 2 \\
--planner_diffusion_gradient_steps 1000000 \\
--critic_gradient_steps  2" C-m

# tmux send-keys -t $SESSION:0.2 "conda activate dv; python veteran_random_mujoco.py \\
# --device cuda:3 \\
# --policy_diffusion_gradient_steps 2 \\
# --planner_diffusion_gradient_steps 2 \\
# --critic_gradient_steps  1000000" C-m

tmux attach -t $SESSION