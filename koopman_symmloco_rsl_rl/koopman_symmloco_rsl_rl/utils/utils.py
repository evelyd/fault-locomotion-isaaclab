# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch

def split_and_pad_trajectories(tensor, dones):
    """ Splits trajectories at done indices. Then concatenates them and padds with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, aditional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1),trajectory_lengths_list)
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)


    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks

def unpad_trajectories(trajectories, masks):
    """ Does the inverse operation of  split_and_pad_trajectories()
    """
    # Need to transpose before and after the masking to have proper reshaping
    return trajectories.transpose(1, 0)[masks.transpose(1, 0)].view(-1, trajectories.shape[0], trajectories.shape[-1]).transpose(1, 0)

def fill_replay_buffer(algorithm_instance, env_instance, state_dim,num_initial_steps=None):
    """
    Initializes the replay buffers within the algorithm by performing dummy rollouts,
    mimicking the regular training loop's data collection process.
    This version correctly handles rsl_rl's RolloutStorage by performing full rollouts
    and clearing storage periodically.

    Args:
        algorithm_instance: PPO or similar
        env_instance: env
        state_dim: Dim of a single state observation
        num_initial_steps: The total number of environment steps to take for pre-filling
    """

    # Determine num_initial_rollouts based on num_initial_steps
    if num_initial_steps is None:
        if hasattr(algorithm_instance, 'replay_buffer') and hasattr(algorithm_instance.replay_buffer.states, 'shape'):
            required_steps_for_koopman = algorithm_instance.replay_buffer.states.shape[0]
        else:
            required_steps_for_koopman = 10000 # Fallback if buffer info not available
            print("Warning: Could not determine Koopman buffer size. Defaulting to 10000 steps.")

        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.storage.num_transitions_per_env
        if steps_per_full_rollout == 0: # Avoid division by zero if not configured
            steps_per_full_rollout = 1 # Dummy value, will lead to few rollouts
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")


        num_initial_rollouts = max(1, (required_steps_for_koopman + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"No num_initial_steps provided. Will perform {num_initial_rollouts} full rollouts to fill Koopman buffer.")
    else:
        # If num_initial_steps is provided, convert it to rollouts
        steps_per_full_rollout = env_instance.num_envs * algorithm_instance.num_steps_per_env
        if steps_per_full_rollout == 0:
            steps_per_full_rollout = 1
            print("Warning: steps_per_full_rollout is zero, likely due to num_envs or num_steps_per_env. Check config.")

        num_initial_rollouts = max(1, (num_initial_steps + steps_per_full_rollout - 1) // steps_per_full_rollout)
        print(f"num_initial_steps ({num_initial_steps}) will result in {num_initial_rollouts} full rollouts.")


    print(f"Initializing replay buffers by performing {num_initial_rollouts} full rollouts...")

    # Set algorithm's actor_critic to evaluation mode during initialization
    if hasattr(algorithm_instance.actor_critic, 'eval'):
        algorithm_instance.actor_critic.eval()

    # Reset environment to get initial observations for the very first rollout
    obs, _ = env_instance.reset()
    obs = env_instance.get_observations()
    privileged_obs = env_instance.get_privileged_observations()
    critic_obs = privileged_obs if privileged_obs is not None else obs
    obs, critic_obs = obs.to(algorithm_instance.device), critic_obs.to(algorithm_instance.device)

    # Ensure koopman_transition is initialized and clear if needed
    if not hasattr(algorithm_instance, 'koopman_transition'):
        raise AttributeError("Algorithm instance missing 'koopman_transition' attribute.")
    algorithm_instance.koopman_transition.clear()

    if hasattr(algorithm_instance, 'storage'):
        algorithm_instance.storage.observations[0].copy_(obs)
        if hasattr(algorithm_instance.storage, 'privileged_observations') and privileged_obs is not None:
             algorithm_instance.storage.privileged_observations[0].copy_(privileged_obs)
        elif hasattr(algorithm_instance.storage, 'critic_observations') and critic_obs is not None:
             algorithm_instance.storage.critic_observations[0].copy_(critic_obs)
    else:
        print("Warning: Algorithm instance does not have 'storage' attribute. Ensure your PPO handles initial observation internally.")

    for rollout_idx in range(num_initial_rollouts):
        for i in range(algorithm_instance.storage.num_transitions_per_env):
            actions = algorithm_instance.act(obs, critic_obs)
            algorithm_instance.act_koopman(obs, actions)
            obs, privileged_obs, rewards, dones, infos = env_instance.step(actions)
            critic_obs = privileged_obs if privileged_obs is not None else obs
            obs, critic_obs, rewards, dones = obs.to(algorithm_instance.device), critic_obs.to(algorithm_instance.device), rewards.to(algorithm_instance.device), dones.to(algorithm_instance.device)
            algorithm_instance.process_env_step(rewards, dones, infos)
            algorithm_instance.process_koopman_step(obs[:, -state_dim:])

        algorithm_instance.compute_returns(critic_obs.clone().detach(), actions.clone().detach())
        algorithm_instance.storage.clear()

    print("Replay buffer initialization complete.")

    # Switch back to train mode
    if hasattr(algorithm_instance.actor_critic, 'train'):
        algorithm_instance.actor_critic.train()