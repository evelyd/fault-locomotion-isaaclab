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

import time
import os
from collections import deque
import statistics
from typing import Union

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from koopman_symmloco_rsl_rl.algorithms import PPO, PPODAEOnline
from koopman_symmloco_rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticSymm
from koopman_symmloco_rsl_rl.env import VecEnv

from koopman_symmloco_rsl_rl.utils import fill_replay_buffer

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.use_wandb = train_cfg["use_wandb"]
        self.task = self.cfg["experiment_name"]
        if "online" in self.task or "rff" in self.task:
            self.koopman_cfg = train_cfg["koopman"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        if "rff" in self.task:
            num_critic_obs += self.koopman_cfg['model']['m']
        elif "dae" in self.task and "online" in self.task:
            if "latent_only" in self.task:
                num_extra_obs = self.env.num_privileged_obs - self.env.num_obs if self.env.num_privileged_obs is not None else 0
                num_critic_obs = num_extra_obs + self.koopman_cfg["robot"]["obs_state_ratio"] * self.koopman_cfg["robot"]["state_dim"]
            else:
                num_critic_obs += self.koopman_cfg['robot']['state_dim'] * self.koopman_cfg['robot']['obs_state_ratio']
        elif "dae" in self.task:
            if "push_door" in self.task:
                num_critic_obs += 129
            else:
                num_critic_obs += 141
            self.model_path = self.cfg["model_path"]
        # input(f"num_critic_obs: {num_critic_obs}, num_obs: {self.env.num_obs}, num_privileged_obs: {self.env.num_privileged_obs}, num_actions: {self.env.num_actions}")
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: Union[ActorCritic | ActorCriticSymm] = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        self.task,
                                                        koopman_cfg=self.koopman_cfg if ("online" in self.task or "rff" in self.task) else None,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        # input(f"Using algorithm: {alg_class.__name__}, Policy: {actor_critic.__class__.__name__}, Task: {self.task}")
        if "ppodaeonline" in alg_class.__name__.lower():
            self.alg = alg_class(actor_critic, self.task, self.koopman_cfg,  dt=env.dt, device=self.device, **self.alg_cfg)
        elif "ppodae" in alg_class.__name__.lower():
            self.alg = alg_class(actor_critic, self.task, model_path=self.model_path, device=self.device, **self.alg_cfg)
        elif "pporff" in alg_class.__name__.lower():
            self.alg = alg_class(actor_critic, self.task, self.koopman_cfg, device=self.device, **self.alg_cfg)
        else:
            self.alg = alg_class(actor_critic, self.task, device=self.device, **self.alg_cfg)

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Initialize the replay buffer
        if "online" in self.task or "rff" in self.task: # RFF also needs the buffer but only at the beginning to fill it
            if hasattr(self.alg, 'replay_buffer') and env.cfg.mode not in ["play", "test"]:
                fill_replay_buffer(self.alg, self.env, self.alg.state_dim) # in the buffer, states are only the state, not the full obs which has whatever history length of states

                # Perform initial update of normalizers
                batch_states_raw, batch_actions_raw, batch_next_states_raw, _, _ = self.alg.replay_buffer.sample(len(self.alg.replay_buffer), self.alg.replay_buffer.beta_initial)
                self.alg.obs_action_normalizer.update(batch_states_raw, batch_actions_raw)

                if "rff" in self.task:

                    if "koopman" in self.task:
                        # Normalize the states and actions
                        # batch_states_normed, batch_actions_normed = self.alg.obs_action_normalizer.normalize(batch_states_raw.to(self.device), batch_actions_raw.to(self.device))
                        # batch_next_states_normed = self.alg.obs_action_normalizer.normalize_states(batch_next_states_raw.to(self.device))
                        # Compute the Koopman operator
                        self.alg.koopman_estimator.compute_koopman_op(batch_states_raw.to(self.device), batch_actions_raw.to(self.device), batch_next_states_raw.to(self.device)) #TODO do this without normalization

                    # else:
                    #     # Compute the latent states from the replay buffer
                    #     batch_states_normed = self.alg.obs_action_normalizer.normalize_states(batch_states_raw.to(self.device))

                    batch_latent_states = self.alg.rff(batch_states_raw.to(self.device)) # TODO don't norm the states before rff either
                    self.alg.latent_normalizer.update(batch_latent_states)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        # Save the RFF since they are static and already initialized
        if "rff" in self.task:
            self.save_rff(os.path.join(self.log_dir, 'rff.pt'))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            self.alg.actor_critic.train() # switch to train mode (for dropout for example)
            # Set DAE to train mode also
            if "online" in self.task:
                self.alg.dae_model.train()
            start = time.time()

            # Initialize lists to collect NEW data for online algorithms
            new_states_this_iter = []
            new_actions_this_iter = []
            new_next_states_this_iter = []

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    if "online" in self.task or "rff" in self.task:

                        # Collect data for DAE
                        current_states_for_dae = obs[:, -self.alg.state_dim:].clone()
                        current_actions_for_dae = actions.clone()

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    if "online" in self.task or "rff" in self.task:
                        # Get the next states for the DAE
                        next_states_for_dae = obs[:, -self.alg.state_dim:].clone()

                        # Add to our new data collection
                        new_states_this_iter.append(current_states_for_dae)
                        new_actions_this_iter.append(current_actions_for_dae)
                        new_next_states_this_iter.append(next_states_for_dae)

                        # Fill the PER buffer
                        self.alg.replay_buffer.insert(
                            current_states_for_dae,
                            current_actions_for_dae,
                            next_states_for_dae,
                        )

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                if "online" in self.task or "rff" in self.task:
                    # Anneal beta for Importance Sampling weights
                    current_beta = self.alg.replay_buffer.beta_initial + (1.0 - self.alg.replay_buffer.beta_initial) * \
                                min(1.0, (it - self.current_learning_iteration) / self.alg.replay_buffer.beta_annealing_steps)

                    # Concatenate the new data
                    batch_states_new = torch.cat(new_states_this_iter, dim=0)
                    batch_actions_new = torch.cat(new_actions_this_iter, dim=0)
                    batch_next_states_new = torch.cat(new_next_states_this_iter, dim=0)

                    # Perform update of normalizers using only the new data
                    self.alg.obs_action_normalizer.update(batch_states_new, batch_actions_new)

                    if "rff" in self.task:
                        batch_latent_states_new = self.alg.rff(batch_states_new)
                        self.alg.latent_normalizer.update(batch_latent_states_new)

                # Learning step
                start = stop

                if "online" in self.task or "koopman" in self.task:
                    self.alg.compute_returns(critic_obs, actions)
                else:
                    self.alg.compute_returns(critic_obs)

            if "rff_koopman" in self.task:
                # Update Koopman matrix
                koopman_computation_start_time = time.time()

                # Normalize the states and actions
                # normed_states, normed_actions = self.alg.obs_action_normalizer.normalize(batch_states_raw.to(self.device), batch_actions_raw.to(self.device))
                # next_normed_states = self.alg.obs_action_normalizer.normalize(batch_next_states_raw.to(self.device))

                # Compute the Koopman operator using only the new data
                self.alg.koopman_estimator.compute_koopman_op(batch_states_new, batch_actions_new, batch_next_states_new)

                # Compute prediction error with the updated K
                pred_error = self.alg.koopman_estimator.compute_pred_error(batch_states_new, batch_actions_new, batch_next_states_new)

                print(pred_error)

                koopman_computation_time = time.time() - koopman_computation_start_time

            if "online" in self.task:
                # Perform DAE training step
                mean_dae_loss = 0.0
                dae_train_time = 0.0
                mean_dae_obs_pred_loss = 0.0
                mean_dae_state_rec_loss = 0.0
                mean_dae_state_pred_loss = 0.0

                if len(self.alg.replay_buffer) >= self.koopman_cfg["model"]["mini_batch_size"]:
                    dae_training_start_time = time.time()

                    dae_num_mini_batches = self.koopman_cfg["model"]["num_mini_batches"]
                    dae_mini_batch_size = self.koopman_cfg["model"]["mini_batch_size"]

                    dae_losses_this_iter = []
                    dae_obs_pred_losses_this_iter = []
                    dae_state_rec_losses_this_iter = []
                    dae_state_pred_losses_this_iter = []
                    dae_ortho_losses_this_iter = []
                    dae_mag_reg_losses_this_iter = []
                    dae_kl_losses_this_iter = []
                    dae_eigval_losses_this_iter = []

                    # Iterating over mini-batches for DAE training
                    for _ in range(dae_num_mini_batches):
                        # Sample from the Prioritized Replay Buffer
                        batch_states_raw, batch_actions_raw, batch_next_states_raw, batch_tree_indices, is_weights = \
                            self.alg.replay_buffer.sample(dae_mini_batch_size, current_beta)

                        # Transfer is_weights to cuda device
                        is_weights = is_weights.to(self.device)

                        sample_generator = self.alg.replay_buffer.preprocess_samples(
                            batch_states_raw, batch_actions_raw, batch_next_states_raw,
                            frames_per_step=self.koopman_cfg["robot"]["frames_per_state"],
                            prediction_horizon=self.koopman_cfg["robot"]["pred_horizon"]
                        )

                        all_state_observations = []
                        all_action_observations = []
                        all_next_state_observations = []
                        for sample in sample_generator:
                            all_state_observations.append(sample["state_observations"].unsqueeze(0))
                            all_action_observations.append(sample["action_observations"].unsqueeze(0))
                            all_next_state_observations.append(sample["next_state_observations"].unsqueeze(0))

                        # If preprocess_samples yielded no valid samples (e.g., traj too short), skip this mini-batch
                        if not all_state_observations:
                            print("Warning: No valid samples generated by preprocess_samples, skipping DAE mini-batch.")
                            continue

                        combined_state_observations = torch.cat(all_state_observations, dim=0).to(self.device)
                        combined_action_observations = torch.cat(all_action_observations, dim=0).to(self.device)
                        combined_next_state_observations = torch.cat(all_next_state_observations, dim=0).to(self.device)

                        # Use the combined_state_observations and combined_action_observations
                        # (which are the raw, un-normalized inputs) to update the statistics.
                        self.alg.obs_action_normalizer.update(
                            batch_states_raw,
                            batch_actions_raw,
                        )

                        # Normalize the observations and actions
                        # normed_states, normed_actions = self.alg.obs_action_normalizer.normalize(
                        #     combined_state_observations, combined_action_observations, norm_mean=True
                        # )
                        # next_normed_states = self.alg.obs_action_normalizer.normalize(
                        #     combined_next_state_observations, norm_mean=True
                        # )

                        # Move preprocessed and normalized batch to the correct device for the DAE model
                        batch = self.alg.replay_buffer.shape_states_actions(
                                combined_state_observations, combined_action_observations, combined_next_state_observations #TODO no norming
                        )

                        batch_on_device = {k: v.to(self.device) for k, v in batch.items()}

                        # Forward pass through DAE
                        if hasattr(self.alg.dae_model, 'action_dim') and self.alg.dae_model.action_dim > 0:
                            outputs = self.alg.dae_model(**batch_on_device)
                        else:
                            outputs = self.alg.dae_model(**batch_on_device)

                        # Compute DAE losses
                        dae_loss_per_sample, dae_metrics = self.alg.dae_model.compute_loss_and_metrics(**outputs, **batch_on_device)

                        # Apply importance sampling weights to the loss
                        actual_batch_size_for_loss = dae_loss_per_sample.shape[0]
                        if is_weights.shape[0] != actual_batch_size_for_loss:
                            is_weights_aligned = is_weights[:actual_batch_size_for_loss]
                        else:
                            is_weights_aligned = is_weights

                        weighted_dae_loss = (dae_loss_per_sample * is_weights_aligned).mean()

                        # Backpropagate and update DAE weights
                        self.alg.dae_optimizer.zero_grad()
                        weighted_dae_loss.backward()
                        self.alg.dae_optimizer.step()

                        # Update priorities in the replay buffer
                        # Use the per-sample losses as errors
                        dae_errors_for_priority_update = dae_loss_per_sample.detach().cpu().numpy()

                        # Ensure that the batch_tree_indices also aligns with the number of samples that actually generated a loss.
                        if len(batch_tree_indices) != actual_batch_size_for_loss:
                            batch_tree_indices_aligned = batch_tree_indices[:actual_batch_size_for_loss]
                        else:
                            batch_tree_indices_aligned = batch_tree_indices

                        # Prioritize samples based on the overall DAE loss
                        self.alg.replay_buffer.update_priorities(
                            batch_tree_indices_aligned,
                            dae_errors_for_priority_update
                        )

                        dae_losses_this_iter.append(weighted_dae_loss.item())
                        dae_obs_pred_losses_this_iter.append(dae_metrics["obs_pred_loss"].item())
                        dae_state_rec_losses_this_iter.append(dae_metrics["state_rec_loss"].item())
                        dae_state_pred_losses_this_iter.append(dae_metrics["state_pred_loss"].item())

                    if dae_losses_this_iter:
                        mean_dae_loss = sum(dae_losses_this_iter) / len(dae_losses_this_iter)
                        mean_dae_obs_pred_loss = sum(dae_obs_pred_losses_this_iter) / len(dae_obs_pred_losses_this_iter)
                        mean_dae_state_rec_loss = sum(dae_state_rec_losses_this_iter) / len(dae_state_rec_losses_this_iter)
                        mean_dae_state_pred_loss = sum(dae_state_pred_losses_this_iter) / len(dae_state_pred_losses_this_iter)
                    else:
                        mean_dae_loss = 0.0 # No batches trained
                        mean_dae_obs_pred_loss = 0.0
                        mean_dae_state_rec_loss = 0.0
                        mean_dae_state_pred_loss = 0.0

                    dae_train_time = time.time() - dae_training_start_time

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                locs = locals()
                if "online" in self.task:
                    locs["mean_dae_loss"] = mean_dae_loss
                    locs["mean_dae_obs_pred_loss"] = mean_dae_obs_pred_loss
                    locs["mean_dae_state_rec_loss"] = mean_dae_state_rec_loss
                    locs["mean_dae_state_pred_loss"] = mean_dae_state_pred_loss
                    locs["dae_train_time"] = dae_train_time

                if "rff_koopman" in self.task:


                    # Get the state and controllability matrices
                    a_matrix = self.alg.koopman_estimator.K_matrix[:, :self.alg.koopman_estimator.feature_dim].detach()
                    b_matrix = self.alg.koopman_estimator.K_matrix[:, self.alg.koopman_estimator.feature_dim:].detach()

                    state_dim = a_matrix.shape[0]
                    action_dim = b_matrix.shape[0]

                    # Eigenvalues of A
                    eigvals = torch.linalg.eigvals(a_matrix)
                    max_eigval = torch.max(torch.abs(eigvals))
                    min_eigval = torch.min(torch.abs(eigvals))

                    # Controllability matrix and singular values
                    controllability_matrix = b_matrix
                    term = b_matrix
                    for i in range(1, state_dim):
                        term = a_matrix.T @ term
                        controllability_matrix = torch.cat((controllability_matrix, term), dim=1)
                    rank_c = torch.linalg.matrix_rank(controllability_matrix.cpu(), tol=1e-6).to(self.device)
                    singular_values = torch.linalg.svdvals(controllability_matrix.cpu()).to(self.device)
                    max_singular_value = torch.max(singular_values)

                    Gbar_x = self.alg.koopman_estimator.Gbar_x

                    # Get the eigenvalues of Gbar_x
                    Gbar_x_eigvals = torch.linalg.eigvals(Gbar_x)

                    Gbar_reg = Gbar_x + self.alg.koopman_estimator.reg

                    # Get the singular values of the Gbar_x + reg term
                    Gbar_reg_sing_vals = torch.linalg.svdvals(Gbar_reg)

                    # Get the min singular value of Gbar_x to compare to gamma
                    min_singular_value = torch.min(torch.linalg.svdvals(Gbar_x))

                    # Get the current gamma value
                    gamma = self.alg.koopman_estimator.gamma

                    locs["koopman_computation_time"] = koopman_computation_time
                    locs["koopman_pred_error"] = pred_error
                    locs["max_eigval"] = max_eigval.item()
                    locs["min_eigval"] = min_eigval.item()
                    locs["rank_controllability"] = rank_c.item()
                    locs["max_singular_value"] = max_singular_value.item()
                    locs["max_min_eigval_ratio"] = max_eigval.item() / min_eigval.item()
                    locs["A_eigvals"] = torch.abs(eigvals)
                    locs["A_cond_number"] = torch.linalg.cond(a_matrix)
                    locs["Gbar_x_eigvals"] = torch.abs(Gbar_x_eigvals)
                    locs["Gbar_reg_sing_vals"] = Gbar_reg_sing_vals
                    locs["Gbar_x_min_sing_val"] = min_singular_value.item()
                    locs["reg_magnitude"] = gamma

                self.log(locs)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                if "rff_koopman" in self.task:
                    self.save_koopman_model(os.path.join(self.log_dir, 'koopman_model_{}.pt'.format(it)))
                if "online" in self.task:
                    dae_save_dict = {
                        'dae_state_dict': self.alg.dae_model.state_dict(),
                        'normalizer_state_dict': self.alg.obs_action_normalizer.state_dict()
                    }
                    torch.save(dae_save_dict, os.path.join(self.log_dir, 'dae_model_{}.pt'.format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        if "online" in self.task:
            dae_save_dict = {
                'dae_state_dict': self.alg.dae_model.state_dict(),
                'normalizer_state_dict': self.alg.obs_action_normalizer.state_dict()
            }
            torch.save(dae_save_dict, os.path.join(self.log_dir, 'dae_model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_log_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_log_dict[f'Episode/{key}'] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if "online" in self.task:
            self.writer.add_scalar("DAE/loss", locs["mean_dae_loss"], locs["it"]) # or self.current_learning_iteration
            self.writer.add_scalar("DAE/obs_pred_loss", locs["mean_dae_obs_pred_loss"], locs["it"])
            self.writer.add_scalar("DAE/state_rec_loss", locs["mean_dae_state_rec_loss"], locs["it"])
            self.writer.add_scalar("DAE/state_pred_loss", locs["mean_dae_state_pred_loss"], locs["it"])
            self.writer.add_scalar("DAE/train_time", locs["dae_train_time"], locs["it"])
        if "rff_koopman" in self.task:
            self.writer.add_scalar("Koopman/pred_error", locs["koopman_pred_error"], locs["it"])
            self.writer.add_scalar("Koopman/compute_time", locs["koopman_computation_time"], locs["it"])
            self.writer.add_scalar("Koopman/max_eigval", locs["max_eigval"], locs["it"])
            self.writer.add_scalar("Koopman/rank_controllability", locs["rank_controllability"], locs["it"])
            self.writer.add_scalar("Koopman/max_singular_value", locs["max_singular_value"], locs["it"])
            self.writer.add_scalar('Koopman/min_eigval', locs['min_eigval'], locs['it'])
            self.writer.add_scalar('Koopman/max_min_eigval_ratio', locs['max_min_eigval_ratio'], locs['it'])
            self.writer.add_scalar('Koopman/A_cond_number', locs['A_cond_number'], locs['it'])
            self.writer.add_scalar('Koopman/Gbar_x_min_sing_val', locs['Gbar_x_min_sing_val'], locs['it'])
            self.writer.add_scalar('Koopman/reg_magnitude', locs['reg_magnitude'], locs['it'])

        wandb_log_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_log_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_log_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_log_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_log_dict['Perf/total_fps'] = fps
        wandb_log_dict['Perf/collection time'] = locs['collection_time']
        wandb_log_dict['Perf/learning_time'] = locs['learn_time']
        if "online" in self.task:
            wandb_log_dict["DAE/loss"] = locs["mean_dae_loss"]
            wandb_log_dict["DAE/obs_pred_loss"] = locs["mean_dae_obs_pred_loss"]
            wandb_log_dict["DAE/state_rec_loss"] = locs["mean_dae_state_rec_loss"]
            wandb_log_dict["DAE/state_pred_loss"] = locs["mean_dae_state_pred_loss"]
            wandb_log_dict["DAE/train_time"] = locs["dae_train_time"]
        if "rff_koopman" in self.task:
            wandb_log_dict["Koopman/pred_error"] = locs["koopman_pred_error"]
            wandb_log_dict["Koopman/compute_time"] = locs["koopman_computation_time"]
            wandb_log_dict["Koopman/max_eigval"] = locs["max_eigval"]
            wandb_log_dict["Koopman/rank_controllability"] = locs["rank_controllability"]
            wandb_log_dict["Koopman/max_singular_value"] = locs["max_singular_value"]
            # Plot the A_eigvals in a histogram
            # Format the data as a list of lists for the W&B Table
            data_for_table = [[val] for val in locs['A_eigvals']]
            table = wandb.Table(data=data_for_table, columns=["eigval"])
            histogram_plot = wandb.plot.histogram(
                table,
                value="eigval",
                title="Eigenvalue Distribution of A"
            )
            wandb_log_dict['Koopman/A_eigvals'] = histogram_plot
            wandb_log_dict['Koopman/min_eigval'] = locs['min_eigval']
            wandb_log_dict['Koopman/max_min_eigval_ratio'] = locs['max_min_eigval_ratio']
            wandb_log_dict['Koopman/A_cond_number'] = locs['A_cond_number']
            # Plot the Gbar_x eigvals in a table
            data_for_table = [[val] for val in locs['Gbar_x_eigvals']]
            table = wandb.Table(data=data_for_table, columns=["eigval"])
            histogram_plot = wandb.plot.histogram(
                table,
                value="eigval",
                title="Eigenvalue Distribution of Gbar_x"
            )
            wandb_log_dict['Koopman/Gbar_x_eigvals'] = histogram_plot
            # Plot the Gbar_x + reg singular values in a table
            data_for_table = [[val] for val in locs['Gbar_reg_sing_vals']]
            table = wandb.Table(data=data_for_table, columns=["singval"])
            histogram_plot = wandb.plot.histogram(
                table,
                value="singval",
                title="Singular Value Distribution of Gbar_x + reg"
            )
            wandb_log_dict['Koopman/Gbar_reg_sing_vals'] = histogram_plot
            wandb_log_dict['Koopman/reg_magnitude'] = locs['reg_magnitude']
            wandb_log_dict['Koopman/Gbar_x_min_sing_val'] = locs['Gbar_x_min_sing_val']

        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            wandb_log_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_log_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        if self.use_wandb:
            wandb.log(wandb_log_dict, step=self.tot_timesteps)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        self.alg.actor_critic.eval() # switch to eval mode for saving the model
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def save_rff(self, path):
        # RFF is not a learnable model, just save its buffers and configuration
        torch.save({
            'rff_state_dict': self.alg.rff.state_dict(),
            'rff_config': {
                'in_features': self.alg.rff.in_features,
                'sigma': self.alg.rff.sigma,
                'kernel_type': self.alg.rff.kernel_type,
                'm': self.alg.rff.m,
            }
        }, path)

    def save_koopman_model(self, path):
            # Koopman is not a learnable model, just save its buffers and configuration
            torch.save({
                'koopman_state_dict': self.alg.koopman_estimator.state_dict(),
                'koopman_config': {
                    'koopman_input_dim': self.alg.koopman_estimator.koopman_input_dim,
                    'koopman_output_dim': self.alg.koopman_estimator.koopman_output_dim,
                    'gamma': self.alg.koopman_estimator.gamma,
                    'K': self.alg.koopman_estimator.K_matrix,
                }
            }, path)

    def load(self, path, load_optimizer=True):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'], strict=False)
        # if load_optimizer:
        #     self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
