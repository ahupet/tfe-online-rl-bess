# -*- coding: utf-8 -*-
"""
TD3 Main - Final Thesis Version (Synchronized Training & Evaluation Loop)

Last Update: Mon June 1 2026

@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

# %% -------import-------
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Force the execution backend engine to standard float32 precision limits
tf.keras.backend.set_floatx('float32')
tf.keras.backend.clear_session()

from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common

import time
import os
from datetime import datetime
from tf_agents.policies import policy_saver
from tqdm import tqdm

# Import the specialized causal BESS environment structures
from Classes_TD3_old_final import Environment, Environmentvalidation, importdata, collect_data

# %% -------Title of run-------
title_of_run = ('TD3 Agent Training Pipeline - Final BESS Implementation Setup')
print(title_of_run)

# %%
seed = 2
np.random.seed(seed)
tf.random.set_seed(seed)

# %% -------Parameters + Data-------
battery_max_power = 20  # Max power limit (MW)
EP_ratio = 1
delta_t = 0.25  # Time step tracking window frequency (15 minutes)
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

LA_steps = 8
penal = 200
SafeProjectionMode = False

# Path mapping for historical market simulation data sheets (2018-2019)
path_training = os.path.normpath('Frame_1819_training.xlsx')
path_validation = os.path.normpath('Frame_1819_validation.xlsx')
path_test = os.path.normpath('Frame_1819_test.xlsx')

print('Importing data + preprocessing...')
index_col_si, index_first_col_MDP, scaler, scaler_list, train_observable_samples, validation_observable_samples, test_observable_samples, \
    train_exact_si_samples, validation_exact_si_samples, test_exact_si_samples = importdata(path_training,
                                                                                            path_validation, path_test,
                                                                                            LA_steps)
print('Preprocessed data imported')

index_soc = 28

# %% -------Hyperparameters-------
tuned_initial_exp = 1   # 0 = yes, 1 = no
nb_quarters_per_episode = 96 # Bounded 24-hour cycle window horizon
replay_buffer_max_length = int(2 * 1e4)
batch_size = 256
num_epochs = 4

gamma = np.float64(0.99)
tau = 1e-2
actor_lr = 1e-4
critic_lr = 1e-3

# Initialize legacy Adam gradient optimizers with specific gradient clipping
actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr, clipnorm=1)
critic_optimizer = tf.keras.optimizers.legacy.Adam(critic_lr, clipnorm=1)

actor_update_T = 4
target_update_T = 4

hidden_actor = (500,500)
hidden_critic = (500,500)

# ------- Export Hyperparameters Configuration -------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.getcwd(), f'Run_{timestamp}_FINAL')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Save logging configurations into a baseline metadata text file
config_path = os.path.join(save_dir, "hyperparameters.txt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(f"{' Category ':*^40}\n")
    f.write("\n[Episode Configuration]\n")
    f.write(f"  - Quarters per episode (24h) : {nb_quarters_per_episode}\n")
    f.write(f"  - Initial exploration (0=Yes): {tuned_initial_exp}\n")
    f.write(f"  - Penalty : {penal}\n")
    f.write(f"  - Look Ahead steps (LA_steps): {LA_steps}\n")
    f.write("\n[Training Specifications]\n")
    f.write(f"  - Number of epochs            : {num_epochs}\n")
    f.write(f"  - Batch size                  : {batch_size}\n")
    f.write(f"  - Replay buffer max length   : {replay_buffer_max_length}\n")
    f.write("\n[RL Dynamics]\n")
    f.write(f"  - Gamma (Discount factor)     : {gamma}\n")
    f.write(f"  - Tau (Soft update rate)      : {tau}\n")
    f.write(f"  - Actor update period (T)     : {actor_update_T}\n")
    f.write(f"  - Target update period (T)   : {target_update_T}\n")
    f.write("\n[Optimizers & Learning Rates]\n")
    f.write(f"  - Actor LR                    : {actor_lr}\n")
    f.write(f"  - Critic LR                   : {critic_lr}\n")
    f.write("  - Optimizer type              : Adam (with clipnorm=1)\n")
    f.write("\n[Network Architecture]\n")
    f.write(f"  - Hidden layers (Actor)      : {hidden_actor}\n")
    f.write(f"  - Hidden layers (Critic)     : {hidden_critic}\n")
    f.write(f"\n{'*'*40}\n")

print(f"Dossier créé et hyperparamètres sauvés : {save_dir}")

# %%
actor_activ = tf.nn.relu
critic_activ = tf.nn.relu
noise_stddev = 0.2
initial_collect_steps = batch_size * 2
collect_steps_per_iteration = 4

num_iterations = int(num_epochs * int((len(train_observable_samples) - initial_collect_steps) / collect_steps_per_iteration))
num_training_episodes = int(num_iterations / nb_quarters_per_episode)

log_interval = 500  
eval_interval = 2000 

num_val_episodes = int(len(validation_observable_samples) / nb_quarters_per_episode)

# %% -------Environments-------
starting_time = time.time()

# Instantiating Python environments and casting them into standard TensorFlow equivalents
train_env = Environment(train_observable_samples, train_exact_si_samples, scaler, scaler_list, battery_max_power,
                        1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal)
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

validation_env = Environmentvalidation(validation_observable_samples, validation_exact_si_samples, scaler, scaler_list, battery_max_power,
                                 1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                                 battery_replacement_cost)
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)

# %% -------Actor and critic-------
# Creating the neural network multi-layer perceptron (MLP) mapping functions for TD3
Critic = critic_network.CriticNetwork((tf_train_env.observation_spec(), tf_train_env.action_spec()),
                                      joint_fc_layer_params=hidden_critic, activation_fn=critic_activ)

Actor = actor_network.ActorNetwork(tf_train_env.observation_spec(), tf_train_env.action_spec(), hidden_actor,
                                   activation_fn=actor_activ)
target_actor = actor_network.ActorNetwork(tf_train_env.observation_spec(), tf_train_env.action_spec(), hidden_actor,
                                          activation_fn=actor_activ)

# %% -------Agent-------
train_step_counter = tf.Variable(0)
agent = td3_agent.Td3Agent(tf_train_env.time_step_spec(), tf_train_env.action_spec(), Actor, Critic,
                           actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                           exploration_noise_std=noise_stddev, target_update_tau=tau, target_update_period=target_update_T,
                           actor_update_period=actor_update_T, td_errors_loss_fn=None, gamma=gamma, target_policy_noise=0.2,
                           reward_scale_factor=np.float64(1), target_policy_noise_clip=0.5, gradient_clipping=None, 
                           debug_summaries=None, target_actor_network=target_actor, summarize_grads_and_vars=False, 
                           train_step_counter=train_step_counter)

agent.initialize()

eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

# %% -------Replay buffer-------
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=agent.collect_data_spec,
                                                               batch_size=tf_train_env.batch_size,
                                                               max_length=replay_buffer_max_length)

# %% ------- Synchronized Evaluation function over Full Validation Set -------
def evaluate_agent(tf_env, policy, num_episodes, nb_quarters):
    """ Evaluates current policy performance over the entire non-stationary Validation Track """
    py_env = tf_env.pyenv.envs[0]
    
    # Reset internal tracking variables to pristine conditions
    py_env._observation_index = 0
    py_env._episode = 0
    py_env._observation_samples[0, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[0]
    time_step = tf_env.reset()

    total_profit = 0.0

    for _ in range(num_episodes):
        episode_profit = 0.0
        for _ in range(nb_quarters):
            if time_step.is_last():
                time_step = tf_env.reset()
                
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_profit += time_step.reward.numpy()[0]
            
        total_profit += episode_profit
        
    return total_profit / num_episodes

# %% -------Initial fill-------
start = datetime.now()
print("\nData collection ... \n")
collect_data(tf_train_env, random_policy, replay_buffer, initial_collect_steps, tuned_initial_exp)

# Compile standard TF dataset pipeline loops from the replay buffer states
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).shuffle(
    buffer_size=replay_buffer_max_length).prefetch(3)

iterator = iter(dataset)

# %% -------Training Preparation-------
agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

print("\nPOLICY EVALUATION BEFORE TRAINING \n")
avg_return_initial = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
print(f"Average return (before training): {avg_return_initial:.2f} € / jour")

returns = [avg_return_initial]
train_rewards = []
average_train_rewards = []
train_loss_plot = []

best_avg_return = avg_return_initial
saver = policy_saver.PolicySaver(agent.policy) 

# %% ------- TRAINING LOOP -------
print("\n START TRAINING ... \n")

for ep in tqdm(range(num_training_episodes), desc="Episodes d'entrainement", unit="ep"):

    for _ in range(nb_quarters_per_episode):

        # 1. Gather environmental operational transition metrics
        rew, act = collect_data(tf_train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration, 1)
        
        if tf.is_tensor(rew):
            train_rewards.append(rew.numpy()[0])
        else:
            train_rewards.append(rew)

        # 2. Execute gradient optimization pass over the actor-critic weights
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        train_loss_plot.append(train_loss.numpy())

        # 3. Step update incrementation
        step = agent.train_step_counter.numpy()

        # LOGGING PERFORMANCE STATS
        if step % log_interval == 0:
            total_steps_collected = log_interval * collect_steps_per_iteration
            profit_per_step = np.sum(train_rewards) / total_steps_collected
            avg_train_per_day = profit_per_step * nb_quarters_per_episode
            
            電力_rewards = avg_train_per_day
            average_train_rewards.append(avg_train_per_day)
            print(f"Step {step:05d} | Loss Critic: {train_loss.numpy():>8.2f} | Avg Reward (Train) : {avg_train_per_day:>8.2f} € / jour")
            train_rewards = []

        # RUN CHRONOLOGICAL DISCRETE VALIDATION CYCLE
        if step % eval_interval == 0:
            avg_return = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
            returns.append(avg_return)
            
            print("\n"+"-"*60)
            print(f"EVALUATION (Step {step:05d}) | Validation Return (complet): {avg_return:>8.2f} € / jour")

            # Checkpoint checkpoint saver logic to isolate top weights configurations
            if avg_return > best_avg_return:
                best_avg_return = avg_return
                policy_dir = os.path.join(save_dir, "saved_policy_final")
                saver.save(policy_dir)
                print(f"New best policy saved with return: {best_avg_return:.2f} € / jour")

# %% -------End of training time-------
end = datetime.now()
td = (end - start).total_seconds()
nb_hours = td / 3600
print(f"\nThe time of execution is : {td:.03f} s")
print(f"Equivalent to : {nb_hours:.03f} h")

# %%=============================================================================
# SEPARATED PLOTTING LAYOUT PIPELINE
# ==============================================================================
print("\nGénération et sauvegarde des graphiques...")

def moving_average(a, n=10):
    if len(a) < n: return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 1. Train Reward Graphic Generation
fig1, ax1 = plt.subplots(figsize=(10, 6))
steps_train = np.arange(len(average_train_rewards)) * log_interval
ax1.plot(steps_train, average_train_rewards, color='blue', linewidth=1.5, label='Train (Raw)')
ax1.set_title("Training Profit Evolution", fontsize=14)
ax1.set_xlabel("Steps")
ax1.set_ylabel("Profit (€ / day)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(save_dir, "1_Train_Reward.png"), dpi=300)

# 2. Validation Tracking Performance Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
steps_val = np.arange(len(returns)) * eval_interval
ax2.plot(steps_val, returns, marker='o', color='red', linewidth=2, label='Validation (Full Dataset)')
ax2.set_title("Evaluation Performance on Validation Set", fontsize=14)
ax2.set_xlabel("Steps")
ax2.set_ylabel("Average Profit (€ / day)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

# 3. Critic Convergence Network Loss curves (With moving-average smoothing filter)
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Plot the raw background tracking data array with transparency
ax3.plot(train_loss_plot, color='orange', alpha=0.3, linewidth=1, label='Loss (Raw)')

# Layer a rolling average smoothed plot on top for better visualization trends
lissage = min(50, len(train_loss_plot)) 
if lissage > 1:
    loss_smoothed = moving_average(train_loss_plot, n=lissage)
    ax3.plot(np.arange(lissage-1, len(train_loss_plot)), loss_smoothed, color='darkorange', linewidth=2, label='Loss (Smoothed)')

ax3.set_title("Network Convergence (Loss)", fontsize=14)
ax3.set_xlabel("Steps / Iterations")
ax3.set_ylabel("Loss")
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
fig3.savefig(os.path.join(save_dir, "3_Loss.png"), dpi=300)

plt.show()
print(f"\nFin d'exécution. Les 3 graphiques ont été sauvegardés dans : {save_dir}")
