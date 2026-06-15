# -*- coding: utf-8 -*-
"""
PPO Main - Thesis Implementation Paradigm (Synchronized Training, Checkpoints & Early Stopping)

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
import random
import time
import os
import shutil 
import gc 
from tqdm import tqdm
from datetime import datetime

# --- GPU MEMORY ALLOCATION SECURITY CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable dynamic memory allocation growth to avoid full VRAM lockups
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: 
        print(e)

tf.keras.backend.set_floatx('float32')
tf.keras.backend.clear_session()

from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.utils import common

# Import the benchmark environment loading classes
from Classes_TD3_old_final import Environment, Environmentvalidation, importdata

# %% -------Title & Seed-------
title_of_run = ('PPO Agent Training Pipeline - Online Baseline Implementation Setup')
print(title_of_run)

seed = 2
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Enforce execution reproducibility steps
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %% -------Parameters + Data-------
battery_max_power = 20
EP_ratio = 1
delta_t = 0.25
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio
LA_steps = 8
penal = 200

path_training = os.path.normpath('Frame_1819_training.xlsx')
path_validation = os.path.normpath('Frame_1819_validation.xlsx')
path_test = os.path.normpath('Frame_1819_test.xlsx')

print('Importing data + preprocessing (OLD)...')
index_col_si, index_first_col_MDP, scaler_si, scaler_list, \
    train_obs, val_obs, test_obs, \
    train_non_obs, val_non_obs, test_non_obs = importdata(path_training, path_validation, path_test, LA_steps)
print('Preprocessed data imported')

# %% -------Hyperparameters-------
nb_quarters_per_episode = 96
nbr_jour_rollout = 1
collect_steps_per_iteration = int(nbr_jour_rollout * nb_quarters_per_episode) # Entire day rollout steps tracking
ppo_epochs = 1
replay_buffer_max_length = collect_steps_per_iteration + 1

gamma = np.float32(0.99)
learning_rate = 1e-4
ratio_clip = 0.2 # Trust region policy clipping parameter constraint
num_epochs_global = 12
log_interval = 25
eval_interval = 100
hidden_actor = (500, 500)
hidden_critic = (500, 500)

# Determine global processing dimensions limits
num_val_episodes = int(len(val_obs) / nb_quarters_per_episode)
num_iterations = int((len(train_obs) * num_epochs_global) / collect_steps_per_iteration)

# Setup custom logging directory configurations
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.getcwd(), f'Run_PPO_OLD_OPTI_{timestamp}_S2')
if not os.path.exists(save_dir): 
    os.makedirs(save_dir)

# %% -------Environments-------
train_env = Environment(train_obs, train_non_obs, scaler_si, scaler_list, battery_max_power,
                        1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal)  
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

validation_env = Environmentvalidation(val_obs, val_non_obs, scaler_si, scaler_list, battery_max_power,
                                 1, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
                                 battery_replacement_cost)
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)

# %% -------Networks & Agent-------
# Instantiate multi-layer perceptrons (MLP) architectures for stochastic processing
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_train_env.observation_spec(), tf_train_env.action_spec(),
    fc_layer_params=hidden_actor, activation_fn=tf.keras.activations.relu)

value_net = value_network.ValueNetwork(
    tf_train_env.observation_spec(),
    fc_layer_params=hidden_critic, activation_fn=tf.keras.activations.relu)

train_step_counter = tf.Variable(0)
agent = ppo_agent.PPOAgent(
    tf_train_env.time_step_spec(), tf_train_env.action_spec(),
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0),
    actor_net=actor_net, value_net=value_net, num_epochs=ppo_epochs,
    importance_ratio_clipping=ratio_clip, entropy_regularization=0.05,
    use_gae=True, use_td_lambda_return=True, discount_factor=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

# --- INSTANTIATE TRAINING STATE CHECKPOINTER MANAGEMENT ---
checkpoint_dir = os.path.join(save_dir, 'checkpoint_complet')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir, max_to_keep=1,
    agent=agent, policy=agent.policy, global_step=train_step_counter)

# Setup Replay Buffer allocations
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=tf_train_env.batch_size,
    max_length=replay_buffer_max_length)

# Establish specialized driver objects for processing sequential interaction rollouts
collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_train_env, agent.collect_policy,
    observers=[replay_buffer.add_batch], num_steps=collect_steps_per_iteration)
collect_driver.run = common.function(collect_driver.run)

# %% ------- Evaluation & Prep -------
def evaluate_agent(tf_env, policy, num_episodes, nb_quarters):
    """ Executes validation evaluations tracking recurrent policy_states for the PPO network """
    py_env = tf_env.pyenv.envs[0]
    py_env._observation_index = 0
    py_env._episode = 0
    py_env._observation_samples[0, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[0]
    time_step = tf_env.reset()
    
    total_profit = 0.0
    # Capture state indicators for stochastic networks architecture tracking requirements
    policy_state = policy.get_initial_state(tf_env.batch_size)

    for _ in range(num_episodes):
        episode_profit = 0.0
        for _ in range(nb_quarters):
            if time_step.is_last(): 
                time_step = tf_env.reset()
                policy_state = policy.get_initial_state(tf_env.batch_size)
                
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state # Update policy state parameters sequence
            time_step = tf_env.step(action_step.action)
            episode_profit += time_step.reward.numpy()[0]
            
        total_profit += episode_profit
    return total_profit / num_episodes

# --- CONFIGURING ADAPTIVE DEFENSIVE EARLY STOPPING CRITERIA ---
patience = 15
wait_counter = 0
min_delta = 1.0 
early_stop_triggered = False

saver = policy_saver.PolicySaver(agent.policy)

# %% -------Start time-------
start = datetime.now()
agent.train_step_counter.assign(0)

print("\nPOLICY EVALUATION BEFORE TRAINING \n")
avg_return_initial = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
print(f"Average return (before training): {avg_return_initial:.2f} € / day")

best_avg_return = avg_return_initial
returns = [avg_return_initial]
average_train_rewards = []
train_loss_plot = []

# %% -------TRAINING LOOP-------
print("\n START PPO TRAINING ... \n")
time_step = tf_train_env.reset()
policy_state = agent.collect_policy.get_initial_state(tf_train_env.batch_size)

for i in tqdm(range(num_iterations), desc="PPO Rollouts", unit="iter"):
    
    # 1. Execute trajectory rollout data generation
    time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
    trajectories = replay_buffer.gather_all()
    
    # Extract online reward metrics
    rollout_reward = tf.reduce_sum(trajectories.reward).numpy()
    avg_train_per_day = rollout_reward / (collect_steps_per_iteration / nb_quarters_per_episode)
    average_train_rewards.append(avg_train_per_day)

    # 2. Gradient backpropagation pass optimization execution
    train_loss = agent.train(experience=trajectories).loss
    train_loss_plot.append(train_loss.numpy())
    replay_buffer.clear() # Clear buffer memory allocations following update

    # 3. Handle console summary tracking
    if (i + 1) % log_interval == 0:
        gc.collect() # Trigger RAM garbage collector routine to eliminate memory leaks
        print(f"Iter {i+1:04d} | Loss: {train_loss.numpy():>8.2f} | Train Profit: {avg_train_per_day:>8.2f} €/day")

    # 4. Periodic validation track evaluation & early stopping loop execution
    if (i + 1) % eval_interval == 0:
        avg_ret = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
        returns.append(avg_ret)
        print(f"\nEVALUATION (Iter {i+1:04d}) | Return: {avg_ret:>8.2f} €/jour")

        # Evaluate progress bounds against historical records configurations
        if avg_ret > (best_avg_return + min_delta):
            print(f" -> AMÉLIORATION : {avg_ret:.2f} > {best_avg_return:.2f} + {min_delta}")
            best_avg_return = avg_ret
            wait_counter = 0
            
            policy_dir = os.path.join(save_dir, "saved_policy_final")
            if os.path.exists(policy_dir): 
                shutil.rmtree(policy_dir)
            saver.save(policy_dir)
            train_checkpointer.save(global_step=agent.train_step_counter)
            print(" -> Record battu : Sauvegarde Checkpoint effectuée.")
        else:
            wait_counter += 1
            print(f" -> Pas d'amélioration. Patience : {wait_counter}/{patience}")
            
        print("-" * 60)

        # Early stopping execution step if convergence plateau limit is breached
        if wait_counter >= patience:
            print("\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
            early_stop_triggered = True
            break

# %% -------End of training time-------
end = datetime.now()
td = (end - start).total_seconds()
if early_stop_triggered:
    print(f"\n[INFO] Entraînement écourté par l'Early Stopping.")
print(f"\nThe time of execution is : {td:.03f} s ({td/3600:.03f} h)")

# %%=============================================================================
# SPREADSHEET LOGGING OUTPUT CONFIGURATIONS (MULTI-SHEET PIPELINE)
# =============================================================================
print("\nExportation des données d'entraînement vers Excel...")

df_val = pd.DataFrame({
    'Environment_Steps': np.arange(len(returns)) * eval_interval * collect_steps_per_iteration,
    'Validation_Reward_Eur_Jour': returns
})

df_train = pd.DataFrame({
    'Environment_Steps': (np.arange(len(average_train_rewards)) + 1) * log_interval * collect_steps_per_iteration,
    'Train_Reward_Eur_Jour': average_train_rewards
})

df_loss = pd.DataFrame({
    'PPO_Iterations': np.arange(len(train_loss_plot)),
    'PPO_Loss_Total': train_loss_plot
})

excel_path = os.path.join(save_dir, "Historique_Entrainement_PPO.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df_val.to_excel(writer, sheet_name='Validation_Reward', index=False)
    df_train.to_excel(writer, sheet_name='Train_Reward', index=False)
    df_loss.to_excel(writer, sheet_name='PPO_Loss', index=False)

print(f"-> Données brutes sauvegardées avec succès dans : {excel_path}")

# %%=============================================================================
# PLOTTING METRICS GENERATION LAYOUT
# =============================================================================
print("\nGenerating and saving plots...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
steps_train = np.arange(len(average_train_rewards)) * log_interval * collect_steps_per_iteration
ax1.plot(steps_train, average_train_rewards, color='blue', linewidth=1.5, label='Train (Raw)')
ax1.set_title("Training Profit Evolution (PPO)", fontsize=14)
ax1.set_xlabel("Environment Steps")
ax1.set_ylabel("Profit (€ / day)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(save_dir, "1_Train_Reward.png"), dpi=300)

fig2, ax2 = plt.subplots(figsize=(10, 6))
steps_val = np.arange(len(returns)) * eval_interval * collect_steps_per_iteration 
ax2.plot(steps_val, returns, marker='o', color='red', linewidth=2, label='Validation')
ax2.set_title("Evaluation Performance on Validation Set (PPO)", fontsize=14)
ax2.set_xlabel("Environment Steps")
ax2.set_ylabel("Average Profit (€ / day)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(train_loss_plot, color='orange', alpha=0.6, linewidth=1, label='Total PPO Loss')
ax3.set_title("Network Convergence (PPO Total Loss)", fontsize=14)
ax3.set_xlabel("PPO Iterations")
ax3.set_ylabel("Loss")
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
fig3.savefig(os.path.join(save_dir, "3_PPO_Loss.png"), dpi=300)

plt.show()
print(f"\nFin d'exécution. Les graphiques et Excel sont dans : {save_dir}")
