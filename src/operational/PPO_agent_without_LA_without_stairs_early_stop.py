# -*- coding: utf-8 -*-
"""
PPO Main - Operational Thesis Version (NEW V5 - With Window / Without Look-Ahead)
- Synchronized evaluation loop executed over the entire validation dataset tracks.
- Separated plotting layouts generated in English (Raw Train, Full Validation, Network Loss).
- Adaptive PPO Rollout Reward Extraction tailored for fractionated sequence collections.

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

tf.keras.backend.set_floatx('float32')
tf.keras.backend.clear_session()

# PPO FRAMEWORK CORE IMPORTS
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.utils import common

import time
import os
from tqdm import tqdm
from datetime import datetime

# === IMPORT CLASSE NEW V5 === #
from Classes_TD3_without_LA_without_stairs import Environment, Environmentvalidation, importdata

# %% -------Title of run-------
title_of_run = ('PPO Operational Agent Training Pipeline - Logic NEW V5 Implementation Setup')
print(title_of_run)

# %% ------- Seed (Total Reproducibility Setup) -------
import random

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Enforce deterministic processing operations even on GPU architectures

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# %% -------Parameters + Data-------
battery_max_power = 20  # MW
EP_ratio = 1
delta_t = 0.25  # time control interval (h)
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# Historical window size parameters (The operational agent monitors a 4-hour historical window = 16 steps back)
window_size = 8

penal = 0

# Causal operational datasets including integrated warm-up historical tracks
path_training = os.path.normpath('Dataset_Train_2018_2019_wdw8.xlsx')
path_validation = os.path.normpath('Dataset_Val_2018_2019_wdw8.xlsx')
path_test = os.path.normpath('Dataset_Test_2018_2019_wdw8.xlsx')

print('Importing data + preprocessing (NEW V5)...')
# Parsing data matrices without global scalers or look-ahead signals
train_obs, val_obs, test_obs, \
    train_non_obs, val_non_obs, test_non_obs = importdata(path_training, path_validation, path_test, window_size)
print('Preprocessed data imported')

# %% -------Hyperparameters PPO-------
nb_quarters_per_episode = 96

# =======================================================
# PPO ALGORITHMIC PARAMETERS SETUP
# =======================================================
nbr_jour_rollout = 1
collect_steps_per_iteration = int(nbr_jour_rollout * nb_quarters_per_episode) 

ppo_epochs = 1
replay_buffer_max_length = collect_steps_per_iteration + 1

gamma = np.float32(0.99)
learning_rate = 1e-4
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, clipnorm=1.0)

hidden_actor = (500, 500)
hidden_critic = (500, 500)

ratio_clip=0.2

num_epochs_global = 12
total_steps_in_dataset = len(train_obs)
num_iterations = int((total_steps_in_dataset * num_epochs_global) / collect_steps_per_iteration)
num_val_episodes = int(len(val_obs) / nb_quarters_per_episode)

# --- TRACKING INTERVALS SYNCHRONIZATION SEGMENTS ---
# Synchronize data evaluation intervals to output a performance point every 9,600 environment steps
steps_between_eval = 9600 
eval_interval = max(1, steps_between_eval // collect_steps_per_iteration)

# Output log file summaries metrics every 1,920 environment steps
steps_between_log = 1920
log_interval = max(1, steps_between_log // collect_steps_per_iteration)

# ------- Export Hyperparameters Configuration -------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.getcwd(), f'Run_PPO_{timestamp}_EARLY_STOP_WDW8_S1_penal_0')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config_path = os.path.join(save_dir, "hyperparameters.txt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(f"{' Category ':*^40}\n")
    f.write(f"  - Algorithm                  : PPO (Logic NEW V5)\n")
    f.write(f"  - Window size (Past History) : {window_size}\n")
    f.write(f"  - Collect Steps (Rollout)    : {collect_steps_per_iteration} (/96 = Days)\n")
    f.write(f"  - PPO Epochs per rollout     : {ppo_epochs}\n")
    f.write(f"  - Global Epochs              : {num_epochs_global}\n")
    f.write(f"  - Gamma (Discount factor)    : {gamma}\n")
    f.write(f"  - Learning Rate              : {learning_rate}\n")
    f.write(f"  - Ratio clipping             : {ratio_clip}\n")
    f.write(f"  - Hidden layers (Actor)      : {hidden_actor}\n")
    f.write(f"  - Hidden layers (Critic)     : {hidden_critic}\n")
    f.write(f"\n{'*'*40}\n")
print(f"Dossier créé et hyperparamètres sauvés : {save_dir}")

# %% -------Environments-------
starting_time = time.time()

# Instantiate causal operational environment layouts without privileged market signals
train_env = Environment(train_obs, train_non_obs, battery_max_power,
                        1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal)  
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

validation_env = Environmentvalidation(val_obs, val_non_obs, battery_max_power,
                                 1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                                 battery_replacement_cost)
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)

# %% -------PPO Actor and critic-------
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_train_env.observation_spec(),
    tf_train_env.action_spec(),
    fc_layer_params=hidden_actor,
    activation_fn=tf.keras.activations.relu)

value_net = value_network.ValueNetwork(
    tf_train_env.observation_spec(),
    fc_layer_params=hidden_critic,
    activation_fn=tf.keras.activations.relu)

# %% -------Agent-------
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    tf_train_env.time_step_spec(),
    tf_train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=ppo_epochs,
    importance_ratio_clipping=ratio_clip,
    entropy_regularization=0.05,  
    use_gae=True,                  
    use_td_lambda_return=True,
    discount_factor=gamma,
    train_step_counter=train_step_counter)

agent.initialize()

# ====================================================================
# --- INITIALIZE CORE CHECKPOINTER LOGIC CONTAINER ---
# ====================================================================
checkpoint_dir = os.path.join(save_dir, 'checkpoint_complet')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1, # Restrict disk space utilization by keeping only top weight matches
    agent=agent,
    policy=agent.policy,
    global_step=train_step_counter
)
# ====================================================================

eval_policy = agent.policy
collect_policy = agent.collect_policy

# %% -------Replay buffer & Driver-------
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_train_env.batch_size,
    max_length=replay_buffer_max_length)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration)

collect_driver.run = common.function(collect_driver.run)

# %% ------- Synchronized Manual Evaluation Interface -------
def evaluate_agent(tf_env, policy, num_episodes, nb_quarters):
    """ Evaluates operational policy returns over the complete validation timeline dataset """
    py_env = tf_env.pyenv.envs[0]
    
    # Pristine variable memory override reset sequence
    py_env._observation_index = 0
    py_env._episode = 0
    py_env._observation_samples[0, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[0]
    time_step = tf_env.reset()

    total_profit = 0.0
    
    # Maintain tracking parameters of hidden policy states for recurrent validation layers
    policy_state = policy.get_initial_state(tf_env.batch_size)

    for _ in range(num_episodes):
        episode_profit = 0.0
        for _ in range(nb_quarters):
            if time_step.is_last():
                time_step = tf_env.reset()
                policy_state = policy.get_initial_state(tf_env.batch_size)
                
            action_step = policy.action(time_step, policy_state)
            policy_state = action_step.state
            time_step = tf_env.step(action_step.action)
            episode_profit += time_step.reward.numpy()[0]
            
        total_profit += episode_profit
        
    return total_profit / num_episodes

# %% -------Start time-------
start = datetime.now()
agent.train_step_counter.assign(0)

print("\nPOLICY EVALUATION BEFORE TRAINING \n")
avg_return_initial = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
print(f"Average return (before training): {avg_return_initial:.2f} € / day")

returns = [avg_return_initial]
average_train_rewards = []
train_loss_plot = []

best_avg_return = avg_return_initial
saver = policy_saver.PolicySaver(agent.policy)


# --- ADAPTIVE DEFENSIVE EARLY STOPPING CONFIGURATION CONTROLS ---
patience = 15             # Maximum allowed consecutive evaluation cycles without performance gains
wait_counter = 0         # Current consecutive convergence plateau count
min_delta = 1.0          # Required performance gain margin parameter used to reset the counter
early_stop_triggered = False 
# -------------------------------------------

print("\n START PPO TRAINING ... \n")
# %% -------BOUCLE D'ENTRAÎNEMENT PPO-------
print("\n START PPO TRAINING ... \n")

time_step = tf_train_env.reset()
policy_state = collect_policy.get_initial_state(tf_train_env.batch_size)

for i in tqdm(range(num_iterations), desc="PPO Rollouts", unit="iter"):

    # 1. TRAJECTORY COLLECTION (Rollout execution pass)
    time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )

    # Extract rollout reward metrics directly from buffer structures
    trajectories = replay_buffer.gather_all()
    rollout_total_reward = tf.reduce_sum(trajectories.reward).numpy()
    
    # Normalize rewards based on fraction day coverage of the rollout steps slice
    avg_train_per_day = rollout_total_reward / (collect_steps_per_iteration / nb_quarters_per_episode)

    # 2. PPO GRADIENTPASS BACKPROPAGATION UPDATE
    train_loss = agent.train(experience=trajectories).loss
    
    # 3. MEMORY BUFFER PURGE
    replay_buffer.clear()

    # --- Append tracking states arrays ---
    train_loss_plot.append(train_loss.numpy())
    average_train_rewards.append(avg_train_per_day)
    
    if (i + 1) % log_interval == 0:
        print(f"Iter {i+1:04d} | Total Loss PPO: {train_loss.numpy():>8.2f} | Avg Reward (Train) : {avg_train_per_day:>8.2f} € / day")
        
    if (i + 1) % eval_interval == 0:
        avg_ret = evaluate_agent(tf_validation_env, agent.policy, num_val_episodes, nb_quarters_per_episode)
        returns.append(avg_ret)
        
        print("\n"+"-"*60)
        print(f"EVALUATION (Iter {i+1:04d}) | Validation Return (full): {avg_ret:>8.2f} € / day")

        # --- EVALUATE PERFORMANCE IMPROVEMENTS FOR EARLY STOPPING PASS ---
        if avg_ret > (best_avg_return + min_delta):
            print(f" -> AMÉLIORATION : {avg_ret:.2f} > {best_avg_return:.2f} + {min_delta}")
            best_avg_return = avg_ret
            wait_counter = 0  # Reset patience index parameter following record breach
            
            # Save checkpoint structures and policy network configurations models
            policy_dir = os.path.join(save_dir, "saved_policy_final")
            saver.save(policy_dir)
            train_checkpointer.save(global_step=agent.train_step_counter)
            print(" -> New best policy AND CHECKPOINT saved.")
            
        else:
            wait_counter += 1
            print(f" -> Pas d'amélioration suffisante. Patience : {wait_counter} / {patience}")
            
        print("-" * 60)

        # Trigger early stopping brake loop sequence if performance stagnation bounds are breached
        if wait_counter >= patience:
            print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
            print(f"L'agent ne progresse plus. Arrêt à l'itération {i+1} pour éviter le sur-apprentissage.")
            early_stop_triggered = True
            break 

# %% -------End of training time-------
end = datetime.now()
td = (end - start).total_seconds()
nb_hours = td / 3600

if early_stop_triggered:
    print(f"\n[INFO] Entraînement écourté par l'Early Stopping.")
    
print(f"\nThe time of execution is : {td:.03f} s ({nb_hours:.03f} h)")

# %%=============================================================================
# SPREADSHEET EXPORT SEGMENTATION (AUTOMATED MULTI-SHEET EXCEL LOGGING)
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
# PLOTTING SEPARATED CHARTS INTERFACE (ENGLISH CODES SETUP)
# =============================================================================
print("\nGenerating and saving plots...")

# 1. Train Reward Curves Layout Configuration
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

# 2. Validation Horizon Return Performance Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
# Render X-axis in absolute environment steps configurations for direct alignment layers matching TD3
steps_val = np.arange(len(returns)) * eval_interval * collect_steps_per_iteration 
ax2.plot(steps_val, returns, marker='o', color='red', linewidth=2, label='Validation (Full Dataset)')
ax2.set_title("Evaluation Performance on Validation Set (PPO)", fontsize=14)
ax2.set_xlabel("Environment Steps")
ax2.set_ylabel("Average Profit (€ / day)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

# 3. Stochastic Total PPO Convergence Network Loss curves
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
print(f"\nFin d'exécution. Les 3 graphiques ont été sauvegardés dans : {save_dir}")
