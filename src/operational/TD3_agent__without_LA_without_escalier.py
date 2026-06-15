# -*- coding: utf-8 -*-
"""
TD3 Main - Operational Thesis Version (NEW V5 - Pure Deduction)
- Synchronized evaluation loop executed over the entire validation dataset tracks.
- Anti-Crash Memory Protections (GPU VRAM Growth, TF Functions, Shutil Cleanup, GC Collect).
- Spreadsheet export generation logging automated training profiles data columns.
- Smooth Defensive Early Stopping verification triggers and full checkpoint storage saves.

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
import gc # <--- Garbage Collector module integration used to strictly force memory releases
from tqdm import tqdm
from datetime import datetime

# --- ANTI-CRASH MEMORY CONFIGURATION SAFEGUARD N°1: VRAM Growth Allocation ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable dynamic memory growth to prevent TensorFlow from locking the whole GPU VRAM capacity
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

tf.keras.backend.set_floatx('float32')
tf.keras.backend.clear_session()

from tf_agents.agents.td3 import td3_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.utils import common
from tf_agents.policies import policy_saver

# === IMPORT CAUSAL OPERATIONAL DATA PIPELINE LAYER (NEW V5) === #
from Classes_TD3_without_LA_without_stairs import Environment, Environmentvalidation, importdata, collect_data

# %% -------Title of run-------
title_of_run = ('TD3 Operational Agent Training Pipeline - Logic NEW V5 Setup')
print(title_of_run)

# %% ------- Seed (Total Performance Reproducibility Setup) -------
seed = 3
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' # Guarantee exact gradient calculation across sessions execution passes

random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed) 

# %% -------Parameters + Data-------
battery_max_power = 20  # MW
EP_ratio = 1
delta_t = 0.25  # time control interval (h)
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# Causal window tracking size (The agent monitors 4 hours = 16 sequential steps into past records)
window_size = 8
penal = 200

# Input training data matrices including warm-up sliding indices blocks
path_training = os.path.normpath('Dataset_Train_2018_2019_wdw8.xlsx')
path_validation = os.path.normpath('Dataset_Val_2018_2019_wdw8.xlsx')
path_test = os.path.normpath('Dataset_Test_2018_2019_wdw8.xlsx') 

print('Importing data + preprocessing (NEW V5 - Pure Déduction)...')
train_obs, val_obs, test_obs, \
    train_non_obs, val_non_obs, test_non_obs = importdata(path_training, path_validation, path_test, window_size)
                                                        
print('Preprocessed data imported')

# %% -------Hyperparameters-------
tuned_initial_exp = 1    # 0 = yes, 1 = no
nb_quarters_per_episode = 96
replay_buffer_max_length = int(2 * 1e4)
batch_size = 256
num_epochs = 20

gamma = np.float64(0.99)
tau = 1e-2
actor_lr = 1e-4
critic_lr = 1e-3

actor_optimizer = tf.keras.optimizers.legacy.Adam(actor_lr, clipnorm=1)
critic_optimizer = tf.keras.optimizers.legacy.Adam(critic_lr, clipnorm=1)

actor_update_T = 4
target_update_T = 4

hidden_actor = (500, 500)
hidden_critic = (500, 500)

# TIMELINE TRACKING INTERVALS
log_interval = 2400       
eval_interval = 9600     

# ------- Print & Export Hyperparameters Summary -------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join(os.getcwd(), f'Run_TD3_{timestamp}_EARLY_STOP_WDW8_S{seed}_TEST')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config_path = os.path.join(save_dir, "hyperparameters.txt")
with open(config_path, "w", encoding="utf-8") as f:
    f.write(f"{' Category ':*^40}\n")
    f.write(f"  - Algorithm                  : TD3 (Logic NEW V5 - Pure Deduction)\n")
    f.write(f"  - Window size (Past History) : {window_size}\n")
    f.write(f"  - Global Epochs              : {num_epochs}\n")
    f.write(f"  - Batch size                  : {batch_size}\n")
    f.write(f"  - Replay buffer max length   : {replay_buffer_max_length}\n")
    f.write(f"  - Gamma (Discount factor)    : {gamma}\n")
    f.write(f"  - Tau (Soft update rate)      : {tau}\n")
    f.write(f"  - Actor LR                    : {actor_lr}\n")
    f.write(f"  - Critic LR                   : {critic_lr}\n")
    f.write(f"  - Hidden layers (Actor)      : {hidden_actor}\n")
    f.write(f"  - Hidden layers (Critic)     : {hidden_critic}\n")
    f.write(f"\n{'*'*40}\n")

print(f"Dossier créé et hyperparamètres sauvés : {save_dir}")

# %% ------- Training setup variables -------
actor_activ = tf.nn.relu
critic_activ = tf.nn.relu
noise_stddev = 0.2
initial_collect_steps = batch_size * 2
collect_steps_per_iteration = 4

num_iterations = int(num_epochs * int((len(train_obs) - initial_collect_steps) / collect_steps_per_iteration))
num_training_episodes = int(num_iterations / nb_quarters_per_episode)
num_val_episodes = int(len(val_obs) / nb_quarters_per_episode)

# %% -------Environments-------
starting_time = time.time()

train_env = Environment(train_obs, train_non_obs, battery_max_power,
                        1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                        battery_replacement_cost, penalty=penal) 
tf_train_env = tf_py_environment.TFPyEnvironment(train_env)

validation_env = Environmentvalidation(val_obs, val_non_obs, battery_max_power,
                                 1, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
                                 battery_replacement_cost) 
tf_validation_env = tf_py_environment.TFPyEnvironment(validation_env)

# %% -------Actor and critic-------
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

# --- INSTANTIATE LOGGING CHECKPOINTER MANAGER ---
checkpoint_dir = os.path.join(save_dir, 'checkpoint_complet')
train_checkpointer = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    global_step=train_step_counter
)

# --- ANTI-CRASH MEMORY SAFEGUARD N°2: Policy Action Compilation Interface ---
class CompiledPolicy:
    def __init__(self, policy):
        self._action_fn = common.function(policy.action)
    def action(self, time_step):
        return self._action_fn(time_step)

compiled_collect_policy = CompiledPolicy(agent.collect_policy)
tf_policy_action = common.function(agent.policy.action)
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())

# %% -------Replay buffer-------
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_train_env.batch_size,
    max_length=replay_buffer_max_length)


# %% ------- Synchronized Manual Evaluation Loop Interface (RAM Safe) -------
def evaluate_agent(tf_env, num_episodes, nb_quarters):
    py_env = tf_env.pyenv.envs[0]
    
    py_env._observation_index = 0
    py_env._episode = 0
    py_env._observation_samples[0, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[0]
    time_step = tf_env.reset()

    total_profit = 0.0

    for ep in range(num_episodes):
        episode_profit = 0.0
        for _ in range(nb_quarters):
            if time_step.is_last():
                time_step = tf_env.reset()
                
            action_step = tf_policy_action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_profit += time_step.reward.numpy()[0]
            
        total_profit += episode_profit
        
        # --- ANTI-CRASH MEMORY SAFEGUARD N°3: Flush Python RAM every 50 episodes days ---
        if (ep + 1) % 50 == 0:
            gc.collect() 
            
    return total_profit / num_episodes

# %% -------Initial fill-------
start = datetime.now()
print("\nData collection ... \n")
collect_data(tf_train_env, random_policy, replay_buffer, initial_collect_steps, tuned_initial_exp)
dataset = replay_buffer.as_dataset(num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2).shuffle(
    buffer_size=replay_buffer_max_length).prefetch(3)

iterator = iter(dataset)

# %% ------- ANTI-CRASH MEMORY SAFEGUARD N°4: Graph-Compiled Training Function Pass -------
agent.train = common.function(agent.train)

@common.function
def train_step_fn():
    # Enclosing batch tracking passes inside static C++ graphs locks down memory leaks pipelines
    experience, _ = next(iterator)
    return agent.train(experience).loss

agent.train_step_counter.assign(0)

print("\nPOLICY EVALUATION BEFORE TRAINING \n")
avg_return_initial = evaluate_agent(tf_validation_env, num_val_episodes, nb_quarters_per_episode)
print(f"Average return (before training): {avg_return_initial:.2f} € / day")

returns = [avg_return_initial]
train_rewards = []
average_train_rewards = []
train_loss_plot = []

best_avg_return = avg_return_initial
saver = policy_saver.PolicySaver(agent.policy) 

# --- CONVERGENCE PATIENCE PARAMETERS SETUP FOR EARLY STOPPING PASS ---
patience = 15              # Allowed sequential evaluation evaluations checks without updates
wait_counter = 0           
min_delta = 0.5            
early_stop_triggered = False 
# ----------------------------------------------------

# %% ------- TRAINING LOOP -------
print("\n START TRAINING ... \n")

for ep in tqdm(range(num_training_episodes), desc="Training Episodes", unit="ep"):

    for _ in range(nb_quarters_per_episode):

        # 1. Trajectory transitions data collection (Queries graph-compiled operational policy network)
        rew, act = collect_data(tf_train_env, compiled_collect_policy, replay_buffer, collect_steps_per_iteration, 1)
        
        if tf.is_tensor(rew):
            train_rewards.append(rew.numpy()[0])
        else:
            train_rewards.append(rew)

        # 2. Execute gradient step pass updates (Direct background C++ graph call invocation)
        train_loss = train_step_fn()
        train_loss_plot.append(train_loss.numpy())

        # 3. Step incrementation
        step = agent.train_step_counter.numpy()

        # CONSOLE CONVERGENCE LOGS OUTPUT PIPELINE
        if step % log_interval == 0:
            total_steps_collected = log_interval * collect_steps_per_iteration
            profit_per_step = np.sum(train_rewards) / total_steps_collected
            avg_train_per_day = profit_per_step * nb_quarters_per_episode
            
            average_train_rewards.append(avg_train_per_day)
            print(f"Step {step:05d} | Loss Critic: {train_loss.numpy():>8.2f} | Avg Reward (Train) : {avg_train_per_day:>8.2f} € / day")
            train_rewards = []
            
            # Periodic background flushes during extended processing runtimes
            gc.collect()

        # PERIODIC SYNCHRONIZED VALIDATION HORIZON CYCLE RUNS
        if step % eval_interval == 0:
            avg_return = evaluate_agent(tf_validation_env, num_val_episodes, nb_quarters_per_episode)
            returns.append(avg_return)
            
            print("\n"+"-"*60)
            print(f"EVALUATION (Step {step:05d}) | Validation Return (full): {avg_return:>8.2f} € / day")

            # --- PROCESS CHECKPOINT WEIGHTS FOR DEFENSIVE EARLY STOPPING CRITERIA ---
            if avg_return > (best_avg_return + min_delta):
                print(f" -> AMÉLIORATION : {avg_return:.2f} > {best_avg_return:.2f} + {min_delta}")
                best_avg_return = avg_return
                wait_counter = 0  
                
                policy_dir = os.path.join(save_dir, "saved_policy_final")
                if os.path.exists(policy_dir):
                    shutil.rmtree(policy_dir) # Strict clearance sequence matching Windows filesystem lock constraints
                
                saver.save(policy_dir)
                train_checkpointer.save(global_step=agent.train_step_counter)
                print(" -> New best policy AND CHECKPOINT saved.")
            else:
                wait_counter += 1
                print(f" -> Pas d'amélioration suffisante. Patience : {wait_counter} / {patience}")
                
            print("-" * 60)

            # Trigger early stopping sequence execution break pass if stagnation bounds are exceeded
            if wait_counter >= patience:
                print(f"\n[!!!] EARLY STOPPING TRIGGERED [!!!]")
                print(f"L'agent TD3 ne progresse plus. Arrêt à l'itération {step} pour éviter l'overfitting.")
                early_stop_triggered = True
                break 
                
    if early_stop_triggered:
        break 

# %% -------End of training time-------
end = datetime.now()
td = (end - start).total_seconds()
nb_hours = td / 3600

if early_stop_triggered:
    print(f"\n[INFO] Entraînement écourté par l'Early Stopping.")
    
print(f"\nThe time of execution is : {td:.03f} s ({nb_hours:.03f} h)")

# %%=============================================================================
# HISTORICAL TRAINING PROFILE GENERATION LOGS (MULTI-SHEET EXCEL EXPORT)
# =============================================================================
print("\nExportation des données d'entraînement vers Excel...")

df_val = pd.DataFrame({
    'Environment_Steps': np.arange(len(returns)) * eval_interval,
    'Validation_Reward_Eur_Jour': returns
})

df_train = pd.DataFrame({
    'Environment_Steps': (np.arange(len(average_train_rewards)) + 1) * log_interval,
    'Train_Reward_Eur_Jour': average_train_rewards
})

df_loss = pd.DataFrame({
    'TD3_Iterations': np.arange(len(train_loss_plot)),
    'TD3_Critic_Loss': train_loss_plot
})

excel_path = os.path.join(save_dir, "Historique_Entrainement_TD3.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df_val.to_excel(writer, sheet_name='Validation_Reward', index=False)
    df_train.to_excel(writer, sheet_name='Train_Reward', index=False)
    df_loss.to_excel(writer, sheet_name='TD3_Loss', index=False)

print(f"-> Données brutes sauvegardées avec succès dans : {excel_path}")

# %%=============================================================================
# PLOTTING SEPARATED CHARTS INTERFACE (ENGLISH CODES HOOK)
# =============================================================================
print("\nGenerating and saving plots...")

def moving_average(a, n=10):
    if len(a) < n: return np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# 1. Training Returns Layout Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
steps_train = np.arange(len(average_train_rewards)) * log_interval
ax1.plot(steps_train, average_train_rewards, color='blue', linewidth=1.5, label='Train (Raw)')
ax1.set_title("Training Profit Evolution (TD3)", fontsize=14)
ax1.set_xlabel("Environment Steps")
ax1.set_ylabel("Profit (€ / day)")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
fig1.savefig(os.path.join(save_dir, "1_Train_Reward.png"), dpi=300)

# 2. Full Validation Horizon Performance Return Layout curve
fig2, ax2 = plt.subplots(figsize=(10, 6))
steps_val = np.arange(len(returns)) * eval_interval
ax2.plot(steps_val, returns, marker='o', color='red', linewidth=2, label='Validation (Full Dataset)')
ax2.set_title("Evaluation Performance on Validation Set (TD3)", fontsize=14)
ax2.set_xlabel("Environment Steps")
ax2.set_ylabel("Average Profit (€ / day)")
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_dir, "2_Validation_Reward.png"), dpi=300)

# 3. Critic Loss Network Convergence curves (Includes moving average smoothing layer filter)
fig3, ax3 = plt.subplots(figsize=(10, 6))

ax3.plot(train_loss_plot, color='orange', alpha=0.3, linewidth=1, label='Loss (Raw)')
lissage = min(500, len(train_loss_plot)) 
if lissage > 1:
    loss_smoothed = moving_average(train_loss_plot, n=lissage)
    ax3.plot(np.arange(lissage-1, len(train_loss_plot)), loss_smoothed, color='darkorange', linewidth=2, label='Loss (Smoothed)')

ax3.set_title("Network Convergence (TD3 Critic Loss)", fontsize=14)
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Loss")
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
plt.tight_layout()
fig3.savefig(os.path.join(save_dir, "3_Loss.png"), dpi=300)

# --- ANTI-CRASH MEMORY SAFEGUARD N°5: Formal interface close releases window memory ---
plt.close('all') 
print(f"\nFin d'exécution. Les 3 graphiques ont été sauvegardés dans : {save_dir}")
