# -*- coding: utf-8 -*-
"""
Evaluation Script for TD3 Agent - OLD MODEL 
- Dynamic de-normalization (inverse_transform)
- Price Maker logic based on 13 steps (Merit-Order)
- Full dataset evaluation

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
import pandas as pd

# Import classes for the old model
from Classes_TD3_old_final import Environmentvalidation, importdata

# %%=============================================================================
# 1. TARGET FOLDER
# =============================================================================
# /!\ REPLACE WITH YOUR ACTUAL OLD ADAPTED RUN FOLDER /!\
DOSSIER_RUN = "Run_20260513_220711_FINAL_OPTI_S2"  
DOSSIER_AGENT = os.path.join(DOSSIER_RUN, "saved_policy_final")
TXT_PATH = os.path.join(DOSSIER_RUN, "hyperparameters.txt")

# %% =============================================================================
# 2. AUTOMATIC HYPERPARAMETER READING (.txt)
# =============================================================================
print("Reading hyperparameters from text file...\n")
hyperparams = {}

# Dynamically parse hyperparameters from the text metadata file
if os.path.exists(TXT_PATH):
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].replace("-", "").strip()
                value = parts[1].strip()
                hyperparams[key] = value

# --- 1. nb_quarters_per_episode ---
key_quarters = "Quarters per episode (24h)"
if key_quarters in hyperparams:
    nb_quarters_per_episode = int(hyperparams[key_quarters])
else:
    nb_quarters_per_episode = 96

# --- 2. LA_steps (Old model) ---
key_la = "Look-Ahead Steps (LA_steps)"
if key_la in hyperparams:
    LA_steps = int(hyperparams[key_la])
    print(f" {key_la} = {LA_steps} (Read from file)")
else:
    LA_steps = 8
    print(f" {key_la} = {LA_steps} (DEFAULT VALUE USED)")  

# %%=============================================================================
# 3. REMAINING PHYSICAL PARAMETERS
# ==============================================================================
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

#%% =============================================================================
# 4. DATA IMPORT AND FUSION (Old Logic)
# =============================================================================
path_training = 'Frame_1819_training.xlsx'
path_validation = 'Frame_1819_validation.xlsx'
path_test = 'Frame_1819_test.xlsx'

print("\nImporting data and applying global normalization (Old Model)...")
index_col_si, index_first_col_MDP, scaler_si, scaler_list, \
train_obs, val_obs, test_obs, \
train_non_obs, val_non_obs, test_non_obs = importdata(
    path_training, path_validation, path_test, LA_steps
)

print("\nSelecting TEST dataset for evaluation...")
eval_obs = test_obs
eval_non_obs = test_non_obs
print(f"Final evaluation dataset size: {eval_obs.shape[0]} time steps.")

# %% =============================================================================
# 5. ENVIRONMENT CREATION AND AI LOADING
# =============================================================================
print("\nCreating Validation environment (Old Model)...")
env_eval = Environmentvalidation(
    eval_obs, eval_non_obs, scaler_si, scaler_list, battery_max_power,
    1.0, nb_quarters_per_episode, index_first_col_MDP, EP_ratio, roundtrip_efficiency,
    battery_replacement_cost
)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

print(f"Loading agent's brain from: {DOSSIER_AGENT}...")
saved_policy = tf.compat.v2.saved_model.load(DOSSIER_AGENT)

# %% =============================================================================
# 6. DETAILED SIMULATION OF SPECIFIC DAYS (Search by DATE - Old Model)
# =============================================================================
print("\nLoading dates for search (Test set only)...")
df_dates = pd.read_excel(path_test, usecols=['FROM_DATE'])

jours_a_simuler = ['2019-06-21', '2019-06-22', '2019-06-23','2018-11-21','2018-11-22']

print(f"\nDetailed simulation for the following days: {jours_a_simuler}...")

for date_str in jours_a_simuler:
    print(f" -> Generating plot for day {date_str}...")
    
    lignes_date = df_dates[df_dates['FROM_DATE'].astype(str).str.startswith(date_str)]
    if lignes_date.empty:
        print(f"   Date {date_str} not found. Skipping.")
        continue
        
    target_index = lignes_date.index[0]
    
    py_env = tf_env_eval.pyenv.envs[0]
    py_env._observation_index = target_index
    py_env._episode = target_index // nb_quarters_per_episode
    
    # Absolute safety: Force SoC to 0.5 in the matrix before environment reset
    py_env._observation_samples[target_index, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[target_index]
    time_step = tf_env_eval.reset()
    
    # Init metrics tracking lists
    actions_list, soc_list, profit_cumul = [], [], []
    price_taker_list, price_maker_list = [], []
    si_initial_list, si_modified_list = [], []
    current_profit = 0.0
    
    EP_ratio = py_env._EP_ratio
    eta = py_env._eta
    max_power = py_env._max_power
    col_MDP = py_env._col_MDP

    # 96 steps continuous execution simulation loop
    for _ in range(nb_quarters_per_episode):
        idx = py_env._observation_index
        obs = py_env._observation
        soc = obs[py_env._SOC_index] 
        
        # --- PRICE TAKER Calculation (Before asset impact) ---
        si_norm = py_env._non_observable_samples[idx]
        si_norm_val = si_norm[0] if isinstance(si_norm, np.ndarray) and len(si_norm.shape) > 0 else si_norm
        
        syst_imb = py_env._scaler_si.inverse_transform(np.array([[si_norm_val]], dtype=np.float32))[0, 0]
        si_initial_list.append(syst_imb)

        base_index = (-syst_imb / 100) + 6
        base_index = int(np.floor(base_index)) if base_index < 6 else int(np.ceil(base_index))
        base_index = int(np.clip(base_index, 0, 12))
        
        prix_norm_taker = obs[col_MDP + base_index - 1]
        price_taker = py_env._scalers_MP[base_index].inverse_transform(np.array([[prix_norm_taker]], dtype=np.float32))[0, 0]
        price_taker_list.append(price_taker)
        
        # --- TD3 Policy Network Action Inference ---
        action_step = saved_policy.action(time_step)
        raw_action = action_step.action.numpy()[0]
        actions_list.append(raw_action)
        soc_list.append(soc)
        
        # --- Physical Interface Impact & PRICE MAKER Logic ---
        new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        fraction_of_max_power = (new_soc - soc) * (4 * EP_ratio)
        battery_charge_MW = fraction_of_max_power * max_power
        
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(eta)
            
        real_si = syst_imb - network_charge_MW
        si_modified_list.append(real_si)
        
        maker_index = (-real_si / 100) + 6
        maker_index = int(np.floor(maker_index)) if maker_index < 6 else int(np.ceil(maker_index))
        maker_index = int(np.clip(maker_index, 0, 12))
        
        prix_norm_maker = obs[col_MDP + maker_index - 1]
        price_maker = py_env._scalers_MP[maker_index].inverse_transform(np.array([[prix_norm_maker]], dtype=np.float32))[0, 0]
        price_maker_list.append(price_maker)
        
        # Environment stepping logic
        time_step = tf_env_eval.step(action_step.action)
        current_profit += time_step.reward.numpy()[0]
        profit_cumul.append(current_profit)
        
    # PLOTTING INTRA-DAY PROFILES
    fig, (ax_price, ax_action_si, ax_soc, ax_profit) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x_axis = np.arange(nb_quarters_per_episode) / 4
    
    ax_price.plot(x_axis, price_taker_list, color='gray', linestyle='--', label='Price Taker (Base)', alpha=0.7)
    ax_price.plot(x_axis, price_maker_list, color='purple', linewidth=2, label='Price Maker (Impacted)')
    ax_price.set_ylabel("Price (€/MWh)")
    ax_price.set_title(f"Market Price Impact - {date_str} (TD3 OLD)")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.5)
    
    ax_action_si.plot(x_axis, si_initial_list, color='black', linestyle='--', label='Initial SI', alpha=0.4)
    ax_action_si.plot(x_axis, si_modified_list, color='blue', label='Modified SI', linewidth=1.5, alpha=0.8)
    ax_action_si.axhline(0, color='black', linewidth=0.8)
    ax_action_si.set_ylabel("SI (MW)", color='blue')
    ax_action_si.tick_params(axis='y', labelcolor='blue')
    
    ax_action_twin = ax_action_si.twinx()
    ax_action_twin.bar(x_axis, actions_list, width=0.15, color=['red' if a < 0 else 'green' for a in actions_list], alpha=0.4, label='Agent Action')
    ax_action_twin.set_ylabel("Action (-1 to 1)", color='green')
    ax_action_twin.set_ylim(-1.1, 1.1)
    ax_action_twin.tick_params(axis='y', labelcolor='green')
    ax_action_si.set_title("Agent Decisions vs Grid Balance")
    
    ax_soc.plot(x_axis, soc_list, color='blue', linewidth=2)
    ax_soc.set_ylabel("SoC (0 to 1)")
    ax_soc.set_ylim(-0.1, 1.1)
    ax_soc.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax_soc.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax_soc.set_title("Battery Level (SoC)")
    ax_soc.grid(True, alpha=0.5)
    
    ax_profit.plot(x_axis, profit_cumul, color='goldenrod', linewidth=2)
    ax_profit.set_ylabel("Cumulative Profit (€)")
    ax_profit.set_xlabel("Time (Hours)")
    ax_profit.set_title("Daily Profitability")
    ax_profit.grid(True, alpha=0.5)
    
    plt.tight_layout()
    chemin_sauvegarde = os.path.join(DOSSIER_RUN, f"Strategy_Day_{date_str}_TD3_OLD.png")
    fig.savefig(chemin_sauvegarde, dpi=300)
    plt.show()

print("\nResetting environment to launch full annual evaluation...")
time_step = tf_env_eval.reset()

# %%=============================================================================
# 7. EVALUATION ON FULL DATASET (SYNCHRONIZED)
# =============================================================================
num_episodes = int(len(eval_obs) / nb_quarters_per_episode)
print(f"\nLaunching simulation over {num_episodes} days (Episodes)...")

yearly_profit_cumul = []
daily_profits = []
total_profit = 0.0

# Absolute reset sequence execution before benchmark tracking loop
py_env = tf_env_eval.pyenv.envs[0]
py_env._observation_index = 0
py_env._episode = 0
py_env._observation_samples[0, py_env._SOC_index] = 0.5
py_env._observation = py_env._observation_samples[0]
time_step = tf_env_eval.reset()

# Loop over validation days tracks
for ep in range(num_episodes):
    episode_profit = 0.0
    
    # Strict 96 steps execution block per day
    for step in range(nb_quarters_per_episode):
        
        # TF-AGENTS session management: restart if the episode is terminated
        if time_step.is_last():
            time_step = tf_env_eval.reset()
        
        action_step = saved_policy.action(time_step)
        time_step = tf_env_eval.step(action_step.action)
        episode_profit += time_step.reward.numpy()[0]
        
    total_profit += episode_profit
    yearly_profit_cumul.append(total_profit)
    daily_profits.append(episode_profit)
    
    if (ep + 1) % 10 == 0:
        print(f"  Day {ep + 1:03d}/{num_episodes} | Cumulative Profit: {total_profit:>10.2f} €")

print("\n" + "="*50)
print(f"FINAL GLOBAL RESULT: {total_profit:,.2f} €")
print(f"Daily Average: {total_profit / num_episodes:,.2f} € / day")
print("="*50)

# --- SAVE RESULTS TO TXT PERFORMANCE REPORT --- #
chemin_resultats = os.path.join(DOSSIER_RUN, "Test_Evaluation_Results.txt") 
with open(chemin_resultats, "w", encoding="utf-8") as f: 
    f.write("="*50 + "\n") 
    f.write(" FINAL EVALUATION REPORT ON TEST SET \n") 
    f.write("="*50 + "\n") 
    f.write(f"Number of tested days: {num_episodes} days\n") 
    f.write(f"Total Cumulative Profit: {total_profit:,.2f} €\n") 
    f.write(f"Daily Average: {total_profit / num_episodes:,.2f} € / day\n") 
    f.write("="*50 + "\n") 
print(f"\n -> Financial results successfully saved in: {chemin_resultats}")

# %%=============================================================================
# 8. RESULTS GRAPH
# =============================================================================
print("\nGenerating final performance graph...")

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(range(num_episodes), yearly_profit_cumul, color='forestgreen', linewidth=2.5)
ax1.fill_between(range(num_episodes), yearly_profit_cumul, color='lightgreen', alpha=0.3)
ax1.set_title("Cumulative Financial Performance on Dataset", fontsize=14, fontweight='bold')
ax1.set_xlabel("Days")
ax1.set_ylabel("Total Profit (€)")
ax1.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()

plot_path = os.path.join(DOSSIER_RUN, "Global_Results_Old.png")
fig.savefig(plot_path, dpi=300)
print(f"Graph saved to: {plot_path}")

plt.show()
