# -*- coding: utf-8 -*-
"""
Evaluation Script - PPO Agent NEW V5 (Window / Escalier NRV)
- English Plots
- Removed Strategic Zoom
- Full dataset synchronized evaluation (Midnight Anchor)

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_agents.environments import tf_py_environment
import pandas as pd

# Import the specialized causal validation environment classes
from Classes_TD3_without_LA_without_stairs import Environmentvalidation, importdata

# %%=============================================================================
# 1. TARGET FOLDER AND HYPERPARAMETERS
# =============================================================================

# /!\ REPLACE WITH YOUR ACTUAL NEW PPO V5 RUN FOLDER NAME /!\
DOSSIER_RUN = "Run_PPO_20260417_231242_EARLY_STOP_WDW8_S1"  

DOSSIER_AGENT = os.path.join(DOSSIER_RUN, "saved_policy_final")

# Fixed simulation profile parameters configurations
nb_quarters_per_episode = 96
window_size = 8 # Operational historical rolling window structure

battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

#%% =============================================================================
# 2. DATA IMPORT AND PREPARATION (NEW V5)
# =============================================================================
path_training = 'Dataset_Train_2018_2019_wdw8.xlsx'
path_validation = 'Dataset_Val_2018_2019_wdw8.xlsx'
path_test = 'Dataset_Test_2018_2019_wdw8.xlsx'

print("\nImporting data (NEW V5)...")
# Pure operational parsing (Omits look-ahead features and global scalers vectors)
train_obs, val_obs, test_obs, \
    train_non_obs, val_non_obs, test_non_obs = importdata(
        path_training, path_validation, path_test, window_size
)

# Parse historical calendar timestamps columns and cast to datetime format types
df_dates = pd.read_excel(path_test, usecols=['FROM_DATE'])
df_dates['FROM_DATE'] = pd.to_datetime(df_dates['FROM_DATE'])

# %% =============================================================================
# 3. ENVIRONMENT CREATION AND PPO BRAIN LOADING
# =============================================================================
print("\nCreating Validation Environment (PPO NEW V5)...")
# The V5 environment structure maps parameters directly without external scalers arguments
env_test = Environmentvalidation(
    test_obs, test_non_obs, battery_max_power,
    1.0, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
    battery_replacement_cost
)
tf_environment_eval = tf_py_environment.TFPyEnvironment(env_test)

print(f"Loading PPO agent's brain from: {DOSSIER_AGENT}...")
saved_policy = tf.compat.v2.saved_model.load(DOSSIER_AGENT)

# %% =============================================================================
# 4. DETAILED SIMULATION OF SPECIFIC DAYS (Search by DATE)
# =============================================================================
print("\nLoading dates for specific day search...")

jours_a_simuler = ['2019-06-21', '2019-06-22', '2019-06-23','2018-11-21','2018-11-22']

print(f"\nDetailed simulation for the following days: {jours_a_simuler}...")

for date_str in jours_a_simuler:
    print(f" -> Generating plot for day {date_str}...")
    
    # Locate row indices matching the first quarter-hour interval of the targeted day
    lignes_date = df_dates[df_dates['FROM_DATE'].dt.strftime('%Y-%m-%d') == date_str]
    if lignes_date.empty:
        print(f"   Date {date_str} not found. Skipping.")
        continue
        
    # Recalibrate Python environment row alignment offset by subtracting window_size
    target_index = lignes_date.index[0] - window_size
    if target_index < 0: 
        continue
    
    # Align validation trackers parameters to the target row index
    py_env = tf_environment_eval.pyenv.envs[0]
    py_env._observation_index = target_index
    py_env._episode = (target_index // nb_quarters_per_episode) + 1 
    
    # Enforce initial mid-point capacity SoC state to isolate performance characteristics
    py_env._observation_samples[target_index, py_env._SOC_index] = 0.5 
    py_env._observation = py_env._observation_samples[target_index] 
    time_step = tf_environment_eval.reset()
        
    # Pre-allocate profile parameters logs containers lists
    actions_list, soc_list, profit_cumul = [], [], []
    price_taker_list, price_maker_list = [], []
    nrv_initial_list, nrv_modified_list = [], []
    current_profit = 0.0
    
    EP_ratio = py_env._EP_ratio
    eta = py_env._eta
    max_power = py_env._max_power
    
    # INITIALIZE STOCHASTIC RECURRENT POLICY HIDDEN STATE CELL VECTORS
    policy_state = saved_policy.get_initial_state(tf_environment_eval.batch_size)
    
    # EXECUTE DETAILED 96 QUARTERS TIMELINE SIMULATION PIPELINE LOOP
    for _ in range(nb_quarters_per_episode):
        idx = py_env._observation_index
        soc = time_step.observation.numpy()[0, 0]
        
        # --- PARSE UNREGULATED UNMODIFIED REWARD SIGNALS (0 = NRV Volume, 1 = Clearing Price) ---
        base_nrv = py_env._non_observable_samples[idx, 0]
        base_price = py_env._non_observable_samples[idx, 1]
        
        nrv_initial_list.append(base_nrv)
        price_taker_list.append(base_price)
        
        # --- INFER STOCHASTIC POLICY ACTION PASSING RNN STATUS TENSORS ---
        action_step = saved_policy.action(time_step, policy_state)
        policy_state = action_step.state 
        raw_action = action_step.action.numpy()[0]
        
        actions_list.append(raw_action)
        soc_list.append(soc)
        
        # --- PHYSICAL SYSTEM IMPACT & PRICE MAKER CAUSAL LOOP IMPLEMENTATION ---
        new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        fraction_of_max_power = (new_soc - soc) * (4 * EP_ratio)
        battery_charge_MW = fraction_of_max_power * max_power
        
        if battery_charge_MW > 0: 
            network_charge_MW = battery_charge_MW / np.sqrt(eta)
        else: 
            network_charge_MW = battery_charge_MW * np.sqrt(eta)
            
        modified_nrv = base_nrv + network_charge_MW
        nrv_modified_list.append(modified_nrv)
        
        # Map modified volume onto the discrete clearing price merit-order step indicators
        idx_mod = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_mod)) if idx_mod < 6 else int(np.ceil(idx_mod))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        # Causal price mapping extracted directly from the hidden reward matrix index bounds
        realized_price = py_env._non_observable_samples[idx, 2 + idx_mod]
        price_maker_list.append(realized_price)
        
        # Advance environment state tracking pass
        time_step = tf_environment_eval.step(action_step.action)
        current_profit += time_step.reward.numpy()[0]
        profit_cumul.append(current_profit)
        
    # 6. CHART LAYOUT GENERATION PIPELINE (ENGLISH PLOTS V5)
    fig, (ax_price, ax_action_nrv, ax_soc, ax_profit) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x_axis = np.arange(nb_quarters_per_episode) / 4
    
    ax_price.plot(x_axis, price_taker_list, color='gray', linestyle='--', label='Price Taker (Base)', alpha=0.7)
    ax_price.plot(x_axis, price_maker_list, color='purple', linewidth=2, label='Price Maker (Impacted)')
    ax_price.set_ylabel("Price (€/MWh)")
    ax_price.set_title(f"Market Price Impact - {date_str} (PPO NEW V5)")
    ax_price.legend(loc="upper left")
    ax_price.grid(True, alpha=0.5)
    
    ax_action_nrv.plot(x_axis, nrv_initial_list, color='black', linestyle='--', label='Initial NRV', alpha=0.4)
    ax_action_nrv.plot(x_axis, nrv_modified_list, color='blue', label='Modified NRV', linewidth=1.5, alpha=0.8)
    ax_action_nrv.axhline(0, color='black', linewidth=0.8)
    ax_action_nrv.set_ylabel("NRV (MW)", color='blue')
    ax_action_nrv.tick_params(axis='y', labelcolor='blue')
    
    ax_action_twin = ax_action_nrv.twinx()
    ax_action_twin.bar(x_axis, actions_list, width=0.15, color=['red' if a < 0 else 'green' for a in actions_list], alpha=0.4, label='Agent Action')
    ax_action_twin.set_ylabel("Action (-1 to 1)", color='green')
    ax_action_twin.set_ylim(-1.1, 1.1)
    ax_action_twin.tick_params(axis='y', labelcolor='green')
    ax_action_nrv.set_title("Agent Decisions vs Grid Balance (NRV)")
    
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
    chemin_sauvegarde = os.path.join(DOSSIER_RUN, f"Strategy_Day_{date_str}_PPO_NEW_V5.png")
    fig.savefig(chemin_sauvegarde, dpi=300)
    plt.show()

# %%=============================================================================
# 5. FULL DATASET GLOBAL EVALUATION (SYNCHRONIZED - MIDNIGHT ANCHOR MECHANISM)
# =============================================================================

# Parse chronology dataframe rows matching exactly midnight hours structures (00:00:00)
index_minuits = df_dates[(df_dates['FROM_DATE'].dt.hour == 0) & (df_dates['FROM_DATE'].dt.minute == 0)].index

# Shift indices back based on historical rolling window limits to lock down absolute day boundaries
index_debuts_jours = index_minuits - window_size

num_episodes = len(index_debuts_jours)
print(f"\nLaunching annual simulation over {num_episodes} exact real days...")

yearly_profit_cumul = []
daily_profits = []
total_profit = 0.0

py_env = tf_environment_eval.pyenv.envs[0] 

# Iterate exclusively over the 96 pristine chronological midnight anchor index points
for ep, start_idx in enumerate(index_debuts_jours):
    episode_profit = 0.0
    
    # --- JUMP ENTIRELY TO THE TARGET MIDNIGHT INDEX (Completely avoids leaking rolling data overlaps) ---
    py_env._observation_index = start_idx
    py_env._episode = ep 
    py_env._observation_samples[start_idx, py_env._SOC_index] = 0.5 # Safety initialization mid-point SoC override
    py_env._observation = py_env._observation_samples[start_idx]
    time_step = tf_environment_eval.reset()
    
    # Reinitialize stochastic hidden recurrent state parameters cell blocks at the start of each day
    policy_state = saved_policy.get_initial_state(tf_environment_eval.batch_size)
    
    for step in range(nb_quarters_per_episode):
        
        # Infer policy decisions output sequence tracking dynamic RNN hidden state channels
        action_step = saved_policy.action(time_step, policy_state)
        policy_state = action_step.state
            
        # Stepping physical interface limits structures
        time_step = tf_environment_eval.step(action_step.action)
        episode_profit += time_step.reward.numpy()[0]
        
    total_profit += episode_profit
    yearly_profit_cumul.append(total_profit)
    daily_profits.append(episode_profit)
    
    if (ep + 1) % 10 == 0:
        print(f"  Day {ep + 1:03d}/{num_episodes} | Cumulative Profit: {total_profit:>10.2f} €")

print("\n" + "="*50)
print(f"FINAL GLOBAL RESULT PPO NEW V5: {total_profit:,.2f} €")
print(f"Daily Average: {total_profit / num_episodes:,.2f} € / day")
print("="*50)

# --- SAVE OUTPUT TEXT RESULTS PERFORMANCE PASS REPORT --- #
chemin_resultats = os.path.join(DOSSIER_RUN, "Test_Evaluation_Results_V5.txt") 
with open(chemin_resultats, "w", encoding="utf-8") as f: 
    f.write("="*50 + "\n") 
    f.write(" FINAL EVALUATION REPORT ON TEST SET (PPO NEW V5) \n") 
    f.write("="*50 + "\n") 
    f.write(f"Number of tested days: {num_episodes} days\n") 
    f.write(f"Total Cumulative Profit: {total_profit:,.2f} €\n") 
    f.write(f"Daily Average: {total_profit / num_episodes:,.2f} € / day\n") 
    f.write("="*50 + "\n") 
print(f"\n -> Financial results successfully saved in: {chemin_resultats}")

# %%=============================================================================
# 6. RESULTS GRAPH (Single Continuous Cumulative Earnings Trend curve)
# =============================================================================
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(range(num_episodes), yearly_profit_cumul, color='forestgreen', linewidth=2.5)
ax1.fill_between(range(num_episodes), yearly_profit_cumul, color='lightgreen', alpha=0.3)
ax1.set_title("Cumulative Financial Performance on Dataset (PPO NEW V5)", fontsize=14, fontweight='bold')
ax1.set_xlabel("Days")
ax1.set_ylabel("Total Profit (€)")
ax1.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plot_path = os.path.join(DOSSIER_RUN, "Global_Results_PPO_NEW_V5.png")
fig.savefig(plot_path, dpi=300)
print(f"Graph saved to: {plot_path}")

plt.show()
