# -*- coding: utf-8 -*-
"""
Long-Term Evaluation Script - PPO Agent (V5 Logic) - 2020-2023
- Performance assessment on the long-term testing dataset tracks (Empirical Price Ladder 6h V3).
- Leverages the hidden recurrent state cells (RNN policy_state) of PPO, reset every midnight.
- Generates a multi-sheet spreadsheet tracking Daily, Monthly, and Annual financial returns.
- FIX 1: Eradication of ghost environment resets via the synchronized tracking index method.
- FIX 2: Logging of the physical effective dispatched action for precise SoC/Plotting synchronization.

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tf_agents.environments import tf_py_environment

# --- Import modules from original timeline structure (No price ladder) ---
from Classes_TD3_without_LA_without_stairs import Environmentvalidation, importdata

# %%=============================================================================
# 1. PARAMETERS AND TARGET DIRECTORIES
# =============================================================================
DOSSIER_RUN = r"C:\Users\pcstudentgele11\TFE_HUPET_ACHILLE\PPO_without_LA_without_stairs\Run_PPO_20260416_112607_EARLY_STOP_WDW8_S1"
DOSSIER_AGENT = os.path.join(DOSSIER_RUN, "saved_policy_final")

# Establish output result logging target folders
SAVE_DIR = "Resultats_Eval_PPO_2020_2023_SEED1_FINAL"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Dossier de sauvegarde des résultats créé : {SAVE_DIR}")

# Fixed hardware specifications and evaluation dimensions configurations
window_size = 8
nb_quarters_per_episode = 96
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# Input data sheets path mapping
path_training = 'Dataset_Train_2018_2019_wdw8.xlsx' 
path_validation = 'Dataset_Val_2018_2019_wdw8.xlsx' 
path_test = 'Dataset_2020_2023_Escalier_Empirique_Am_6h_V3_wdw8.xlsx' 

# %% =============================================================================
# 2. DATA IMPORT AND TIME TIMELINE SYNCHRONIZATION
# =============================================================================
print("\nImporting data (Calibrating scalers on 2018-2019, applying to 2020-2023)...")
_, _, test_obs, _, _, test_non_obs = importdata(
    path_training, path_validation, path_test, window_size
)

# Parse calendar timestamps columns from the long-term evaluation test sheet
df_dates = pd.read_excel(path_test, usecols=['FROM_DATE'])
df_dates['FROM_DATE'] = pd.to_datetime(df_dates['FROM_DATE'])

# %% =============================================================================
# 3. ENVIRONMENT CREATION AND MODEL INFERENCE PIPELINE
# =============================================================================
print("\nCreating Validation Environment...")
env_eval = Environmentvalidation(
    test_obs, test_non_obs, battery_max_power,
    1.0, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
    battery_replacement_cost
)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

print(f"Loading PPO agent's brain from: {DOSSIER_AGENT}...")
saved_policy = tf.compat.v2.saved_model.load(DOSSIER_AGENT)

# %% =============================================================================
# 4. DETAILED SIMULATION OF TARGET EVALUATION DAYS
# =============================================================================
jours_a_simuler = ['2021-01-08', '2022-08-24', '2022-12-12']
print(f"\nDetailed simulation for target days: {jours_a_simuler}...")

for date_str in jours_a_simuler:
    print(f" -> Generating plot for day {date_str}...")
    lignes_date = df_dates[df_dates['FROM_DATE'].dt.strftime('%Y-%m-%d') == date_str]
    if lignes_date.empty: 
        print(f"    Date {date_str} not found in dataset. Skipping.")
        continue
        
    idx_excel = lignes_date.index[0]
    target_index = idx_excel - window_size
    if target_index < 0: 
        continue

    py_env = tf_env_eval.pyenv.envs[0]
    py_env._observation_index = target_index
    
    # ALIGNMENT FIX: Calibrate processing steps boundaries onto 96-quarter day blocks
    py_env._episode = (target_index // nb_quarters_per_episode) + 1
    
    py_env._observation_samples[target_index, py_env._SOC_index] = 0.5 # Enforce initial mid-capacity SoC
    py_env._observation = py_env._observation_samples[target_index]
    time_step = tf_env_eval.reset()
    
    # Initialize stochastic hidden recurrent state parameters cell blocks
    policy_state = saved_policy.get_initial_state(tf_env_eval.batch_size)
    
    actions_list, soc_list, profit_cumul = [], [], []
    price_taker_list, price_maker_list = [], []
    nrv_initial_list, nrv_modified_list = [], []
    current_profit = 0.0
    
    for _ in range(nb_quarters_per_episode):
        idx = py_env._observation_index
        soc = py_env._observation[py_env._SOC_index]
        
        base_nrv = py_env._non_observable_samples[idx, 0]
        base_price = py_env._non_observable_samples[idx, 1]
        
        # Infer policy decisions output sequence tracking dynamic RNN hidden state channels
        action_step = saved_policy.action(time_step, policy_state)
        policy_state = action_step.state
        raw_action = action_step.action.numpy()[0]
        
        # --- COMPUTE THE REAL EFFECTIVE DISPATCHED ACTION (PHYSICAL HARDWARE CONSTRAINTS) ---
        new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        fraction_power = (new_soc - soc) * (4 * EP_ratio)
        bat_MW = fraction_power * battery_max_power
        net_MW = bat_MW / np.sqrt(roundtrip_efficiency) if bat_MW > 0 else bat_MW * np.sqrt(roundtrip_efficiency)
        
        modified_nrv = base_nrv + net_MW
        
        idx_raw = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_raw)) if idx_raw < 6 else int(np.ceil(idx_raw))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        realized_price = py_env._non_observable_samples[idx, 2 + idx_mod]
        
        # FIX: Append 'fraction_power' (true clipped action output) instead of raw network action
        actions_list.append(fraction_power)
        soc_list.append(soc)
        nrv_initial_list.append(base_nrv)
        nrv_modified_list.append(modified_nrv)
        price_taker_list.append(base_price)
        price_maker_list.append(realized_price)
        
        time_step = tf_env_eval.step(action_step.action)
        current_profit += time_step.reward.numpy()[0]
        profit_cumul.append(current_profit)

    # --- INTRA-DAY PROFILE PLOTTING ---
    fig, (ax_price, ax_action_si, ax_soc, ax_profit) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x_axis = np.arange(nb_quarters_per_episode) / 4
    
    ax_price.plot(x_axis, price_taker_list, color='gray', linestyle='--', label='Price Taker (Base)', alpha=0.7)
    ax_price.plot(x_axis, price_maker_list, color='purple', linewidth=2, label='Price Maker (Impacted)')
    ax_price.set_ylabel("Price (€/MWh)")
    ax_price.set_title(f"Market Price Impact - {date_str} (PPO V5)")
    ax_price.legend(loc="upper left"); ax_price.grid(True, alpha=0.5)
    
    ax_action_si.plot(x_axis, nrv_initial_list, color='black', linestyle='--', label='Initial NRV', alpha=0.4)
    ax_action_si.plot(x_axis, nrv_modified_list, color='blue', label='Modified NRV', linewidth=1.5, alpha=0.8)
    ax_action_si.axhline(0, color='black', linewidth=0.8)
    ax_action_si.set_ylabel("NRV (MW)", color='blue'); ax_action_si.tick_params(axis='y', labelcolor='blue')
    
    ax_action_twin = ax_action_si.twinx()
    ax_action_twin.bar(x_axis, actions_list, width=0.15, color=['red' if a < 0 else 'green' for a in actions_list], alpha=0.4, label='Agent Action')
    ax_action_twin.set_ylabel("Action (-1 to 1)", color='green'); ax_action_twin.set_ylim(-1.1, 1.1); ax_action_twin.tick_params(axis='y', labelcolor='green')
    ax_action_si.set_title("Agent Decisions vs Grid Balance")
    
    ax_soc.plot(x_axis, soc_list, color='blue', linewidth=2)
    ax_soc.set_ylabel("SoC (0 to 1)"); ax_soc.set_ylim(-0.1, 1.1)
    ax_soc.axhline(0, color='black', linestyle='--', alpha=0.5); ax_soc.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax_soc.set_title("Battery Level (SoC)"); ax_soc.grid(True, alpha=0.5)
    
    ax_profit.plot(x_axis, profit_cumul, color='goldenrod', linewidth=2)
    ax_profit.set_ylabel("Cumulative Profit (€)"); ax_profit.set_xlabel("Time (Hours)")
    ax_profit.set_title("Daily Profitability"); ax_profit.grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"Strategy_Day_{date_str}_PPO.png"), dpi=300)
    plt.close(fig) 

# %%=============================================================================
# 5. LONG-TERM MULTI-YEAR HORIZON EVALUATION (2020-2023)
# =============================================================================
print("\nLaunching Long-Term 2020-2023 Evaluation (Midnight Anchor)...")
index_minuits = df_dates[(df_dates['FROM_DATE'].dt.hour == 0) & (df_dates['FROM_DATE'].dt.minute == 0)].index
index_debuts_jours = [i - window_size for i in index_minuits if (i - window_size) >= 0]

num_episodes = len(index_debuts_jours)
total_profit = 0.0

dates_export = []
daily_profit_export = []
cumul_profit_export = []
daily_mean_abs_action_export = []
daily_variance_action_export = []

py_env = tf_env_eval.pyenv.envs[0]

for ep, start_idx in enumerate(index_debuts_jours):
    episode_profit = 0.0
    idx_excel = start_idx + window_size
    current_date = df_dates.iloc[idx_excel]['FROM_DATE']
    
    py_env._observation_index = start_idx
    py_env._episode = (start_idx // nb_quarters_per_episode) + 1
    py_env._observation_samples[start_idx, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[start_idx]
    time_step = tf_env_eval.reset()
    
    # Reinitialize stochastic hidden recurrent state cells at the start of a new day profile
    policy_state = saved_policy.get_initial_state(tf_env_eval.batch_size)
    
    actions_list_ep = []
    
    for _ in range(nb_quarters_per_episode):
        soc_avant_step = py_env._observation[py_env._SOC_index]
        
        action_step = saved_policy.action(time_step, policy_state)
        policy_state = action_step.state 
        raw_action = action_step.action.numpy()[0]
        
        # --- FIX: Extract real physical action for summary statistics reporting ---
        new_soc_virtuel = np.clip(soc_avant_step + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        action_effective = (new_soc_virtuel - soc_avant_step) * (4 * EP_ratio)
        actions_list_ep.append(action_effective)
        
        time_step = tf_env_eval.step(action_step.action)
        episode_profit += time_step.reward.numpy()[0]
        
    total_profit += episode_profit
    
    daily_mean_abs_action_export.append(np.mean(np.abs(actions_list_ep)))
    daily_variance_action_export.append(np.var(actions_list_ep))
    
    dates_export.append(current_date)
    daily_profit_export.append(episode_profit)
    cumul_profit_export.append(total_profit)
    
    if (ep + 1) % 100 == 0:
        print(f"  Day {ep+1:04d}/{num_episodes} ({current_date.strftime('%Y-%m-%d')}) | Cumulative: {total_profit:>12.2f} €")

# %% =============================================================================
# 6. SPREADSHEET LOGGING OUTPUT CONFIGURATIONS (JOUR / MOIS / ANNEE MULTI-SHEET)
# =============================================================================
print("\nExporting multi-sheet Excel results with action metrics...")
df_jour = pd.DataFrame({
    'Date': dates_export,
    'Profit Journalier (€)': daily_profit_export,
    'Profit Cumule (€)': cumul_profit_export,
    'Mean Abs Action': daily_mean_abs_action_export,
    'Action Variance': daily_variance_action_export
})

# Monthly performance summary mapping parameters
df_jour['Month'] = df_jour['Date'].dt.to_period('M')
df_mois = df_jour.groupby('Month')['Profit Journalier (€)'].sum().reset_index()
df_mois.rename(columns={'Profit Journalier (€)': 'Profit Mensuel (€)'}, inplace=True)
df_mois['Month'] = df_mois['Month'].dt.strftime('%Y-%m')

# Annual performance tracking aggregations
df_jour['Year'] = df_jour['Date'].dt.year
df_annee = df_jour.groupby('Year').agg({
    'Profit Journalier (€)': 'sum',
    'Mean Abs Action': 'mean',
    'Action Variance': 'mean'
}).reset_index()

df_annee.rename(columns={'Profit Journalier (€)': 'Profit Annuel (€)'}, inplace=True)

# Write structural sheets arrays columns out to Excel file
excel_path = os.path.join(SAVE_DIR, "Gains_PPO_2020_2023.xlsx")
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df_jour_export = df_jour.drop(columns=['Month', 'Year']).copy()
    df_jour_export['Date'] = df_jour_export['Date'].dt.strftime('%Y-%m-%d')
    
    df_jour_export.to_excel(writer, sheet_name='Journalier', index=False)
    df_mois.to_excel(writer, sheet_name='Mensuel', index=False)
    df_annee.to_excel(writer, sheet_name='Annuel', index=False)

print(f"Excel successfully saved at: {excel_path}")

# --- GLOBAL LONG TERM MULTI YEAR RETURNS PLOT ---
fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(num_episodes), cumul_profit_export, color='forestgreen', linewidth=2)
fig_f.gca().fill_between(range(num_episodes), cumul_profit_export, color='lightgreen', alpha=0.3)
ax_f.set_title("Cumulative Financial Performance PPO (2020-2023)", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Days Evaluated")
ax_f.set_ylabel("Total Profit (€)")
ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plot_global_path = os.path.join(SAVE_DIR, "Global_Results_2020_2023_PPO.png")
fig_f.savefig(plot_global_path, dpi=300)

print("\n" + "="*75)
print("              PPO ANNUAL SUMMARY TABLE FOR LATEX")
print("="*75)
df_annee_print = df_annee.copy()
df_annee_print['Profit Annuel (€)'] = df_annee_print['Profit Annuel (€)'].map('{:,.2f} €'.format)
df_annee_print['Mean Abs Action'] = df_annee_print['Mean Abs Action'].round(3)
df_annee_print['Action Variance'] = df_annee_print['Action Variance'].round(3)
print(df_annee_print.to_string(index=False))
print("="*75)

print(f"\nDaily Average over {num_episodes} days: {total_profit / num_episodes:,.2f} € / day")
print("Process finished.")
