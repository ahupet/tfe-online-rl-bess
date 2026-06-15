# -*- coding: utf-8 -*-
"""
Evaluation Script for TD3 Agent - NEW V5 (Synchronized Midnight - Pure Déduction)
- Uses the environment WITHOUT the price ladder stairs in observations.
- Embedded Anti-Crash Safeguards (GPU & RAM management layers).
- Crimson Red color theme specific to the TD3 operational network.
- Automated generation and logging of the final test results file (.txt).

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import gc # <-- Anti-crash Python memory garbage collection layer
from tf_agents.environments import tf_py_environment

# --- ANTI-CRASH MEMORY SAFEGUARD N°1: VRAM Growth Allocation ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Enable dynamic memory growth to prevent TensorFlow from locking the whole GPU VRAM capacity
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# --- CORRECTED IMPORT: NO PRICE LADDER IN THE OBSERVATION PIPELINE ---
from Classes_TD3_without_LA_without_stairs import Environmentvalidation, importdata

# %%=============================================================================
# 1. TARGET DIRECTORIES
# =============================================================================
# --- Target operational seed directory execution folder ---
DOSSIER_RUN = r"Run_TD3_20260418_123006_EARLY_STOP_WDW8_S3" 
DOSSIER_AGENT = os.path.join(DOSSIER_RUN, "saved_policy_final")
TXT_PATH = os.path.join(DOSSIER_RUN, "hyperparameters.txt")

# %% =============================================================================
# 2. AUTOMATIC HYPERPARAMETER READING (.txt)
# =============================================================================
print("Reading hyperparameters from text file...\n")
hyperparams = {}
if os.path.exists(TXT_PATH):
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                parts = line.split(":", 1)
                key = parts[0].replace("-", "").strip()
                value = parts[1].strip()
                hyperparams[key] = value

nb_quarters_per_episode = int(hyperparams.get("Quarters per episode (24h)", 96))
window_size = int(hyperparams.get("Window size (Past History)", 8))

# %%=============================================================================
# 3. PHYSICAL PARAMETERS
# =============================================================================
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# Input training data matrices including window_size historical warming tracks
path_training = 'Dataset_Train_2018_2019_wdw8.xlsx'
path_validation = 'Dataset_Val_2018_2019_wdw8.xlsx'
path_test = 'Dataset_Test_2018_2019_wdw8.xlsx'

# %% =============================================================================
# 4. DATA IMPORT AND MIDNIGHT ANCHOR CONFIGURATIONS
# =============================================================================
print("\nImporting data (NEW V5 logic - Pure Deduction)...")
train_obs, val_obs, test_obs, \
train_non_obs, val_non_obs, test_non_obs = importdata(
    path_training, path_validation, path_test, window_size
)

# Parse historical calendar timestamps columns and cast to datetime format types
df_dates = pd.read_excel(path_test, usecols=['FROM_DATE'])
df_dates['FROM_DATE'] = pd.to_datetime(df_dates['FROM_DATE'])

eval_obs = test_obs
eval_non_obs = test_non_obs

# %% =============================================================================
# 5. ENVIRONMENT CREATION AND POLICY LOADING
# =============================================================================
print("\nCreating Evaluation environment...")
env_eval = Environmentvalidation(
    eval_obs, eval_non_obs, battery_max_power,
    1.0, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency,
    battery_replacement_cost
)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

print(f"Loading agent from: {DOSSIER_AGENT}...")
saved_policy = tf.compat.v2.saved_model.load(DOSSIER_AGENT)

# %% =============================================================================
# 6. DETAILED SIMULATION OF SPECIFIC DAYS (Search by DATE)
# =============================================================================
jours_a_simuler = ['2019-06-21', '2019-06-22', '2019-06-23', '2018-11-21', '2018-11-22']

for date_str in jours_a_simuler:
    print(f" -> Plotting day {date_str}...")
    lignes_date = df_dates[df_dates['FROM_DATE'].dt.strftime('%Y-%m-%d') == date_str]
    if lignes_date.empty: 
        continue
    
    # Recalibrate Python environment row alignment offset by subtracting window_size
    target_index = lignes_date.index[0] - window_size
    if target_index < 0: 
        continue

    py_env = tf_env_eval.pyenv.envs[0]
    py_env._observation_index = target_index
    py_env._observation_samples[target_index, py_env._SOC_index] = 0.5 # Force initial mid-point SoC state
    py_env._observation = py_env._observation_samples[target_index]
    time_step = tf_env_eval.reset()
    
    # Pre-allocate profile logs containers lists
    actions_list, soc_list, profit_cumul = [], [], []
    price_taker_list, price_maker_list = [], []
    nrv_initial_list, nrv_modified_list = [], []
    current_profit = 0.0
    
    for _ in range(nb_quarters_per_episode):
        idx = py_env._observation_index
        soc = py_env._observation[py_env._SOC_index]
        
        # --- PARSE UNMODIFIED DATA COLUMNS (Index 0 = NRV, Index 1 = Price Taker baseline) ---
        base_nrv = py_env._non_observable_samples[idx, 0]
        base_price = py_env._non_observable_samples[idx, 1]
        
        # Query the frozen actor policy network for the battery power setpoint
        action_step = saved_policy.action(time_step)
        raw_action = action_step.action.numpy()[0]
        
        # Evaluate net physical electrical change at the grid interface boundaries
        new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        fraction_power = (new_soc - soc) * (4 * EP_ratio)
        bat_MW = fraction_power * battery_max_power
        net_MW = bat_MW / np.sqrt(roundtrip_efficiency) if bat_MW > 0 else bat_MW * np.sqrt(roundtrip_efficiency)

        # Apply strict V5 causal logic path to compute market price modifications
        modified_nrv = base_nrv + net_MW
        idx_raw = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_raw)) if idx_raw < 6 else int(np.ceil(idx_raw))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        # Retrieve clearing price from the non-observable hidden ladder starting at index column 2
        realized_price = py_env._non_observable_samples[idx, 2 + idx_mod]
        
        # Log performance metrics per step
        actions_list.append(raw_action)
        soc_list.append(soc)
        nrv_initial_list.append(base_nrv)
        nrv_modified_list.append(modified_nrv)
        price_taker_list.append(base_price)
        price_maker_list.append(realized_price)
        
        # Advance environment state tracking pass
        time_step = tf_env_eval.step(action_step.action)
        current_profit += time_step.reward.numpy()[0]
        profit_cumul.append(current_profit)

    # --- INTRA-DAY PROFILE PLOTTING (CRIMSON RED THEME SPECIFIC TO TD3 TARGET LABELS) ---
    fig, (ax_price, ax_action_si, ax_soc, ax_profit) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    x_axis = np.arange(nb_quarters_per_episode) / 4
    
    ax_price.plot(x_axis, price_taker_list, color='gray', linestyle='--', label='Price Taker (Base)', alpha=0.7)
    ax_price.plot(x_axis, price_maker_list, color='crimson', linewidth=2, label='Price Maker (Impacted)')
    ax_price.set_ylabel("Price (€/MWh)")
    ax_price.set_title(f"Market Price Impact - {date_str} (TD3 V5 - Pure Deduction)")
    ax_price.legend(loc="upper left"); ax_price.grid(True, alpha=0.5)
    
    ax_action_si.plot(x_axis, nrv_initial_list, color='black', linestyle='--', label='Initial SI', alpha=0.4)
    ax_action_si.plot(x_axis, nrv_modified_list, color='crimson', label='Modified SI', linewidth=1.5, alpha=0.8)
    ax_action_si.axhline(0, color='black', linewidth=0.8)
    ax_action_si.set_ylabel("SI (MW)", color='crimson'); ax_action_si.tick_params(axis='y', labelcolor='crimson')
    
    ax_action_twin = ax_action_si.twinx()
    ax_action_twin.bar(x_axis, actions_list, width=0.15, color=['red' if a < 0 else 'green' for a in actions_list], alpha=0.4, label='Agent Action')
    ax_action_twin.set_ylabel("Action (-1 to 1)", color='green'); ax_action_twin.set_ylim(-1.1, 1.1); ax_action_twin.tick_params(axis='y', labelcolor='green')
    ax_action_si.set_title("Agent Decisions vs Grid Balance")
    
    ax_soc.plot(x_axis, soc_list, color='crimson', linewidth=2)
    ax_soc.set_ylabel("SoC (0 to 1)"); ax_soc.set_ylim(-0.1, 1.1)
    ax_soc.axhline(0, color='black', linestyle='--', alpha=0.5); ax_soc.axhline(1, color='black', linestyle='--', alpha=0.5)
    ax_soc.set_title("Battery Level (SoC)"); ax_soc.grid(True, alpha=0.5)
    
    ax_profit.plot(x_axis, profit_cumul, color='goldenrod', linewidth=2)
    ax_profit.set_ylabel("Cumulative Profit (€)"); ax_profit.set_xlabel("Time (Hours)")
    ax_profit.set_title("Daily Profitability"); ax_profit.grid(True, alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(DOSSIER_RUN, f"Strategy_Day_{date_str}_TD3_V5.png"), dpi=300)
    plt.close(fig) # --- ANTI-CRASH METRIC RELEASE SAFEGUARD

# %% =============================================================================
# 7. FULL DATASET GLOBAL EVALUATION (SYNCHRONIZED MIDNIGHT ANCHOR)
# =============================================================================
print("\nLaunching annual evaluation (Midnight Anchor)...")
# Parse chronology dataframe rows matching exactly midnight hours structures (00:00:00)
index_minuits = df_dates[(df_dates['FROM_DATE'].dt.hour == 0) & (df_dates['FROM_DATE'].dt.minute == 0)].index
index_debuts_jours = [i - window_size for i in index_minuits if (i - window_size) >= 0]

num_episodes = len(index_debuts_jours)
total_profit = 0.0
yearly_cumul = []

py_env = tf_env_eval.pyenv.envs[0]
# Iterate exclusively over the 96 pristine chronological midnight anchor index points
for ep, start_idx in enumerate(index_debuts_jours):
    episode_profit = 0.0
    
    # --- JUMP ENTIRELY TO THE TARGET MIDNIGHT INDEX (Completely avoids leaking rolling data overlaps) ---
    py_env._observation_index = start_idx
    py_env._observation_samples[start_idx, py_env._SOC_index] = 0.5 # Safety initialization mid-point SoC override
    py_env._observation = py_env._observation_samples[start_idx]
    time_step = tf_env_eval.reset()
    
    for _ in range(nb_quarters_per_episode):
        action_step = saved_policy.action(time_step)
        time_step = tf_env_eval.step(action_step.action)
        episode_profit += time_step.reward.numpy()[0]
        
    total_profit += episode_profit
    yearly_cumul.append(total_profit)
    if (ep + 1) % 10 == 0:
        print(f"  Day {ep+1:03d}/{num_episodes} | Cumulative: {total_profit:>10.2f} €")
        gc.collect() # --- ANTI-CRASH MEMORY FLUSH SAFEGUARD

# --- OUTPUT TEXT FILE PERFORMANCE TRACKING REPORT ---
chemin_resultats = os.path.join(DOSSIER_RUN, "Test_Evaluation_Results_V5.txt")
with open(chemin_resultats, "w", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write(f"Number of tested days: {num_episodes} days\n")
    f.write(f"Total Cumulative Profit: {total_profit:,.2f} €\n")
    f.write(f"Daily Average: {total_profit / num_episodes:,.2f} € / day\n")
    f.write("="*50 + "\n")

# --- RESULTS GRAPH GENERATION LAYOUT (CRIMSON THEME ALIGNED FOR TD3 TRACKS) ---
fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(range(num_episodes), yearly_cumul, color='crimson', linewidth=2.5)
ax_f.fill_between(range(num_episodes), yearly_cumul, color='lightcoral', alpha=0.3)
ax_f.set_title("Cumulative Financial Performance on Dataset (TD3)", fontsize=14, fontweight='bold')
ax_f.set_xlabel("Days"); ax_f.set_ylabel("Total Profit (€)"); ax_f.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
fig_f.savefig(os.path.join(DOSSIER_RUN, "Global_Results_V5.png"), dpi=300)
plt.show()

print(f"\nFINAL RESULT: {total_profit:,.2f} € | Avg: {total_profit/num_episodes:,.2f} €/day")


# -*- coding: utf-8 -*-
"""
Impact Analysis Script - TD3 Operational (New V5)
Indicateurs : % de dégradation et Intensité moyenne
Logique : SI = -NRV
Export : Sauvegarde des données SI Initial vs Modifié pour graphique de distribution
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tf_agents.environments import tf_py_environment

# --- IMPORT DES CLASSES OPERATIONNELLES ---
from Classes_TD3_without_LA_without_stairs import Environmentvalidation, importdata

# %%=============================================================================
# 1. CONFIGURATION
# ==============================================================================
DOSSIER_RUN = r"Run_TD3_20260418_123006_EARLY_STOP_WDW8_S3" 
DOSSIER_AGENT = os.path.join(DOSSIER_RUN, "saved_policy_final")
FILE_TEST = 'Dataset_Test_2018_2019_wdw8.xlsx'
WINDOW_SIZE = 8 

# Parametres physiques
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio

# %%=============================================================================
# 2. CHARGEMENT DES DONNEES
# ==============================================================================
print("Loading data for Grid Impact Analysis...")

path_training = 'Dataset_Train_2018_2019_wdw8.xlsx'
path_validation = 'Dataset_Val_2018_2019_wdw8.xlsx'

_, _, test_obs, _, _, test_non_obs = importdata(
    path_training, path_validation, FILE_TEST, WINDOW_SIZE
)

df_dates = pd.read_excel(FILE_TEST, usecols=['FROM_DATE'])
df_dates['FROM_DATE'] = pd.to_datetime(df_dates['FROM_DATE'])

env_eval = Environmentvalidation(test_obs, test_non_obs, battery_max_power, 1.0, 96, EP_ratio, roundtrip_efficiency, battery_replacement_cost)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)
saved_policy = tf.compat.v2.saved_model.load(DOSSIER_AGENT)

# %%=============================================================================
# 3. IDENTIFICATION DES POINTS D'ANCRAGE (MIDNIGHT)
# ==============================================================================
index_minuits = df_dates[(df_dates['FROM_DATE'].dt.hour == 0) & (df_dates['FROM_DATE'].dt.minute == 0)].index
index_debuts_jours = [i - WINDOW_SIZE for i in index_minuits if (i - WINDOW_SIZE) >= 0]

# %%=============================================================================
# 4. BOUCLE D'ANALYSE ET COLLECTE DE DONNEES
# ==============================================================================
py_env = tf_env_eval.pyenv.envs[0]
count_active_steps = 0
count_degradation = 0
total_intensity_degradation = 0.0

# Listes pour stocker les donnees du graphique de distribution
list_si_initial = []
list_si_modified = []

print(f"Analyzing impact over {len(index_debuts_jours)} days...")

for start_idx in index_debuts_jours:
    # --- SYNCHRO ENVIRONNEMENT ---
    py_env._observation_index = start_idx
    py_env._episode = (start_idx // 96) + 1 
    py_env._observation_samples[start_idx, py_env._SOC_index] = 0.5
    py_env._observation = py_env._observation_samples[start_idx]
    time_step = tf_env_eval.reset()
    
    for _ in range(96):
        idx = py_env._observation_index
        soc = py_env._observation[py_env._SOC_index]
        
        # 1. SI Initial (SI = -NRV)
        nrv_initial = py_env._non_observable_samples[idx, 0]
        si_initial = -1.0 * nrv_initial
        
        # 2. Action de l'agent
        action_step = saved_policy.action(time_step)
        raw_action = action_step.action.numpy()[0]
        
        # 3. Impact physique
        new_soc = np.clip(soc + (raw_action / (4 * EP_ratio)), 0.0, 1.0)
        net_charge = (new_soc - soc) * (4 * EP_ratio) * battery_max_power
        
        if net_charge > 0:
            net_charge /= np.sqrt(roundtrip_efficiency)
        else:
            net_charge *= np.sqrt(roundtrip_efficiency)
            
        # 4. SI Modifie
        si_modified = si_initial - net_charge
        
        # --- STOCKAGE POUR DISTRIBUTION ---
        list_si_initial.append(si_initial)
        list_si_modified.append(si_modified)

        # --- CALCUL DES METRIQUES ---
        if abs(net_charge) > 0.1:
            count_active_steps += 1
            if abs(si_modified) > abs(si_initial):
                count_degradation += 1
                total_intensity_degradation += (abs(si_modified) - abs(si_initial))

        # Passage au pas suivant
        time_step = tf_env_eval.step(action_step.action)

# %%=============================================================================
# 5. RESULTATS ET EXPORT EXCEL
# ==============================================================================
percentage_degradation = (count_degradation / count_active_steps) * 100 if count_active_steps > 0 else 0
avg_intensity = total_intensity_degradation / count_degradation if count_degradation > 0 else 0

print("\n" + "="*55)
print(" GRID IMPACT ANALYSIS - OPERATIONAL MODEL (V5) ")
print("="*55)
print(f"Total Active Steps (Action != 0) : {count_active_steps}")
print(f"SI Degradation Rate              : {percentage_degradation:.2f} %")
print(f"Average Degradation Intensity     : {avg_intensity:.2f} MW")
print("="*55)

# 1. Sauvegarde des métriques en texte
with open(os.path.join(DOSSIER_RUN, "Grid_Impact_Operational.txt"), "w") as f:
    f.write(f"SI Degradation Rate: {percentage_degradation:.2f} %\n")
    f.write(f"Average Degradation Intensity: {avg_intensity:.2f} MW\n")

# 2. Sauvegarde des données brutes en Excel pour le graphique de distribution
df_impact = pd.DataFrame({
    'SI_Initial': list_si_initial,
    'SI_Modified': list_si_modified
})
excel_path = os.path.join(DOSSIER_RUN, "SI_Impact_Data_Operational.xlsx")
df_impact.to_excel(excel_path, index=False)

print(f"\nDonnees d'impact sauvegardees dans : {excel_path}")
