# -*- coding: utf-8 -*-
"""
Continual Learning Script - PPO Agent (V5 Logic - Pure Déduction) - 2020-2023
- Restores the pretrained model architecture using the 18-19 offline checkpoint setup (Seed 1).
- Dynamically updates the clipping boundary and epochs count based on the previous block's variance.
- Parses the weighted historical price ladder variance metric from the generated Excel log file.
- Generates a multi-sheet spreadsheet tracking Daily, Monthly, and Annual financial returns columns.
- Integrated adaptive control law checking for block-by-block or weekly operational triggers.

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
import gc # <-- Anti-crash memory protection garbage collector integration

# --- ANTI-CRASH MEMORY CONFIGURATION SAFEGUARD N°1: VRAM Growth Allocation ---
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            # Enable dynamic memory growth to prevent TensorFlow from locking the whole GPU VRAM capacity
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e: 
        print(e)

tf.keras.backend.set_floatx("float32")
tf.keras.backend.clear_session()

# FRAMEWORK CORE TF-AGENTS IMPORTS
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_saver
from tf_agents.utils import common

# IMPORT MODULES FROM ORIGINAL OPERATIONAL PARADIGM (NO PRICE LADDER)
from Classes_TD3_without_LA_without_stairs import Environment, Environmentvalidation, importdata

# %%=============================================================================
# 0. CONTINUAL LEARNING USER CONTROL INTERFACE OPTIONS
# ==============================================================================
FREQUENCE_MISE_A_JOUR = "MOIS"     # Operational options: "MOIS" (Monthly) or "SEMAINE" (Weekly)
SAUVEGARDER_POLICY_PERIODIQUE = False 

learn_rate = 5e-5
gamma = 0.99
ppo_clip_initial = 0.05

# =============================================================================
# 1. PARAMETERS AND TARGET DIRECTORIES
# =============================================================================
DOSSIER_RUN_INITIAL = r"C:\Users\pcstudentgele11\TFE_HUPET_ACHILLE\PPO_without_LA_without_stairs\Run_PPO_20260416_112607_EARLY_STOP_WDW8_S1"
CHECKPOINT_DIR = os.path.join(DOSSIER_RUN_INITIAL, "checkpoint_complet")
FILE_INDICATORS = "Analyse_Indicateurs_Clipping_PPO.xlsx"

SAVE_DIR = f"Resultats_Online_{FREQUENCE_MISE_A_JOUR}_PPO_Adaptive_Clipping_S1_V4"
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Dossier de résultats Continual Learning créé : {SAVE_DIR}")

# Fixed operational constraints parameters and hardware coefficients
window_size = 8
nb_quarters_per_episode = 96
battery_max_power = 20
EP_ratio = 1
roundtrip_efficiency = 0.9
battery_replacement_cost = 3e+5 * battery_max_power * EP_ratio
penal = 200

path_training = "Dataset_Train_2018_2019_wdw8.xlsx" 
path_validation = "Dataset_Val_2018_2019_wdw8.xlsx" 
path_test = "Dataset_2020_2023_Escalier_Empirique_Am_6h_V3_wdw8.xlsx" 

# %%=============================================================================
# 1.B LOAD SPREADSHEET MONITORING HISTORICAL VOLATILITY INDICATORS
# ==============================================================================
if os.path.exists(FILE_INDICATORS):
    print(f"Chargement des indicateurs de clipping depuis : {FILE_INDICATORS}")
    df_indicators = pd.read_excel(FILE_INDICATORS, sheet_name="Indicateurs_Clipping")
    df_indicators["Mois"] = df_indicators["Mois"].astype(str)
else:
    print(f"Erreur : Le fichier {FILE_INDICATORS} est introuvable. Lancez le script de calcul d abord.")
    df_indicators = pd.DataFrame()

# %% =============================================================================
# 2. DATA IMPORT AND CHRONOLOGICAL SEGMENTATION
# =============================================================================
print("\nImporting data (Calibrating scalers on 2018-2019, applying to 2020-2023)...")
_, _, test_obs, _, _, test_non_obs = importdata(path_training, path_validation, path_test, window_size)

df_dates = pd.read_excel(path_test, usecols=["FROM_DATE"])
df_dates["FROM_DATE"] = pd.to_datetime(df_dates["FROM_DATE"])

# Segment continuous timestamps columns into uniform continual learning segments
if FREQUENCE_MISE_A_JOUR == "MOIS":
    df_dates["Periode"] = df_dates["FROM_DATE"].dt.to_period("M")
elif FREQUENCE_MISE_A_JOUR == "SEMAINE":
    df_dates["Periode"] = df_dates["FROM_DATE"].dt.to_period("W-MON")

# %% =============================================================================
# 3. ENVIRONMENT INSTANTIATION (2020-2023 CRITICAL HORIZON)
# =============================================================================
print("\nCreating Train and Eval Environments for Online Learning...")
env_eval = Environmentvalidation(test_obs, test_non_obs, battery_max_power, 1.0, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency, battery_replacement_cost)
tf_env_eval = tf_py_environment.TFPyEnvironment(env_eval)

env_train = Environment(test_obs, test_non_obs, battery_max_power, 1.0, nb_quarters_per_episode, EP_ratio, roundtrip_efficiency, battery_replacement_cost, penalty=penal)
tf_env_train = tf_py_environment.TFPyEnvironment(env_train)

# %% =============================================================================
# 4. RE-ESTABLISH PPO AGENT PARADIGM WITH ENCAPSULATED DYNAMIC CLIPPING VARIABLE
# =============================================================================
print("\nRe-creating PPO Agent architecture...")
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env_train.observation_spec(), tf_env_train.action_spec(),
    fc_layer_params=(500, 500), activation_fn=tf.keras.activations.relu)

value_net = value_network.ValueNetwork(
    tf_env_train.observation_spec(), fc_layer_params=(500, 500), activation_fn=tf.keras.activations.relu)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learn_rate, clipnorm=1.0)
train_step_counter = tf.Variable(0)

# Encapsulate the clipping ratio within a TensorFlow variable to allow dynamic graph mutations mid-loop
ppo_clip_var = tf.Variable(ppo_clip_initial, dtype=tf.float32, name="ppo_clip_dynamic")

agent = ppo_agent.PPOAgent(
    tf_env_train.time_step_spec(), tf_env_train.action_spec(), optimizer=optimizer,
    actor_net=actor_net, value_net=value_net, num_epochs=1,
    importance_ratio_clipping=ppo_clip_var, entropy_regularization=0.05,
    use_gae=True, use_td_lambda_return=True, discount_factor=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

# Restore baseline 2018-2019 parameters via checkpointer modules
print(f"Restoring Full Brain (Checkpoint) from: {CHECKPOINT_DIR}...")
checkpointer = common.Checkpointer(ckpt_dir=CHECKPOINT_DIR, max_to_keep=1, agent=agent, policy=agent.policy, global_step=train_step_counter)
checkpointer.initialize_or_restore()

buffer_size = 3500 
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=tf_env_train.batch_size, max_length=buffer_size)

tf_policy_action = common.function(agent.policy.action)
tf_collect_policy_action = common.function(agent.collect_policy.action)

# %%=============================================================================
# 5. CORE ADAPTIVE GATED CONTINUAL LEARNING LOOP TIMELINE
# ==============================================================================
liste_des_periodes = sorted(df_dates["Periode"].dropna().unique())

dates_export, daily_profit_export, cumul_profit_export = [], [], []
total_profit_global = 0.0

for periode in liste_des_periodes:
    print(f"\n{'='*50}\n TRAITEMENT DE LA PÉRIODE : {periode}\n{'='*50}")
    
    # Track perfect midnight index anchors to map exact daily transitions bounds
    index_minuits_periode = df_dates[(df_dates["Periode"] == periode) & (df_dates["FROM_DATE"].dt.hour == 0) & (df_dates["FROM_DATE"].dt.minute == 0)].index
    index_debuts_jours = [i - window_size for i in index_minuits_periode if (i - window_size) >= 0]
    
    if not index_debuts_jours: 
        continue
        
    # --- AUTOMATIC REPLAY VERIFICATION TARGET DERIVED FROM PREVIOUS MONTH METRICS ---
    date_debut_bloc = index_minuits_periode[0]
    date_reelle = df_dates.iloc[date_debut_bloc]["FROM_DATE"]
    
    # Offset operational target timeline back by exactly 1 month to extract late market profiles
    mois_precedent = (date_reelle - pd.DateOffset(months=1)).to_period("M")
    mois_precedent_str = str(mois_precedent)
    
    variance_ref = 300.0 # Default backup profile mapping parameter (Baseline quiet regime status)
    if not df_indicators.empty:
        row_match = df_indicators[df_indicators["Mois"] == mois_precedent_str]
        if not row_match.empty:
            variance_ref = row_match["Variance_Ponderee_Occurrences"].values[0]
            
    # Apply dynamic control law criteria (Defensive Strategy Gating Mechanism - Threshold bound set at 5,000)
    if variance_ref <= 5000:
        current_clip = 0.0
        current_epochs = 0
        print(f" -> Régime calme à modéré ({mois_precedent_str} = {variance_ref:.1f})")
        print(" -> ACTION : ppo_clip fixé à 0.0 | GEL DU CERVEAU STRICT (Sauvegarde du modèle statique)")
        
    elif 5000 < variance_ref <= 10000:
        current_clip = 0.20
        current_epochs = 3
        print(f" -> Régime en transition volatile ({mois_precedent_str} = {variance_ref:.1f})")
        print(f" -> ACTION : ppo_clip fixé à 0.20 | FINE-TUNING REQUIS ({current_epochs} Epochs)")
        
    else:
        current_clip = 0.30
        current_epochs = 3  # Configurable between 3 and 6 epochs depending on tracking reactivity requirements
        print(f" -> Régime de crise intensive ({mois_precedent_str} = {variance_ref:.1f})")
        print(f" -> ACTION : ppo_clip fixé à 0.30 | PLASTICITÉ MAXIMALE ({current_epochs} Epochs)")
        
    # Dynamically push the newly selected clipping ratio boundary constraint into the compiled graph
    ppo_clip_var.assign(current_clip)
        
    # -------------------------------------------------------------------------
    # PHASE A: CAUSAL EVALUATION (Agent trades using frozen current network status)
    # -------------------------------------------------------------------------
    py_env_eval = tf_env_eval.pyenv.envs[0]
    profit_de_la_periode = 0.0
    
    for start_idx in index_debuts_jours:
        episode_profit = 0.0
        current_date = df_dates.iloc[start_idx + window_size]["FROM_DATE"]
        
        py_env_eval._observation_index = start_idx
        py_env_eval._episode = (start_idx // nb_quarters_per_episode) + 1
        
        py_env_eval._observation_samples[start_idx, py_env_eval._SOC_index] = 0.5
        py_env_eval._observation = py_env_eval._observation_samples[start_idx]
        time_step = tf_env_eval.reset()
        
        policy_state = agent.policy.get_initial_state(tf_env_eval.batch_size)
        
        for _ in range(nb_quarters_per_episode):
            action_step = tf_policy_action(time_step, policy_state)
            policy_state = action_step.state
            time_step = tf_env_eval.step(action_step.action)
            episode_profit += time_step.reward.numpy()[0]
            
        total_profit_global += episode_profit
        profit_de_la_periode += episode_profit
        
        dates_export.append(current_date)
        daily_profit_export.append(episode_profit)
        cumul_profit_export.append(total_profit_global)
        
    print(f" -> Evaluation terminee. Profit realise sur {periode} : {profit_de_la_periode:,.2f} €")
    
    # -------------------------------------------------------------------------
    # PHASE B & C: EXPERIENCE COLLECTION AND CONDITIONAL GATED FINE-TUNING
    # -------------------------------------------------------------------------
    if current_epochs > 0:
        py_env_train = tf_env_train.pyenv.envs[0]
        nb_jours_periode = len(index_debuts_jours)
        
        for epoch in range(current_epochs):
            idx_debut_periode = index_debuts_jours[0]
            
            py_env_train._observation_index = idx_debut_periode
            py_env_train._episode = (idx_debut_periode // nb_quarters_per_episode) + 1
            
            py_env_train._observation_samples[idx_debut_periode, py_env_train._SOC_index] = 0.5
            py_env_train._observation = py_env_train._observation_samples[idx_debut_periode]
            time_step_train = tf_env_train.reset()
            
            policy_state_train = agent.collect_policy.get_initial_state(tf_env_train.batch_size)
            
            # Record transition samples trajectories across the targeted rolling timeframe block
            for step in range(nb_jours_periode * nb_quarters_per_episode):
                action_step = tf_collect_policy_action(time_step_train, policy_state_train)
                next_time_step = tf_env_train.step(action_step.action)
                
                traj = trajectory.from_transition(time_step_train, action_step, next_time_step)
                replay_buffer.add_batch(traj)
                
                time_step_train = next_time_step
                policy_state_train = action_step.state
                
                if next_time_step.is_last():
                    time_step_train = tf_env_train.reset()
                    policy_state_train = agent.collect_policy.get_initial_state(tf_env_train.batch_size)
                    
            trajectories = replay_buffer.gather_all()
            train_loss = agent.train(experience=trajectories).loss
            replay_buffer.clear() # Flush memory allocations vectors on update completion
            
        print(f" -> Fine-Tuning termine. Loss finale : {train_loss.numpy():.2f}")
    else:
        print(" -> Phase de Fine-Tuning contournee. Poids preserves intacts pour ce bloc temporel.")
        
    gc.collect()

# %% =============================================================================
# 6. EXPORT SPREADSHEET REPORTS AND CUMULATIVE EARNINGS CHART
# =============================================================================
print("\nExporting multi-sheet Excel results...")
df_jour = pd.DataFrame({'Date': dates_export, 'Profit Journalier (€)': daily_profit_export, 'Profit Cumule (€)': cumul_profit_export})

df_jour["Month"] = df_jour["Date"].dt.to_period("M")
df_mois = df_jour.groupby("Month")["Profit Journalier (€)"].sum().reset_index()
df_mois.rename(columns={"Profit Journalier (€)": "Profit Mensuel (€)"}, inplace=True)
df_mois["Month"] = df_mois["Month"].dt.strftime("%Y-%m") 

df_jour["Year"] = df_jour["Date"].dt.year
df_annee = df_jour.groupby("Year")["Profit Journalier (€)"].sum().reset_index()
df_annee.rename(columns={"Profit Journalier (€)": "Profit Annuel (€)"}, inplace=True)

excel_path = os.path.join(SAVE_DIR, f"Gains_Adaptive_Online_{FREQUENCE_MISE_A_JOUR}_PPO_2020_2023.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_jour_export = df_jour.drop(columns=["Month", "Year"]).copy()
    df_jour_export["Date"] = df_jour_export["Date"].dt.strftime("%Y-%m-%d")
    df_jour_export.to_excel(writer, sheet_name="Journalier", index=False)
    df_mois.to_excel(writer, sheet_name="Mensuel", index=False)
    df_annee.to_excel(writer, sheet_name="Annuel", index=False)

fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(df_jour["Date"], df_jour["Profit Cumule (€)"], color="teal", linewidth=2)
ax_f.fill_between(df_jour["Date"], df_jour["Profit Cumule (€)"], color="paleturquoise", alpha=0.4)
ax_f.set_title(f"Performance avec Gated Continual Learning PPO (Frequence : {FREQUENCE_MISE_A_JOUR})", fontsize=14, fontweight="bold")
ax_f.set_xlabel("Date")
ax_f.set_ylabel("Profit Cumule (€)")
ax_f.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
fig_f.savefig(os.path.join(SAVE_DIR, "Graphique_Gated_Online_Learning.png"), dpi=300)
plt.close("all")

print("\n" + "="*50)
print(f"BILAN FINAL CONTINUAL LEARNING ADAPTATIF : {total_profit_global:,.2f} €")
print(f"Les matrices et graphiques sont enregistres dans : {SAVE_DIR}")
print("="*50)
