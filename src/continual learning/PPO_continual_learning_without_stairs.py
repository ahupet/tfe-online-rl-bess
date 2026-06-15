# -*- coding: utf-8 -*-
"""
Continual Learning Script - PPO Agent (V5 Logic - Pure Déduction) - 2020-2023
- Restores the pretrained model architecture using the 18-19 offline checkpoint setup.
- Flexible tracking frequency update options: Monthly (MOIS) or Weekly (SEMAINE).
- Causal validation sequencing: Evaluates the target block, updates weights via Fine-Tuning, and rolls forward.
- Generates a multi-sheet spreadsheet tracking Daily, Monthly, and Annual financial returns columns.
- Integrated Anti-Crash Python memory garbage collection layers.
- FIX: Alignment of operational execution episodes to completely prevent ghost environment resets.

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
FREQUENCE_MISE_A_JOUR = "SEMAINE"     # Operational options: "MOIS" (Monthly) or "SEMAINE" (Weekly)
SAUVEGARDER_POLICY_PERIODIQUE = False 
EPOCHS_FINE_TUNING = 1            # Number of continuous fine-tuning iterations executed per rolling block

learn_rate = 5e-5
gamma = 0.99
ppo_clip = 0.1

# =============================================================================
# 1. PARAMETERS AND TARGET DIRECTORIES
# =============================================================================
DOSSIER_RUN_INITIAL = r"C:\Users\pcstudentgele11\TFE_HUPET_ACHILLE\PPO_without_LA_without_stairs\Run_PPO_20260416_112607_EARLY_STOP_WDW8_S1"
CHECKPOINT_DIR = os.path.join(DOSSIER_RUN_INITIAL, "checkpoint_complet")

SAVE_DIR = f"Resultats_Online_{FREQUENCE_MISE_A_JOUR}_PPO_2020_2023_S1_e1_c1"
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
# 1.B LOG METADATA HYPERPARAMETERS SUMMARY
# ==============================================================================
config = {
    "Algorithme": "PPO (Pure Déduction - Without Stairs)",
    "Période cible": "2020-2023",
    "Dataset Test": path_test,
    "Fréquence Mise à jour": FREQUENCE_MISE_A_JOUR,
    "Seed Checkpoint Initial": DOSSIER_RUN_INITIAL,
    "Epochs Fine-Tuning": EPOCHS_FINE_TUNING,
    "Learning Rate (Fine-Tuning)": learn_rate,
    "Gamma (Discount Factor)": gamma,
    "PPO Clipping": ppo_clip,
    "Window Size (Historique)": window_size
}

file_path = os.path.join(SAVE_DIR, "hyperparameters_continual_learning.txt")
with open(file_path, "w", encoding="utf-8") as f:
    f.write("="*50 + "\n")
    f.write(f" CONFIGURATION CONTINUAL LEARNING PPO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*50 + "\n\n")
    for key, value in config.items():
        f.write(f"- {key:<30} : {value}\n")
    f.write("\n" + "="*50 + "\n")
print(f"Hyperparamètres sauvés avec succès dans : {file_path}")

# %% =============================================================================
# 2. DATA IMPORT AND CHRONOLOGICAL SEGMENTATION
# =============================================================================
print("\nImporting data (Calibrating scalers on 2018-2019, applying to 2020-2023)...")
_, _, test_obs, _, _, test_non_obs = importdata(path_training, path_validation, path_test, window_size)

df_dates = pd.read_excel(path_test, usecols=["FROM_DATE"])
df_dates["FROM_DATE"] = pd.to_datetime(df_dates["FROM_DATE"])

# Discretize continuous calendar lines into uniform continual learning blocks
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
# 4. RE-ESTABLISH STOCHASTIC PPO AGENT PARADIGM
# =============================================================================
print("\nRe-creating PPO Agent architecture...")
actor_net = actor_distribution_network.ActorDistributionNetwork(
    tf_env_train.observation_spec(), tf_env_train.action_spec(),
    fc_layer_params=(500, 500), activation_fn=tf.keras.activations.relu)

value_net = value_network.ValueNetwork(
    tf_env_train.observation_spec(), fc_layer_params=(500, 500), activation_fn=tf.keras.activations.relu)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learn_rate, clipnorm=1.0)
train_step_counter = tf.Variable(0)

agent = ppo_agent.PPOAgent(
    tf_env_train.time_step_spec(), tf_env_train.action_spec(), optimizer=optimizer,
    actor_net=actor_net, value_net=value_net, num_epochs=1,
    importance_ratio_clipping=ppo_clip, entropy_regularization=0.05,
    use_gae=True, use_td_lambda_return=True, discount_factor=gamma,
    train_step_counter=train_step_counter)
agent.initialize()

# Restore baseline 2018-2019 weights parameter metrics via the Checkpointer manager
print(f"Restoring Full Brain (Checkpoint) from: {CHECKPOINT_DIR}...")
checkpointer = common.Checkpointer(ckpt_dir=CHECKPOINT_DIR, max_to_keep=1, agent=agent, policy=agent.policy, global_step=train_step_counter)
checkpointer.initialize_or_restore()

buffer_size = 3500 
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec, batch_size=tf_env_train.batch_size, max_length=buffer_size)

tf_policy_action = common.function(agent.policy.action)
tf_collect_policy_action = common.function(agent.collect_policy.action)

# %% =============================================================================
# 5. CORE SYSTEM CONTINUAL LEARNING SEQUENCING LOOP
# =============================================================================
liste_des_periodes = sorted(df_dates["Periode"].dropna().unique())

dates_export, daily_profit_export, cumul_profit_export = [], [], []
total_profit_global = 0.0

# Loop sequentially over the sorted chronological training blocks
for periode in liste_des_periodes:
    print(f"\n{'='*50}\n TRAITEMENT DE LA PÉRIODE : {periode}\n{'='*50}")
    
    # Track perfect midnight structural index anchors to isolate distinct day frames
    index_minuits_periode = df_dates[(df_dates["Periode"] == periode) & (df_dates["FROM_DATE"].dt.hour == 0) & (df_dates["FROM_DATE"].dt.minute == 0)].index
    index_debuts_jours = [i - window_size for i in index_minuits_periode if (i - window_size) >= 0]
    
    if not index_debuts_jours: 
        continue
        
    # -------------------------------------------------------------------------
    # PHASE A: CAUSAL EVALUATION (Agent trades using frozen current network status)
    # -------------------------------------------------------------------------
    py_env_eval = tf_env_eval.pyenv.envs[0]
    profit_de_la_periode = 0.0
    
    for start_idx in index_debuts_jours:
        episode_profit = 0.0
        current_date = df_dates.iloc[start_idx + window_size]["FROM_DATE"]
        
        py_env_eval._observation_index = start_idx
        
        # ALIGNMENT INTEGRITY FIX: Prevents ghost resets within evaluation loops
        py_env_eval._episode = (start_idx // nb_quarters_per_episode) + 1
        
        py_env_eval._observation_samples[start_idx, py_env_eval._SOC_index] = 0.5
        py_env_eval._observation = py_env_eval._observation_samples[start_idx]
        time_step = tf_env_eval.reset()
        
        # Instantiate recurrent policy cells state parameters vectors
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
    # PHASE B: ONLINE FINE-TUNING (Agent adapts weights to the observed historical block)
    # -------------------------------------------------------------------------
    print(f" -> Lancement du Fine-Tuning ({EPOCHS_FINE_TUNING} Epochs)...")
    py_env_train = tf_env_train.pyenv.envs[0]
    nb_jours_periode = len(index_debuts_jours)
    
    for epoch in range(EPOCHS_FINE_TUNING):
        idx_debut_periode = index_debuts_jours[0]
        
        py_env_train._observation_index = idx_debut_periode
        
        # ALIGNMENT INTEGRITY FIX: Prevents ghost environment resets during experience collection
        py_env_train._episode = (idx_debut_periode // nb_quarters_per_episode) + 1
        
        py_env_train._observation_samples[idx_debut_periode, py_env_train._SOC_index] = 0.5
        py_env_train._observation = py_env_train._observation_samples[idx_debut_periode]
        time_step_train = tf_env_train.reset()
        
        policy_state_train = agent.collect_policy.get_initial_state(tf_env_train.batch_size)
        
        # Execute rolling step transitions across the entire chronological block horizon
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
        replay_buffer.clear() # Purge uniform memory storage allocations following parameters updates
        
    print(f" -> Fine-Tuning termine. Loss finale : {train_loss.numpy():.2f}")

    # -------------------------------------------------------------------------
    # PHASE C: CHRONOLOGICAL SYSTEM CHECKPOINT EXTRACTION
    # -------------------------------------------------------------------------
    if SAUVEGARDER_POLICY_PERIODIQUE:
        nom_periode = str(periode).replace("/", "_")
        dossier_periode = os.path.join(SAVE_DIR, f"Policy_{nom_periode}")
        saver = policy_saver.PolicySaver(agent.policy)
        saver.save(dossier_periode)
        print(f" -> Cerveau mis a jour sauvegarde dans : {dossier_periode}")
        
    gc.collect() # Trigger background garbage collection flushes on block boundaries steps

# %%=============================================================================
# 6. SPREADSHEET FINANCIAL RETURN EXPORT (JOUR / MOIS / ANNEE MULTI-SHEET LOGGING)
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

excel_path = os.path.join(SAVE_DIR, f"Gains_Online_{FREQUENCE_MISE_A_JOUR}_PPO_2020_2023.xlsx")
with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    df_jour_export = df_jour.drop(columns=["Month", "Year"]).copy()
    df_jour_export["Date"] = df_jour_export["Date"].dt.strftime("%Y-%m-%d")
    df_jour_export.to_excel(writer, sheet_name="Journalier", index=False)
    df_mois.to_excel(writer, sheet_name="Mensuel", index=False)
    df_annee.to_excel(writer, sheet_name="Annuel", index=False)

# Final performance returns layout trend curve generation
fig_f, ax_f = plt.subplots(figsize=(12, 6))
ax_f.plot(df_jour["Date"], df_jour["Profit Cumule (€)"], color="teal", linewidth=2)
ax_f.fill_between(df_jour["Date"], df_jour["Profit Cumule (€)"], color="paleturquoise", alpha=0.4)
ax_f.set_title(f"Performance avec Continual Learning PPO (Mise a jour : {FREQUENCE_MISE_A_JOUR})", fontsize=14, fontweight="bold")
ax_f.set_xlabel("Date")
ax_f.set_ylabel("Profit Cumule (€)")
ax_f.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

plot_global_path = os.path.join(SAVE_DIR, "Graphique_Online_Learning.png")
fig_f.savefig(plot_global_path, dpi=300)

print("\n" + "="*50)
print(f"BILAN FINAL CONTINUAL LEARNING PPO : {total_profit_global:,.2f} €")
print(f"Excel et Graphique sauvegardes dans : {SAVE_DIR}")
print("="*50)
