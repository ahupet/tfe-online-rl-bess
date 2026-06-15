# Benchmark RL Submodule (Privileged Framework)

This directory contains the code implementation and datasets for the **Benchmark Paradigms**. In this module, the Reinforcement Learning agents operate under a non-causal "privileged" setup (including look-ahead features), serving as an ideal upper-bound reference ceiling.

## 📂 Directory Contents

* **`Classes_TD3_old_final.py`**: The shared Python environment class containing the battery characteristics (`SoC`, `eta`, `wear costs`) and the synchronized baseline price-maker logic.
* **`TD3_agent_old_final.py`**: The synchronized training pipeline for the benchmark off-policy deterministic (`TD3`) agent.
* **`PPO_old_final_checkpoint.py`**: The synchronized training pipeline for the benchmark on-policy stochastic (`PPO-clip`) agent, including checkpoints and early stopping metrics.
* **`Evaluation_TD3_Old_final.py`**: Automated validation execution pass for the `TD3` baseline model over target historical timelines.
* **`Evaluation_PPO_old_final.py`**: Automated validation evaluation pass tracking recurrent `policy_state` arrays for the `PPO` network.
* **`Frame_1819_training.xlsx`**: Historical training data spreadsheet used to calibrate baseline statistical trackers.
* **`Frame_1819_validation.xlsx`**: Core offline validation dataset sheet.
* **`Frame_1819_test.xlsx`**: Evaluation baseline test sheet used to plot intra-day strategy profiles.

## 🚀 Execution Note
Ensure that all three `Frame_1819_*.xlsx` spreadsheets are located directly inside this folder before triggering any training or evaluation pipelines.
