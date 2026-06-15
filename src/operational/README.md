# Operational RL Submodule (Strictly Causal Inductive Framework)

This directory contains the code implementation and datasets for the **Operational Framework (NEW V5)**. In this module, look-ahead parameters and market price ladders are completely omitted from the agent's view to secure a 100% realistic, strictly causal trading architecture.

## 📂 Directory Contents

* **`operational_classes.py`**: The customized V5 operational environment class where the merit-order price ladder is fully hidden from the observation vector, relying purely on chronological windows and Net Regulation Volume (NRV).
* **`operational_td3_main.py`**: The core operational training pipeline for the `TD3` agent, featuring advanced anti-crash memory safeguards (TF graphs, graphic close flushes, and garbage collection).
* **`operational_ppo_main.py`**: The core operational training loop for the `PPO` agent, managing rolling data purges, early stopping, and checkpoint tracking.
* **`operational_td3_evaluation.py`**: Full dataset evaluation script for the causal `TD3` model, featuring a crimson-red chart layout.
* **`operational_ppo_evaluation.py`**: Full dataset evaluation script for the causal `PPO` model, featuring midnight anchor alignment to isolate distinct day horizons.
* **`Dataset_Train_2018_2019_wdw8.xlsx`**: Input training data matrix containing sliding historical window warm-ups.
* **`Dataset_Val_2018_2019_wdw8.xlsx`**: Reference offline validation data sheet for causal loops.
* **`Dataset_Test_2018_2019_wdw8.xlsx`**: Evaluation test matrix matching the V5 deduction operational state dimensions.

## 💡 Pure Deduction Principle
The agent's state dimension is exclusively composed of the current battery SoC, cyclic calendar properties, and an isolated 4-hour historical window (`window_size = 8`). It can deduce market trends without observing prices directly.
