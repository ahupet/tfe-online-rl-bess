# tfe-online-rl-bess
# Online Reinforcement Learning for Battery Energy Storage Participation in the Imbalance Settlement Mechanism

This repository contains the official code for the Master's thesis in Civil Energy Engineering submitted to the **Faculté Polytechnique de Mons (UMONS)**.

**Author:** Achille Hupet  
**Supervisors:** Prof. François Vallée, Dr. Ir. Jean-François Toubeau, Ir. Cyril Rasic  
**Academic Year:** June 2026  

---

## 📌 Project Overview

This research project applies **Deep Reinforcement Learning (DRL)** to maximize the revenues of a large battery (BESS, 20 MW/20 MWh) playing on the Belgian imbalance market (Elia). 

We compare two main AI agents (**TD3** and **PPO**) across different scenarios, fix a critical logic bias found in past research to make the code fully realistic, and implement **Continual Learning** so the AI can adapt automatically to major economic updates (like the 2022 energy crisis).

---

## 📂 Repository Structure

The project is organized as follows:

```text
├── src/
│   ├── benchmark/           # Non-causal baseline setup (TD3 & PPO agents and evaluation)
│   ├── operational/         # Realistic, causal setup (TD3 & PPO agents and evaluation)
│   └── continual_learning/  # Online fine-tuning framework and adaptive learning loops
|   └── price_ladder_reconstruction/ # Code used to rebuild the Elia price ladder for the continual learning strategy
