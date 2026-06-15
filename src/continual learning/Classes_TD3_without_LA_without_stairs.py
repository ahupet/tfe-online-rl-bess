# -*- coding: utf-8 -*-
"""
TD3 classes & functions - Version Finale TFE (Pure Déduction)
- Le SI a été totalement retiré.
- L'escalier a été TOTALEMENT retiré des observations de l'agent (ni t, ni t-1).
- Logique Price Maker 100% basée sur le NRV (cachée dans l'environnement).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_agents.trajectories.policy_step as ps
from tf_agents.trajectories import trajectory
from sklearn.preprocessing import StandardScaler 
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class Environment(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, 
        max_power: float, discount_rate: float, nb_quarters_per_episode: int, 
        EP: float, eta: float, bat_replacement_cost, penalty
    ):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-10.0, maximum=10.0, name="observation"
        )

        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)
        self._penalty = float(penalty)
        
        self._SOC_index = 0 
        self._observation_index = 0
        self._episode = 0
        self._episode_ended = False
        self._observation = self._observation_samples[self._observation_index]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        initial_soc = self._observation[self._SOC_index]
        new_soc = initial_soc + (action / (4 * self._EP_ratio))

        if 0 <= new_soc <= 1:
            penalized = 0.0
        else:
            penalized = self._penalty

        new_soc = np.clip(new_soc, 0.0, 1.0)
        fraction_of_max_power = (new_soc - initial_soc) * (4 * self._EP_ratio)

        cost = 0.0
        if fraction_of_max_power < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        battery_charge_MW = fraction_of_max_power * self._max_power
        
        if battery_charge_MW > 0: 
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:        
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        # Index 0 = NRV, Index 1 = Price, Index 2 à 14 = Escalier
        base_nrv = self._non_observable_samples[self._observation_index, 0]
        
        modified_nrv = base_nrv + network_charge_MW
        
        idx_mod = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_mod)) if idx_mod < 6 else int(np.ceil(idx_mod))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        realized_price = self._non_observable_samples[self._observation_index, 2 + idx_mod]
        
        profit = -network_charge_MW * realized_price * 0.25
        reward = profit - cost - penalized 

        self._observation_index += 1

        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]
            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)


class Environmentvalidation(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, 
        max_power: float, discount_rate: float, nb_quarters_per_episode: int,
        EP: float, eta: float, bat_replacement_cost 
    ):
        super().__init__()

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-10.0, maximum=10.0, name="observation"
        )

        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._SOC_index = 0 
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)

        self._observation_index = 0
        self._episode_ended = False
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._episode += 1
        self._episode_ended = False
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        fraction_of_max_power = float(action)
        soc = self._observation[self._SOC_index]
        new_soc = soc + (fraction_of_max_power / (4 * self._EP_ratio))
        new_soc = np.clip(new_soc, 0.0, 1.0)

        fraction_of_max_power = (new_soc - soc) * (4 * self._EP_ratio)

        cost = 0.0
        if action < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        battery_charge_MW = fraction_of_max_power * self._max_power

        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)
            
        base_nrv = self._non_observable_samples[self._observation_index, 0]
        
        modified_nrv = base_nrv + network_charge_MW
        
        idx_mod = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_mod)) if idx_mod < 6 else int(np.ceil(idx_mod))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        realized_price = self._non_observable_samples[self._observation_index, 2 + idx_mod]
        
        profit = -network_charge_MW * realized_price * 0.25
        reward = profit - cost

        self._observation_index += 1

        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]
            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)

def std_around_value(data, arbitrary_value):
    deviations = np.array(data) - arbitrary_value
    return np.sqrt(np.mean(deviations ** 2))

# %% -------Import data and preprocessing-------
def importdata(path_training, path_validation, path_test, window_size):
    """
    Importation "Pure Déduction".
    L'escalier n'est plus du tout passé en observation à l'agent.
    """ 
    history_cols = ['NRV', 'Price'] 
    cyclic_cols = ['min_sin', 'min_cos', 'h_sin', 'h_cos', 'day_sin', 'day_cos'] 
    price_cols=[f"{i}MW" for i in range (-600,700,100)]
    
    # Nouvel ordre pour le non_observable: NRV (index 0), Price (index 1), -600MW... (index 2+)
    reward_cols = ['NRV', 'Price'] + price_cols
    
    df_train = pd.read_excel(path_training).ffill().bfill() 
    df_val = pd.read_excel(path_validation).ffill().bfill() 
    df_test = pd.read_excel(path_test).ffill().bfill() 
    
    scaler_hist = StandardScaler() 
    scaler_hist.fit(df_train[history_cols]) 
    
    def create_windowed_dataset(df, window): 
        n_samples = len(df) - window
        data_scaled_hist = scaler_hist.transform(df[history_cols]) 
        
        history_windows = [] 
        for i in range(window, len(df)): 
            # L'historique prend de t-window à t-1
            window_slice = data_scaled_hist[i-window : i].flatten() 
            history_windows.append(window_slice) 
        history_windows = np.array(history_windows, dtype=np.float32) 
            
        # Les variables cycliques restent à l'instant t
        cyclic_current = df.iloc[window:][cyclic_cols].values.astype(np.float32) 
        soc_column = np.zeros((n_samples, 1), dtype=np.float32) 
        
        # --- L'agent ne voit que le SoC, le temps et l'historique NRV/Prix ---
        observable_data = np.hstack([soc_column, cyclic_current, history_windows]) 
        
        # Les données non-observables (profit réel avec l'escalier)
        non_observable_data = df.iloc[window:][reward_cols].values.astype(np.float32) 
        
        return observable_data, non_observable_data 
    
    print("Traitement du Training set...") 
    train_obs, train_non_obs = create_windowed_dataset(df_train, window_size) 
    print("Traitement du Validation set...") 
    val_obs, val_non_obs = create_windowed_dataset(df_val, window_size) 
    print("Traitement du Test set...") 
    test_obs, test_non_obs = create_windowed_dataset(df_test, window_size) 
    
    print("--- Terminé ---") 
    print(f"Dimension de l'état (State Dim) : {train_obs.shape[1]}") 
    
    return train_obs, val_obs, test_obs, train_non_obs, val_non_obs, test_non_obs

# %% -------Data collection-------
def collect_data(environment, policy, buffer, steps, initial):
    reward = 0
    unfeas_action = []
    x = 16 * 20
    arbitrary_actions = [-1, -1, -0.5, 1, 1, 1, 1, 1, 1, 0.1, -1, -1, -1, -1, 0.5, -0.5] 

    if initial == 0:
        for _ in range(int(x / 16)):
            for i in range(16):
                sample = arbitrary_actions[i]
                time_step = environment.current_time_step()
                reward = time_step.reward
                action_step = tf.constant(sample, dtype='float32', shape=(1,))
                action_step = ps.PolicyStep(action_step)
                next_time_step = environment.step(sample)
                traj = trajectory.from_transition(time_step, action_step, next_time_step)
                buffer.add_batch(traj)
                if next_time_step.is_last():
                    environment.reset()

        for _ in range(steps - x):
            time_step = environment.current_time_step()
            reward = time_step.reward
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            buffer.add_batch(traj)
            if next_time_step.is_last():
                environment.reset()

    else:
        for _ in range(steps):
            time_step = environment.current_time_step()
            reward += time_step.reward
            action_step = policy.action(time_step)
            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            buffer.add_batch(traj)
            if next_time_step.is_last():
                environment.reset()

    return reward, unfeas_action