# -*- coding: utf-8 -*-
"""
TD3 classes & functions - Version Finale TFE (Pure Déduction Paradigm)
- System Imbalance (SI) tracking completely removed from the agent's view.
- Price ladder merit-order completely omitted from agent observations (neither t nor historical context).
- Causal Price Maker logic fully driven by Net Regulation Volume (NRV) hidden in the environment.

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
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

        # Specs - Action and observation space limits configuration
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-10.0, maximum=10.0, name="observation"
        )

        # Operational arrays mapping (The agent only views windowed inputs)
        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        # Operational setup parameters and hardware coefficients
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)
        self._penalty = float(penalty)
        
        # In the pure deduction paradigm, the State of Charge index is shifted to index 0
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
        # Initialize battery state of charge to 0.5 capacity limits
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        initial_soc = self._observation[self._SOC_index]
        # Calculate next State of Charge setpoint derived from the actor network inference
        new_soc = initial_soc + (action / (4 * self._EP_ratio))

        # Boundaries evaluation validation parameters
        if 0 <= new_soc <= 1:
            penalized = 0.0
        else:
            penalized = self._penalty

        # Strict clipping to guarantee hardware energy safety bounds
        new_soc = np.clip(new_soc, 0.0, 1.0)
        fraction_of_max_power = (new_soc - initial_soc) * (4 * self._EP_ratio)

        # Non-linear degradation cost formulation model
        cost = 0.0
        if fraction_of_max_power < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        battery_charge_MW = fraction_of_max_power * self._max_power
        
        # Map localized roundtrip conversion efficiency scaling
        if battery_charge_MW > 0: 
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:        
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        # --- HIDDEN ENVIRONMENTAL LOGIC (Index 0: NRV, Index 1: Price, Index 2+: Merit-Order Escalier) ---
        base_nrv = self._non_observable_samples[self._observation_index, 0]
        
        # Evaluate causal asset modification on the baseline Net Regulation Volume
        modified_nrv = base_nrv + network_charge_MW
        
        # Track price clearing grid indices on the hidden market metrics
        idx_mod = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_mod)) if idx_mod < 6 else int(np.ceil(idx_mod))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        # Extract clearing settlement price from the non-observable tracking vector
        realized_price = self._non_observable_samples[self._observation_index, 2 + idx_mod]
        
        # Financial reward accounting loop (quarter-hourly basis factor)
        profit = -network_charge_MW * realized_price * 0.25
        reward = profit - cost - penalized 

        self._observation_index += 1

        # Case 1: Mid-episode transition timeline track
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]
            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        # Case 2: Out of bounds scenario -> clean database restart sequence trigger
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

        # Pure evaluation setup constraints parameters
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
            
        # Hidden operational data mapping calculation (Pure inductive deduction verification)
        base_nrv = self._non_observable_samples[self._observation_index, 0]
        modified_nrv = base_nrv + network_charge_MW
        
        idx_mod = (modified_nrv / 100) + 6
        idx_mod = int(np.floor(idx_mod)) if idx_mod < 6 else int(np.ceil(idx_mod))
        idx_mod = int(np.clip(idx_mod, 0, 12))
        
        realized_price = self._non_observable_samples[self._observation_index, 2 + idx_mod]
        
        # Compute pure daily profit without training safety parameters
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
    Pure Causal Preprocessing Interface.
    Omits the price ladder escalier elements entirely from observable data tracking streams.
    The agent solely tracks calendar properties, battery SoC, and a historical rolling window.
    """ 
    history_cols = ['NRV', 'Price'] 
    cyclic_cols = ['min_sin', 'min_cos', 'h_sin', 'h_cos', 'day_sin', 'day_cos'] 
    price_cols=[f"{i}MW" for i in range (-600,700,100)]
    
    # Re-ordering baseline non-observable arrays: NRV (idx 0), Base Price (idx 1), Escalier steps (idx 2+)
    reward_cols = ['NRV', 'Price'] + price_cols
    
    df_train = pd.read_excel(path_training).ffill().bfill() 
    df_val = pd.read_excel(path_validation).ffill().bfill() 
    df_test = pd.read_excel(path_test).ffill().bfill() 
    
    # Standardize data matrices using statistical properties of the Training track
    scaler_hist = StandardScaler() 
    scaler_hist.fit(df_train[history_cols]) 
    
    def create_windowed_dataset(df, window): 
        n_samples = len(df) - window
        data_scaled_hist = scaler_hist.transform(df[history_cols]) 
        
        history_windows = [] 
        for i in range(window, len(df)): 
            # Roll over historical slices matching window boundary from t-window up to t-1
            window_slice = data_scaled_hist[i-window : i].flatten() 
            history_windows.append(window_slice) 
        history_windows = np.array(history_windows, dtype=np.float32) 
            
        # Current time steps cyclic indicators mapping parameters
        cyclic_current = df.iloc[window:][cyclic_cols].values.astype(np.float32) 
        soc_column = np.zeros((n_samples, 1), dtype=np.float32) 
        
        # --- THE OBSERVATION VECTOR IS EXCLUSIVELY COMPOSED OF SOC, TIME, AND CAUSAL ROLLING HISTORY ---
        observable_data = np.hstack([soc_column, cyclic_current, history_windows]) 
        
        # Map non-observable reference arrays used strictly inside the environment step block
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
    # Data collection sequence to capture trajectories and feed the off-policy memory
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
        # Standard step tracking exploration loop following current network parameters weights
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
