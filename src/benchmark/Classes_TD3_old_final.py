# -*- coding: utf-8 -*-
"""
TD3 classes & functions - Reprise de travail de fin d'études (TFE)

Last Update: Mon June 1 2026

@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""
# %% -------import-------
# from __future__ import absolute_import, division, print_function

from numpy.matlib import zeros
import numpy as np
import tensorflow as tf
import tf_agents.trajectories.policy_step as ps
from tf_agents.trajectories import trajectory
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from math import floor, ceil
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class Environment(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, scaler, scalers,
        max_power: float, discount_rate: float, nb_quarters_per_episode: int,
        col_MDP: int, EP: float, eta: float, bat_replacement_cost, penalty
    ):
        super().__init__()

        # Specs - Definition of normalized action and observation spaces
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-1.1, maximum=1.1, name="observation"
        )

        # Main dataset tracks (Historical market and grid state variables)
        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        # BESS and Elia market physical/economic constants
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._SOC_index = 28 # Index mapping the SoC position within the observation vector
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._col_MDP = col_MDP
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)
        self._penalty = float(penalty)

        # Internal environment tracking states
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
        # Reset the battery State of Charge to mid-capacity (0.5) at the start of each episode
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        initial_soc = self._observation[self._SOC_index]
        # Evaluate the proposed State of Charge resulting from the agent's action
        new_soc = initial_soc + (action / (4 * self._EP_ratio))

        # Check physical boundary safety constraints (Apply virtual penalty if unfeasible)
        if 0 <= new_soc <= 1:
            penalized = 0.0
        else:
            penalized = self._penalty

        # Enforce strict physical battery constraints by clipping the SoC between 0 and 1
        new_soc = np.clip(new_soc, 0.0, 1.0)

        # Compute the adjusted power fraction matching the clipped SoC limits
        fraction_of_max_power = (new_soc - initial_soc) * (4 * self._EP_ratio)

        # Non-linear electrochemical degradation cost model (Hardware Wear-and-Tear)
        cost = 0.0
        if fraction_of_max_power < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        # Convert the operational fraction into actual battery power in MW
        battery_charge_MW = fraction_of_max_power * self._max_power

        # Retrieve the true, unobservable baseline System Imbalance (SI) volume
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1)
        )[0, 0]

        # Calculate net grid interface exchange by incorporating conversion efficiency (eta)
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        # Evaluate the price-maker effect on the final synchronized System Imbalance
        real_si = syst_imb - network_charge_MW

        # Project modified volume onto Elia's 13-point discrete price ladder merit order
        index = (-real_si / 100) + 6
        index = floor(index) if index < 6 else ceil(index)
        index = int(np.clip(index, 0, 12))

        # Reverse the MinMax normalization to extract the true clearing market price
        MP_price = self._scalers_MP[index].inverse_transform(
            np.array([[self._observation[self._col_MDP + index - 1]]], dtype=np.float32)
        )[0, 0]

        # Multi-objective reward function (Balances market arbitrage profits against degradation)
        profit = MP_price * (-network_charge_MW / 4)
        reward = profit - cost - penalized

        # Move forward to the next quarter-hour index
        self._observation_index += 1

        # Case 1: Active episode tracking (Dataset horizon limit not reached)
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]

            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        # Case 2: Reference dataset limit reached -> Trigger clean environment restart loop
        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)




class Environmentvalidation(py_environment.PyEnvironment):
    def __init__(
        self, observations, non_observable_states, scaler, scalers,
        max_power: float, discount_rate: float, nb_quarters_per_episode: int,
        col_MDP: int, EP: float, eta: float, bat_replacement_cost
    ):
        super().__init__()

        # Specs
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1.0, maximum=1.0, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(observations.shape[1],), dtype=np.float32,
            minimum=-1.1, maximum=1.1, name="observation"
        )

        # Input tracking data arrays (Stored as float32 to reduce global RAM footprint)
        self._observation_samples = np.asarray(observations, dtype=np.float32)
        self._non_observable_samples = np.asarray(non_observable_states, dtype=np.float32)

        # Validation environment setup constants (Runs without artificial tracking penalties)
        self._nb_quarters_per_episode = nb_quarters_per_episode
        self._scaler_si = scaler
        self._scalers_MP = scalers
        self._SOC_index = 28
        self._max_power = float(max_power)
        self._discount = float(discount_rate)
        self._col_MDP = col_MDP
        self._EP_ratio = float(EP)
        self._eta = float(eta)
        self._R = float(bat_replacement_cost)

        # Internal evaluation states
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

        # Reset state of charge to baseline 0.5 capacity
        self._observation_samples[self._observation_index, self._SOC_index] = 0.5
        return ts.restart(self._observation)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        # Parse power setpoint action outputted by the validation policy network
        fraction_of_max_power = float(action)

        # Update battery State of Charge metrics
        soc = self._observation[self._SOC_index]
        new_soc = soc + (fraction_of_max_power / (4 * self._EP_ratio))
        new_soc = np.clip(new_soc, 0.0, 1.0)

        # Readjust real power output based on strict hardware boundaries
        fraction_of_max_power = (new_soc - soc) * (4 * self._EP_ratio)

        # Compute aging wear costs exclusively during battery discharging phases
        cost = 0.0
        if action < 0:
            cost = self._R * 5.24e-4 * (abs(fraction_of_max_power) / 4) ** 2.03

        # Compute physical output power capacity
        battery_charge_MW = fraction_of_max_power * self._max_power

        # Invert the MinMax normalization layer to retrieve true grid system imbalance data
        syst_imb = self._scaler_si.inverse_transform(
            self._non_observable_samples[self._observation_index].reshape(1, 1)
        )[0, 0]

        # Map net power interactions injected into or withdrawn from Elia's high-voltage grid
        if battery_charge_MW > 0:
            network_charge_MW = battery_charge_MW / np.sqrt(self._eta)
        else:
            network_charge_MW = battery_charge_MW * np.sqrt(self._eta)

        real_si = syst_imb - network_charge_MW

        # Locate the clearing bracket index on the market ladder
        index = (-real_si / 100) + 6
        index = floor(index) if index < 6 else ceil(index)
        index = int(np.clip(index, 0, 12))

        MP_price = self._scalers_MP[index].inverse_transform(
            np.array([[self._observation[self._col_MDP + index - 1]]], dtype=np.float32)
        )[0, 0]

        # Formulate pure economic validation return (Excludes training safety penalties)
        profit = MP_price * (-network_charge_MW / 4)
        reward = profit - cost

        # Advance rolling operational tracking timeline index
        self._observation_index += 1

        # Case 1: Mid-episode trajectory phase
        if self._observation_index < len(self._observation_samples):
            self._observation_samples[self._observation_index, self._SOC_index] = new_soc
            self._observation = self._observation_samples[self._observation_index]

            if self._observation_index < (self._nb_quarters_per_episode * self._episode):
                return ts.transition(self._observation, reward, discount=self._discount)
            else:
                self._episode_ended = True
                return ts.termination(self._observation, reward)

        # Case 2: Validation horizon track finalized
        self._observation_index = 0
        self._episode = 0
        self._observation = self._observation_samples[self._observation_index]
        self._episode_ended = True
        return ts.termination(self._observation, reward)


# %% -------Usefull func-------
def floor(x):
    return int(np.floor(x))


def ceil(x):
    return int(np.ceil(x))

# Evaluates the root mean square deviation of price steps around the forecast target
def std_around_value(data, arbitrary_value):
    deviations = np.array(data) - arbitrary_value
    return np.sqrt(np.mean(deviations ** 2))


# %% -------Import data and preprocessing-------
def importdata(path_training, path_validation, path_test, num_future):
    LA_steps = num_future-1
    # Column placement configuration for raw Elia spreadsheet parsing
    index_col_si = 0
    index_first_col_MDP = 16
    index_last_col_MIP = index_first_col_MDP + 12  # Bounds covering all 13 price levels

    df = pd.read_excel(path_training, usecols="B:AD")

    # Allocate the blank battery tracking state column (SoC initialized at 0)
    df = df.assign(soc=np.zeros(len(df)))

    train_samples = df.to_numpy().astype("float32")
 
    # Force initial battery state of charge to 0.5 to give the agent bilateral cycling flexibility
    train_samples[0, -1] = 0.5

    df2 = pd.read_excel(path_validation, usecols="B:AD")
    df2 = df2.assign(soc=np.zeros(len(df2)))
    validation_samples = df2.to_numpy().astype("float32")
    validation_samples[0, -1] = 0.5

    df3 = pd.read_excel(path_test, usecols="B:AD")
    df3 = df3.assign(soc=np.zeros(len(df3)))
    test_samples = df3.to_numpy().astype("float32")
    test_samples[0, -1] = 0.5

    nb_rows_train = len(df)
    nb_rows_validation = len(df2)
    nb_rows_test = len(df3)
    nb_total_rows = nb_rows_train + nb_rows_validation + nb_rows_test

    # Vertical dataset concatenation to generate uniform MinMax scaling metrics
    samples = np.concatenate([train_samples, validation_samples, test_samples], axis=0)

    # Compute statistical descriptors (Mean and Standard Deviation) for System Imbalance distribution quantiles
    SI_quant_mean_and_std = zeros((nb_total_rows + LA_steps, 2), dtype='float32')


    for i in range(nb_total_rows):
        SI_quant_mean_and_std[i, 0] = np.mean(samples[i, 1:11])
        SI_quant_mean_and_std[i, 1] = np.std(samples[i, 1:11])

    for i in range(LA_steps):
        SI_quant_mean_and_std[i + nb_total_rows, 0] = SI_quant_mean_and_std[i, 0]
        SI_quant_mean_and_std[i + nb_total_rows, 1] = SI_quant_mean_and_std[i, 1]

    # Structure future tracking parameters for the privileged Look Ahead features matrix
    LA_SI_features = np.zeros((nb_total_rows, LA_steps * 2), dtype='float32')

    for i in range(nb_total_rows):
        for j in range(LA_steps):
            LA_SI_features[i, j * 2] = SI_quant_mean_and_std[i + j + 1, 0]  # Future expected mean SI
            LA_SI_features[i, j * 2 + 1] = SI_quant_mean_and_std[i + j + 1, 1]  # Future grid uncertainty std

    if LA_steps != 0:
        samples = np.hstack((samples, LA_SI_features))

    # Construct the future market pricing projection indicators for the Benchmark agent
    MP_mean_and_std = zeros((nb_total_rows + LA_steps, 2), dtype='float32')

    for i in range(nb_total_rows):
        index_MP = (-1 * SI_quant_mean_and_std[i, 0] / 100) + 6
        index_MP = np.clip(index_MP, 0, 12)

        if index_MP < 6:
            index_MP = floor(index_MP)
        else:
            index_MP = ceil(index_MP)

        MP_mean_and_std[i, 0] = samples[i, index_MP + index_first_col_MDP]  # Future target price step
        MP_mean_and_std[i, 1] = std_around_value(samples[i, index_first_col_MDP: index_last_col_MIP],
                                                 MP_mean_and_std[i, 0])  # Local price volatility variance

    for i in range(LA_steps):
        MP_mean_and_std[i + nb_total_rows, 0] = MP_mean_and_std[i, 0]
        MP_mean_and_std[i + nb_total_rows, 1] = MP_mean_and_std[i, 1]

    LA_MP_features = np.zeros((nb_total_rows, LA_steps * 2), dtype='float32')

    for i in range(nb_total_rows):
        for j in range(LA_steps):
            LA_MP_features[i, j * 2] = MP_mean_and_std[i + j + 1, 0]  # Look-ahead expected price
            LA_MP_features[i, j * 2 + 1] = MP_mean_and_std[i + j + 1, 1]  # Look-ahead volatility spread

    if LA_steps != 0:
        samples = np.hstack((samples, LA_MP_features))
        

    # Separate out the true, realized System Imbalance vector (Hidden state variable)
    non_observable_exact_si = samples[:, index_col_si]

    # Drop the unobservable raw SI data column from the direct inputs matrix
    observable_samples = np.delete(samples, index_col_si, 1)

    scaled_observable_samples = observable_samples

    # Scale down the system imbalance values uniformly within a stable (-1, 1) range
    scaler_si = MinMaxScaler(feature_range=(-1, 1))  
    scaled_si = scaler_si.fit_transform(non_observable_exact_si.reshape(-1, 1))

    scaler_list = []
    # Loop over observable market signals to normalize variables while skipping cyclic calendar features
    for j in np.concatenate((np.arange(11), np.arange(index_first_col_MDP - 1, index_last_col_MIP),
                             np.arange(index_last_col_MIP + 1, index_last_col_MIP + (4 * LA_steps) + 1))):
        scaler_MP = MinMaxScaler(feature_range=(0, 1))
        scaled_MP = scaler_MP.fit_transform(observable_samples[:, j].reshape(-1, 1))
        scaled_observable_samples[:, j] = scaled_MP[:, 0]
        if index_first_col_MDP - 1 <= j <= index_last_col_MIP - 1:
            scaler_list.append(scaler_MP)

    # Restructure data structures back into independent training, validation, and testing arrays
    train_observable_samples = scaled_observable_samples[0:nb_rows_train, :]
    validation_observable_samples = scaled_observable_samples[nb_rows_train:nb_rows_train + nb_rows_validation, :]
    test_observable_samples = scaled_observable_samples[
                              nb_rows_train + nb_rows_validation:nb_rows_train + nb_rows_validation + nb_rows_test, :]

    train_non_observable_samples = scaled_si[0:nb_rows_train, :]
    validation_non_observable_samples = scaled_si[nb_rows_train:nb_rows_train + nb_rows_validation, :]
    test_non_observable_samples = scaled_si[
                                  nb_rows_train + nb_rows_validation:nb_rows_train + nb_rows_validation + nb_rows_test,
                                  :]


    return index_col_si, index_first_col_MDP, scaler_si, scaler_list, train_observable_samples, validation_observable_samples, \
        test_observable_samples, train_non_observable_samples, validation_non_observable_samples, test_non_observable_samples






# %% -------Data Collection-------
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
# !!!! we provide a policy, not an agent => no cvxpy corrections !!!!!!!
# /!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\/!\
def collect_data(environment, policy, buffer, steps, initial):
    # Sequential trajectory exploration loop to populate the TD3 off-policy Replay Memory Buffer
    reward = 0
    unfeas_action = []

    # Enforce a non-policy exploration step size for initial data variance stability
    x = 16 * 20

    arbitrary_actions = [-1, -1, -0.5, 1, 1, 1, 1, 1, 1, 0.1, -1, -1, -1, -1, 0.5, -0.5]    # Script containing non-physical exploration steps
    # arbitrary_actions = [-1, -0.5, -0.5, 1, 0.5, 0.75, 0.25, 0.5, 1, -0.1, -0.7, -0.2, 0.8, -1, 0.5, -0.5]    # Script containing physical actions

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

                # Store exploration interaction trajectory transitions inside the Replay memory
                buffer.add_batch(traj)
                if next_time_step.is_last():
                    environment.reset()

        for _ in range(steps - x):

            time_step = environment.current_time_step()
            reward = time_step.reward
            action_step = policy.action(time_step)

            observation = time_step.observation
            crt_SoCs = tf.gather(observation, indices=[28], axis=1)  # Extract current State of Charge features

            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            buffer.add_batch(traj)
            if next_time_step.is_last():
                environment.reset()

    else:
        # Standard continuous background collection loop executing the current actor network policy parameters
        for _ in range(steps):

            time_step = environment.current_time_step()
            reward += time_step.reward
            action_step = policy.action(time_step)

            observation = time_step.observation

            crt_SoCs = tf.gather(observation, indices=[28], axis=1) 

            next_time_step = environment.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)

            buffer.add_batch(traj)
            if next_time_step.is_last():
                environment.reset()

    return reward, unfeas_action

print(" ")
print("---------- Classes defined ----------")
print(" ")
