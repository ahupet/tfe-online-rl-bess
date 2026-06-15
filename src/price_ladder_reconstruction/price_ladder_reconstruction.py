# -*- coding: utf-8 -*-
"""
Empirical Price Ladder Generator (2020-2024) - Calibrated Bootstrapped Setup (V3)
- Leverages 2018-2019 data tracks as a bootstrap to completely avoid cold-start issues on Jan 1st, 2020.
- Categorization boundaries perfectly mapped to the Agent's V5 mathematical logic (Ceil / Floor).
- Enforces strict balancing constraints on the 0 MW tier = mean(-100MW, +100MW).

Last Update: Mon June 1 2026
@author: Achille Hupet
Institution: Faculté Polytechnique de Mons (UMONS)
"""

import pandas as pd
import numpy as np

# ==========================================
# SIMULATION HORIZON CONFIGURATIONS
# ==========================================
# Target files to be dropped manually in your project directory
FICHIER_AMORCE = 'Dataset_Combined_2018_2019.xlsx' 
FICHIER_2020 = 'Dataset_2020_2024.xlsx'
FICHIER_SORTIE = 'Dataset_2020_2024_Escalier_Empirique_Am_6h_V3.xlsx' 

FREQUENCE = '6h' 

# Define the 13 discrete price ladder capacity bands matching the merit-order curve
marches_noms = [f"{i}MW" for i in range(-600, 700, 100)] 

# ==========================================
# STEP 1: IMPORT AND VERTICAL CONCATENATION
# ==========================================
print("Loading bootstrapping historical data (2018-2019) and target tracking window (2020-2024)...")
df_1819 = pd.read_excel(FICHIER_AMORCE)
df_2024 = pd.read_excel(FICHIER_2020)

# Standardize date column representations into datetime parameters
df_1819['FROM_DATE'] = pd.to_datetime(df_1819['FROM_DATE'])
df_2024['FROM_DATE'] = pd.to_datetime(df_2024['FROM_DATE'])

# Enforce continuum boundary layers: Dec 31st 2019 values will smoothly populate Jan 1st 2020 slices
df_complet = pd.concat([df_1819, df_2024], ignore_index=True)

# ==========================================
# STEP 2: CATEGORIZATION BOUNDARIES (STRICT V5 AGENT LOGIC)
# ==========================================
print(f"Computing empirical price ladders across rolling horizons of {FREQUENCE}...")

# 1. Re-align categorization checks with the exact mathematical indexing formula of the agent
nrv_scaled = (df_complet['NRV'] / 100) + 6
idx_mod = np.where(nrv_scaled < 6, np.floor(nrv_scaled), np.ceil(nrv_scaled))
idx_mod = np.clip(idx_mod, 0, 12).astype(int)

# 2. Append bracket keys and floor time signatures to matching rolling frequencies blocks
df_complet['Palier'] = np.array(marches_noms)[idx_mod]
df_complet['Time_Block'] = df_complet['FROM_DATE'].dt.floor(FREQUENCE)

# Group by unique temporal blocks and extract the median clearing price structure
escaliers_bruts = df_complet.groupby(['Time_Block', 'Palier'], observed=False)['Price'].median().unstack()

# Integrity safeguard: Enforce the presence of all 13 price dimensions columns
escaliers_bruts = escaliers_bruts.reindex(columns=marches_noms)

# ==========================================
# STEP 3: MISSING BRACKETS INTERPOLATION
# ==========================================
# 1. Execute directional propagation: Null metrics in 2020 retrieve late 2019 parameters history
escaliers_remplis = escaliers_bruts.ffill().bfill()

# 2. Enforce structural pricing balancing constraints for the null-exchange band (0 MW)
escaliers_remplis['0MW'] = (escaliers_remplis['-100MW'] + escaliers_remplis['100MW']) / 2.0

# 3. Horizontal spatial interpolation pass
# Cast column headers temporary to real numeric integers to form a true mathematical X-axis
escaliers_remplis.columns = [int(str(col).replace('MW', '')) for col in escaliers_remplis.columns]

# Execute horizontal nearest-neighbor spatial fill updates matching grid dimensions axis
escaliers_remplis = escaliers_remplis.interpolate(axis=1, method='nearest', limit_direction='both')

# Restore original string structural nomenclature keys ("-600MW", etc.)
escaliers_remplis.columns = marches_noms

# ==========================================
# STEP 4: MONOTONICITY CONSTRAINTS FILTERING
# ==========================================
# 1. Backward processing pass (Right to Left): A tier cannot be more expensive than its right-hand neighbor
df_escaliers_propres = escaliers_remplis.iloc[:, ::-1].cummin(axis=1).iloc[:, ::-1]

# 2. Forward processing pass (Left to Right): A tier cannot be cheaper than its left-hand neighbor
df_escaliers_propres = df_escaliers_propres.cummax(axis=1)

# ==========================================
# STEP 5: REMOVE STRATEGIC BOOTSTRAP BLOCKS AND MERGE
# ==========================================
print("Extracting target output records strictly bounded between 2020 and 2024...")

# Drop preexisting placeholder columns inside the 2020-2024 dataframe template
colonnes_a_garder = [col for col in df_2024.columns if col not in marches_noms]
df_propre_2024 = df_2024[colonnes_a_garder].copy()
df_propre_2024['Time_Block'] = df_propre_2024['FROM_DATE'].dt.floor(FREQUENCE)

# Merge reconstructed columns matching solely the target output dataframe horizon timelines
df_final = pd.merge(df_propre_2024, df_escaliers_propres, left_on='Time_Block', right_index=True, how='left')
df_final.drop(columns=['Time_Block'], inplace=True)

# ==========================================
# OUTPUT FILE GENERATION LOGGING
# ==========================================
print(f"Saving finalized dataset profile to: {FICHIER_SORTIE}...")
df_final.to_excel(FICHIER_SORTIE, index=False)
print("-> Done! Your crisis-adapted empirical price ladder dataset is fully processed.")
