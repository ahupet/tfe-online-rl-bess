# Empirical Price Ladder Reconstruction Submodule

This directory houses the statistical preprocessing algorithms used to reconstruct historical price trends during macroeconomic shifts and structural crises, preventing "cold-start" distribution tracking issues.

## 📂 Directory Contents

* **`price_ladder_reconstruction.py`**: The core V3 generation script. It aggregates historical tracks, applies nearest-neighbor horizontal spatial interpolations, enforces strict monotonicity bounds (`cummin` / `cummax` ordering), and maps entries to the matching V5 environment pricing indexes.

## 📥 Required Fichier Dependencies
To run this reconstruction script, drop the following external spreadsheets directly into this directory:
1. **`Dataset_Combined_2018_2019.xlsx`**: Stable reference tracks used as a statistical bootstrap (the "Amorce") to feed initial 2020 boundary states.
2. **`Dataset_2020_2024.xlsx`**: The raw, volatile target dataset covering the European energy crisis horizon.

The output will be automatically saved as a crisis-adapted multi-column spreadsheet file ready for the long-term evaluation layers.
