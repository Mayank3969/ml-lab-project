<!-- README.md: Column dictionary, source link, and join keys for collision, vehicle, and casualty datasets. -->
# Dataset Documentation

## Overview
This directory contains the raw and processed datasets used in the collision severity prediction pipeline:
- `raw/collision.csv`
- `raw/vehicle.csv`
- `raw/casualty.csv`
- `processed/merged_train.csv` (joined 3 tables, leakage cols dropped, encoded, ~64k rows)
- `processed/merged_test.csv` (same as merged_train, ~16k rows, held out from start)
- `processed/meta_train.csv` (X_meta_train [64k x 13] + y_meta_train — Phase A output)
- `processed/meta_test.csv` (X_meta_test [16k x 13] — Phase B output)

## Join Keys
- Primary join identifier linking collision, vehicle, and casualty tables: `collision_id` / `accident_index`.

## Data Dictionary & Leakage Prevention
- Target leakage columns dropped before training.
- Categorical features encoded for regression and classification models.
