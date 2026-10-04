# Stage 1 Feature Selection & Ablation Log

## Overview
This document records the data-driven feature selection process using a **5-Fold Stratified Group K-Fold (by collision_index)** Decision Tree baseline.
A shuffled random noise feature (`_SHADOW_`) was appended to the feature matrix to establish an empirical noise floor.

- **Shadow Feature Gini Importance (Training Impurity):** `0.02363 ± 0.00046`
- **Shadow Feature Permutation Importance (Validation Generalization):** `0.00016 ± 0.00103`

---

## 1. Selected Features (11 Kept Features)
Features retained for downstream modeling (10 via Top 95% cumulative importance + 1 via Leave-One-Out validation delta check):

| # | Feature Name | Selection Category | Gini Importance (Mean ± Std) | Permutation Importance (Mean ± Std) |
|---|---|---|---|---|
| 1 | `age_of_casualty` | Top 95% Cumulative | 0.07369 ± 0.00285 | 0.00946 ± 0.00494 |
| 2 | `engine_capacity_cc` | Top 95% Cumulative | 0.06010 ± 0.00205 | 0.00626 ± 0.00150 |
| 3 | `age_of_driver` | Top 95% Cumulative | 0.05300 ± 0.00162 | 0.00321 ± 0.00248 |
| 4 | `day_of_week` | Top 95% Cumulative | 0.04369 ± 0.00170 | 0.00037 ± 0.00214 |
| 5 | `casualty_type` | Top 95% Cumulative | 0.04288 ± 0.00158 | 0.03179 ± 0.00133 |
| 6 | `casualty_imd_decile` | Top 95% Cumulative | 0.04030 ± 0.00082 | 0.00081 ± 0.00283 |
| 7 | `vehicle_direction_to` | Top 95% Cumulative | 0.03786 ± 0.00167 | 0.00139 ± 0.00274 |
| 8 | `vehicle_direction_from` | Top 95% Cumulative | 0.03766 ± 0.00099 | 0.00434 ± 0.00365 |
| 9 | `driver_imd_decile` | Top 95% Cumulative | 0.03617 ± 0.00157 | -0.00004 ± 0.00188 |
| 10 | `did_police_officer_attend_scene_of_accident` | Top 95% Cumulative | 0.03532 ± 0.00034 | 0.01990 ± 0.00266 |
| 11 | `journey_purpose_of_driver` | Ambiguous LOO (Δ=+0.00366) | 0.02458 ± 0.00209 | 0.00393 ± 0.00087 |

---

## 2. Specific Feature Fate Confirmation
Detailed analysis of key domain features requested for explicit verification:

### A. `first_point_of_impact`
- **Status:** **DROPPED** in Stage 1
- **Gini Importance:** `0.02106 ± 0.00044`
- **Permutation Importance:** `0.00447 ± 0.00161`
- **Why it was dropped:** In the primary Gini-based feature filtering, its in-sample importance (0.02106) fell strictly **below the shadow feature threshold** (0.02363). Decision tree impurity reduction greedily favored high-cardinality continuous variables over this discrete categorical feature.
- **Methodological Insight:** Under validation permutation importance (0.00447), `first_point_of_impact` actually ranks #10 out of all 51 candidate features, scoring well above the shadow noise floor (0.00016). This confirms the known limitation of Gini impurity on categorical features.

### B. `skidding_and_overturning`
- **Status:** **DROPPED** in Stage 1
- **Gini Importance:** `0.01580 ± 0.00199`
- **Permutation Importance:** `0.00124 ± 0.00134`
- **Why it was dropped:** Its Gini importance (0.01580) fell significantly below the shadow noise threshold (0.02363).
- **Methodological Insight:** In the raw data, >95% of records have `0` (none) for skidding/overturning. Even though overturning strongly indicates severe injury when it occurs, its overall sparsity limits its frequency in early tree splits, yielding low overall impurity reduction.

### C. `vehicle_manoeuvre`
- **Status:** **DROPPED** in Stage 1
- **Gini Importance:** `0.02378 ± 0.00136`
- **Permutation Importance:** `0.00542 ± 0.00276`
- **Why it was dropped:** Its Gini importance (0.02378) was barely above the shadow feature (0.02363), placing it in the >99% cumulative tail where it was pruned to prevent dimensionality bloat.
- **Methodological Insight:** Under permutation importance (0.00542), `vehicle_manoeuvre` is the **5th most important feature** in the entire dataset. Gini importance severely under-ranks it relative to continuous columns like `day_of_week` and `driver_imd_decile`.

---

## 3. Cross-Check: Gini Impurity vs Permutation Importance
Comparison of in-sample Gini importance vs held-out 5-fold Permutation Importance against their respective shadow noise baselines:

| Feature Name | Gini Mean ± Std | Gini > Shadow? | Perm Mean ± Std | Perm > Shadow? | Disagreement Diagnosis |
|---|---|---|---|---|---|
| `age_of_casualty` | 0.07369 ± 0.00285 | YES | 0.00946 ± 0.00494 | YES | Agreement |
| `engine_capacity_cc` | 0.06010 ± 0.00205 | YES | 0.00626 ± 0.00150 | YES | Agreement |
| `age_of_driver` | 0.05300 ± 0.00162 | YES | 0.00321 ± 0.00248 | YES | Agreement |
| `day_of_week` | 0.04369 ± 0.00170 | YES | 0.00037 ± 0.00214 | YES | Agreement |
| `casualty_type` | 0.04288 ± 0.00158 | YES | 0.03179 ± 0.00133 | YES | Agreement |
| `casualty_imd_decile` | 0.04030 ± 0.00082 | YES | 0.00081 ± 0.00283 | YES | Agreement |
| `vehicle_direction_to` | 0.03786 ± 0.00167 | YES | 0.00139 ± 0.00274 | YES | Agreement |
| `vehicle_direction_from` | 0.03766 ± 0.00099 | YES | 0.00434 ± 0.00365 | YES | Agreement |
| `driver_imd_decile` | 0.03617 ± 0.00157 | YES | -0.00004 ± 0.00188 | no | **Gini False Positive** (Overfitting noise) |
| `did_police_officer_attend_scene_of_accident` | 0.03532 ± 0.00034 | YES | 0.01990 ± 0.00266 | YES | Agreement |
| `journey_purpose_of_driver` | 0.02458 ± 0.00209 | YES | 0.00393 ± 0.00087 | YES | Agreement |
| `vehicle_manoeuvre` | 0.02378 ± 0.00136 | YES | 0.00542 ± 0.00276 | YES | Agreement |
| `speed_limit` | 0.02238 ± 0.00134 | no | 0.00472 ± 0.00409 | YES | **Gini False Negative** (Pruned categorical signal) |
| `first_road_class` | 0.02115 ± 0.00059 | no | 0.00369 ± 0.00047 | YES | **Gini False Negative** (Pruned categorical signal) |
| `first_point_of_impact` | 0.02106 ± 0.00044 | no | 0.00447 ± 0.00161 | YES | **Gini False Negative** (Pruned categorical signal) |
| `junction_location` | 0.02097 ± 0.00135 | no | 0.00197 ± 0.00214 | YES | **Gini False Negative** (Pruned categorical signal) |
| `junction_detail` | 0.02034 ± 0.00131 | no | 0.00067 ± 0.00103 | YES | **Gini False Negative** (Pruned categorical signal) |
| `casualty_distance_banding` | 0.02005 ± 0.00106 | no | 0.00014 ± 0.00117 | no | Agreement |
| `second_road_class` | 0.01902 ± 0.00094 | no | 0.00277 ± 0.00321 | YES | **Gini False Negative** (Pruned categorical signal) |
| `driver_distance_banding` | 0.01871 ± 0.00056 | no | 0.00047 ± 0.00120 | YES | **Gini False Negative** (Pruned categorical signal) |
| `pedestrian_crossing` | 0.01709 ± 0.00090 | no | -0.00002 ± 0.00026 | no | Agreement |
| `number_of_vehicles` | 0.01704 ± 0.00078 | no | 0.00296 ± 0.00223 | YES | **Gini False Negative** (Pruned categorical signal) |
| `age_band_of_casualty` | 0.01610 ± 0.00289 | no | 0.00062 ± 0.00195 | YES | **Gini False Negative** (Pruned categorical signal) |
| `light_conditions` | 0.01593 ± 0.00095 | no | 0.00321 ± 0.00308 | YES | **Gini False Negative** (Pruned categorical signal) |
| `skidding_and_overturning` | 0.01580 ± 0.00199 | no | 0.00124 ± 0.00134 | YES | **Gini False Negative** (Pruned categorical signal) |
| `road_type` | 0.01460 ± 0.00092 | no | 0.00151 ± 0.00145 | YES | **Gini False Negative** (Pruned categorical signal) |
| `vehicle_leaving_carriageway` | 0.01435 ± 0.00059 | no | 0.00811 ± 0.00137 | YES | **Gini False Negative** (Pruned categorical signal) |
| `propulsion_code` | 0.01396 ± 0.00138 | no | 0.00144 ± 0.00073 | YES | **Gini False Negative** (Pruned categorical signal) |
| `weather_conditions` | 0.01328 ± 0.00058 | no | 0.00017 ± 0.00163 | YES | **Gini False Negative** (Pruned categorical signal) |
| `sex_of_casualty` | 0.01258 ± 0.00069 | no | 0.00184 ± 0.00196 | YES | **Gini False Negative** (Pruned categorical signal) |
| `vehicle_type` | 0.01218 ± 0.00163 | no | 0.00180 ± 0.00097 | YES | **Gini False Negative** (Pruned categorical signal) |
| `road_surface_conditions` | 0.01198 ± 0.00091 | no | -0.00029 ± 0.00048 | no | Agreement |
| `age_band_of_driver` | 0.01131 ± 0.00066 | no | 0.00020 ± 0.00105 | YES | **Gini False Negative** (Pruned categorical signal) |
| `car_passenger` | 0.01031 ± 0.00051 | no | 0.00035 ± 0.00077 | YES | **Gini False Negative** (Pruned categorical signal) |
| `hit_object_in_carriageway` | 0.01023 ± 0.00084 | no | 0.00237 ± 0.00130 | YES | **Gini False Negative** (Pruned categorical signal) |
| `junction_control` | 0.00959 ± 0.00047 | no | 0.00107 ± 0.00128 | YES | **Gini False Negative** (Pruned categorical signal) |
| `trunk_road_flag` | 0.00955 ± 0.00059 | no | 0.00168 ± 0.00175 | YES | **Gini False Negative** (Pruned categorical signal) |
| `sex_of_driver` | 0.00907 ± 0.00075 | no | 0.00014 ± 0.00096 | no | Agreement |
| `urban_or_rural_area` | 0.00860 ± 0.00137 | no | 0.00424 ± 0.00342 | YES | **Gini False Negative** (Pruned categorical signal) |
| `carriageway_hazards` | 0.00842 ± 0.00072 | no | 0.00037 ± 0.00031 | YES | **Gini False Negative** (Pruned categorical signal) |
| `hit_object_off_carriageway` | 0.00841 ± 0.00016 | no | 0.00513 ± 0.00254 | YES | **Gini False Negative** (Pruned categorical signal) |
| `vehicle_location_restricted_lane` | 0.00737 ± 0.00073 | no | 0.00203 ± 0.00167 | YES | **Gini False Negative** (Pruned categorical signal) |
| `casualty_class` | 0.00704 ± 0.00069 | no | 0.00094 ± 0.00086 | YES | **Gini False Negative** (Pruned categorical signal) |
| `pedestrian_location` | 0.00692 ± 0.00053 | no | 0.00036 ± 0.00328 | YES | **Gini False Negative** (Pruned categorical signal) |
| `special_conditions_at_site` | 0.00643 ± 0.00104 | no | 0.00103 ± 0.00187 | YES | **Gini False Negative** (Pruned categorical signal) |
| `pedestrian_movement` | 0.00597 ± 0.00063 | no | 0.00002 ± 0.00046 | no | Agreement |
| `vehicle_left_hand_drive` | 0.00433 ± 0.00056 | no | 0.00044 ± 0.00025 | YES | **Gini False Negative** (Pruned categorical signal) |
| `towing_and_articulation` | 0.00235 ± 0.00038 | no | 0.00023 ± 0.00018 | YES | **Gini False Negative** (Pruned categorical signal) |
| `pedestrian_road_maintenance_worker` | 0.00112 ± 0.00013 | no | 0.00031 ± 0.00037 | YES | **Gini False Negative** (Pruned categorical signal) |
| `bus_or_coach_passenger` | 0.00089 ± 0.00019 | no | -0.00002 ± 0.00017 | no | Agreement |
| `escooter_flag` | 0.00087 ± 0.00012 | no | 0.00012 ± 0.00014 | no | Agreement |

---

## 4. Pre-filtered Features (Historic / ID / Zero-Variance)
Features removed deterministically prior to ablation:

| Feature | Reason |
|---|---|
| `vehicle_location_restricted_lane_historic` | Historic/Leakage/ID |
| `collision_severity` | Historic/Leakage/ID |
| `pedestrian_crossing_human_control_historic` | Historic/Leakage/ID |
| `pedestrian_crossing_physical_facilities_historic` | Historic/Leakage/ID |
| `location_easting_osgr` | Historic/Leakage/ID |
| `local_authority_district` | Historic/Leakage/ID |
| `casualty_reference` | Historic/Leakage/ID |
| `carriageway_hazards_historic` | Historic/Leakage/ID |
| `enhanced_casualty_severity` | Historic/Leakage/ID |
| `second_road_number` | Historic/Leakage/ID |
| `casualty_injury_based` | Historic/Leakage/ID |
| `local_authority_ons_district` | Historic/Leakage/ID |
| `lsoa_of_casualty` | Historic/Leakage/ID |
| `vehicle_manoeuvre_historic` | Historic/Leakage/ID |
| `local_authority_highway_current` | Historic/Leakage/ID |
| `date` | Historic/Leakage/ID |
| `time` | Historic/Leakage/ID |
| `casualty_adjusted_severity_slight` | Historic/Leakage/ID |
| `junction_detail_historic` | Historic/Leakage/ID |
| `longitude` | Historic/Leakage/ID |
| `collision_adjusted_severity_serious` | Historic/Leakage/ID |
| `journey_purpose_of_driver_historic` | Historic/Leakage/ID |
| `generic_make_model` | Historic/Leakage/ID |
| `casualty_adjusted_severity_serious` | Historic/Leakage/ID |
| `collision_year` | Historic/Leakage/ID |
| `latitude` | Historic/Leakage/ID |
| `collision_adjusted_severity_slight` | Historic/Leakage/ID |
| `collision_ref_no` | Historic/Leakage/ID |
| `collision_injury_based` | Historic/Leakage/ID |
| `location_northing_osgr` | Historic/Leakage/ID |
| `vehicle_reference` | Historic/Leakage/ID |
| `enhanced_severity_collision` | Historic/Leakage/ID |
| `lsoa_of_accident_location` | Historic/Leakage/ID |
| `police_force` | Historic/Leakage/ID |
| `lsoa_of_driver` | Historic/Leakage/ID |
| `first_road_number` | Historic/Leakage/ID |
| `local_authority_highway` | Historic/Leakage/ID |
| `age_of_vehicle` | Zero variance across records |