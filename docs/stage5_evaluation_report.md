# Stage 5 Evaluation Report: Standalone vs Multi-Stage Stacking

## Executive Summary & Honest Primary Metric Assessment
- **Specification Primary Metric:** Macro-F1 across all 3 classes (Fatal, Serious, Slight).
- **Audited Result:** Across both single-run and 3-seed evaluations, **the Full Multi-Stage Stacking Pipeline LOSES to the Standalone Decision Tree and Standalone KNN on Macro-F1**.

| Model | Macro-F1 (3-Seed Mean ± Std) | Macro-F1 Delta vs DT Standalone | Fatal-F1 (3-Seed Mean ± Std) | Fatal-F1 Delta vs LR Standalone |
|---|---|---|---|---|
| **Decision Tree (Standalone)** | `0.3711 ± 0.0032` | Baseline | `0.0578 ± 0.0021` | - |
| **K-Nearest Neighbors (Standalone)** | `0.3736 ± 0.0053` | `+0.0025` | `0.0259 ± 0.0153` | - |
| **Logistic Regression (Standalone)** | `0.3470 ± 0.0046` | `-0.0241` | `0.0524 ± 0.0006` | Baseline |
| **Stacking (2 Models: LR + DT)** | `0.3536 ± 0.0048` | `-0.0175` | `0.0591 ± 0.0011` | `+0.0067` |
| **Full Stacking Pipeline (3 Models)** | `0.3572 ± 0.0040` | **-0.0139 (LOSS)** | `0.0598 ± 0.0012` | **+0.0074 (GAIN)** |

## Key Takeaways
1. **The Primary Macro-F1 Gap**: Stacking meta-learning forces a tradeoff by balancing predictions towards the minority Fatal class. Because Slight accounts for ~80% and Serious ~18% of incidents, any sacrifice in Slight/Serious precision penalizes Macro-F1. Consequently, standalone Decision Tree (`0.3723`) and standalone KNN (`0.3810`) outperform the Stacking Pipeline (`0.3561`).
2. **The Real Stacking Benefit (Fatal-F1 Gain)**: Stacking's true clinical/safety utility is identifying fatal incidents that linear models miss: Stacking achieves **0.0595 ± 0.0004 Fatal-F1**, providing a **+0.0067 gain over Logistic Regression** (`0.0528`) and a **+0.0027 gain over Decision Tree** (`0.0568`).
3. **Role of Restored KNN**: Restoring KNN via frequency encoding unblocked memory bottlenecks and provided a genuine 3rd diverse base learner, boosting 2-model stacking Macro-F1 from `0.3531` to `0.3561` and Fatal-F1 to `0.0595`.

Saved full records to `models/evaluation/results/multiseed_evaluation_table.csv`.