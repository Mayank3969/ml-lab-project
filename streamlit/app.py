"""
app.py — Master Streamlit Dashboard for Collision Severity Prediction
Interactive Web Application featuring:
1. End-to-End System Architecture & Data Flow Diagram
2. Mathematical Loss Functions & Theory for all stages
3. Benchmark Performance Comparisons & Confusion Matrix
4. Live Single Incident Custom Prediction
5. Bulk File Upload & Batch Testing (CSV / Excel)
6. Pipeline Execution Audit Logs
"""
import sys
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "streamlit"))

from inference import predict_pipeline, predict_single, get_models, META_FEATURES, BASE_CLASSIFIER_FEATURES
from mappings import (
    CASUALTY_CLASS_MAP, CASUALTY_TYPE_MAP, DAY_OF_WEEK_MAP, ROAD_CLASS_MAP,
    ROAD_TYPE_MAP, SPEED_LIMIT_OPTIONS, JUNCTION_DETAIL_MAP, JUNCTION_CONTROL_MAP,
    LIGHT_CONDITIONS_MAP, WEATHER_CONDITIONS_MAP, ROAD_SURFACE_CONDITIONS_MAP,
    SPECIAL_CONDITIONS_MAP, CARRIAGEWAY_HAZARDS_MAP, URBAN_RURAL_MAP,
    SEVERITY_LABELS, SEVERITY_COLORS, SEVERITY_ICONS
)
from styles import get_custom_css

# Page Configuration
st.set_page_config(
    page_title="UK Collision Severity AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar Navigation & Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🚦 Navigation")
    menu = st.radio(
        "Select Dashboard Section:",
        [
            "📐 System Architecture & Flow",
            "🔬 Model Theory & Loss Functions",
            "📊 Model Comparisons & Metrics",
            "🎯 Single Incident Prediction",
            "📁 Batch CSV / Excel Testing",
            "📋 Pipeline Audit Logs"
        ],
        index=3  # Default to Live Prediction
    )
    st.markdown("---")
    
    # Model Status Check
    models = get_models()
    if models.loaded:
        st.success("✅ Models Active (All 5 Saved PKLs Loaded)")
    else:
        st.error(f"❌ Models Not Found: {models.error}")
        st.info("Run `python pipeline.py` in terminal to train and serialize models.")

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:#94a3b8;'>"
        "<b>UK STATS19 Accident Dataset</b><br>"
        "Target: <code>casualty_severity</code><br>"
        "Classes: 1 (Fatal), 2 (Serious), 3 (Slight)<br>"
        "Anti-Leakage Stacking Ensemble"
        "</div>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# SECTION 1: System Architecture & Data Flow Diagram
# ---------------------------------------------------------------------------
if menu == "📐 System Architecture & Flow":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">System Flow & Pipeline Architecture</div>
        <div class="hero-subtitle">
            A comprehensive overview of how UK road collision data flows through the 5-stage stacked ensemble pipeline — from raw multi-table joins to feature engineering, out-of-fold cross validation, and meta-classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔄 End-to-End Data Flow Diagram")
    
    # Render interactive Mermaid Architecture Diagram
    mermaid_code = """
    graph TD
        subgraph Raw Data Ingestion
            C[collision.csv] --> M[Merge & Join Casualty Level]
            V[vehicle.csv] --> M
            A[casualty.csv] --> M
        end

        subgraph Stage 1: Preprocessing & Anti-Leakage
            M --> D1[Drop 9 Target Leakage Columns]
            D1 --> D2[Drop 65 Irrelevant / Memorization Columns]
            D2 --> GSS[GroupShuffleSplit on collision_index 80/20]
            GSS --> TR[merged_train.csv 93,975 rows]
            GSS --> TE[merged_test.csv 23,528 rows]
        end

        subgraph Stage 2: Collision Risk Regression
            TR --> REG[Linear Regression vs Decision Tree]
            REG --> |R2 = 0.0232 Winner: Linear Regression| RK[5-Fold OOF Predictions]
            RK --> RS[Engineered risk_score Feature]
            RS --> TR2[merged_train with risk_score 19 cols]
            RS --> TE2[merged_test with risk_score 19 cols]
        end

        subgraph Stage 3A: Base Classifiers Stacking
            TR2 --> |Phase A: 5-Fold StratifiedGroupKFold| OOF[Out-of-Fold Probability Generator]
            OOF --> |Models Destroyed Each Fold| MT[meta_train.csv 93,975 x 12 cols]
            TR2 --> |Phase B: Fit on Full Train| FB[Final Base Models: LogReg + KNN + DT]
            TE2 --> |Predict Proba on Held-Out Test| FB
            FB --> MTEST[meta_test.csv 23,528 x 12 cols]
            FB --> PKL[Serialized Final Models stage2_*.pkl]
        end

        subgraph Stage 4: Meta-Classifier
            MT --> |10 Meta-Features: 9 Probas + risk_score| META[Balanced Meta-Learner LogisticRegression]
            MTEST --> META
            META --> FIN[Final Severity Prediction: Fatal / Serious / Slight]
            META --> PKL2[stage3_meta_classifier.pkl]
        end

        subgraph Stage 5: Inference Application
            FIN --> APP[Streamlit Interactive Inference Web App]
        end

        classDef raw fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#fff;
        classDef stage fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#fff;
        classDef meta fill:#1e1b4b,stroke:#a855f7,stroke-width:2px,color:#fff;
        classDef out fill:#064e3b,stroke:#059669,stroke-width:2px,color:#fff;

        class C,V,A raw;
        class M,D1,D2,GSS,REG,RK,OOF,FB stage;
        class META meta;
        class FIN,APP out;
    """
    
    st.markdown(f"```mermaid\n{mermaid_code}\n```")

    st.markdown("---")
    st.markdown("### 🧩 Deep Dive: The 5 Core Pipeline Stages")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <span class="flow-pill flow-pill-blue">Stage 1</span>
            <h4>Merge, Clean & Group-Aware Split</h4>
            <p><b>Data Source:</b> Three disparate UK STATS19 tables joined at the casualty level:</p>
            <ul>
                <li><code>collision.csv</code>: Environmental conditions, road characteristics, weather, speed limits.</li>
                <li><code>vehicle.csv</code>: Vehicle types, manoeuvres, impacts.</li>
                <li><code>casualty.csv</code>: Casualty type, class (driver/passenger/pedestrian), severity target.</li>
            </ul>
            <p><b>Crucial Leakage Prevention:</b></p>
            <ul>
                <li>Dropped <b>9 target-derived columns</b> (e.g., <code>enhanced_casualty_severity</code>, <code>collision_severity</code>).</li>
                <li>Group-aware train/test split via <code>GroupShuffleSplit</code> grouped on <code>collision_index</code> (same multi-casualty crash never split across train and test).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <span class="flow-pill flow-pill-emerald">Stage 3A</span>
            <h4>Base Classifiers & Anti-Leakage OOF Stacking</h4>
            <p>Three complementary machine learning algorithms are utilized:</p>
            <ul>
                <li><b>Logistic Regression:</b> Calibrated linear probabilistic boundaries with balanced class weighting.</li>
                <li><b>K-Nearest Neighbors (k=15):</b> Non-parametric local density estimation with inverse-distance weighting.</li>
                <li><b>Decision Tree (depth=8):</b> Non-linear orthogonal decision boundaries capturing interactions.</li>
            </ul>
            <p><b>Phase A (Train-side):</b> 5-Fold <code>StratifiedGroupKFold</code> out-of-fold cross-validation. Every training row receives probabilities from a model that never saw it during fitting. All 15 fold-models are immediately destroyed.</p>
            <p><b>Phase B (Test-side):</b> Final models fitted on the full training set and evaluated on held-out test data.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <span class="flow-pill flow-pill-purple">Stage 2</span>
            <h4>Continuous Collision Risk Regression</h4>
            <p>Predicts expected casualty count (<code>number_of_casualties</code>) using purely environmental / road features:</p>
            <ul>
                <li>Speed limit, road type, junction detail, light conditions, weather.</li>
                <li>Collapsed to collision-level to eliminate row duplication bias.</li>
            </ul>
            <p><b>Model Tournament:</b></p>
            <ul>
                <li><b>Linear Regression:</b> $R^2 = 0.0232$, $\text{RMSE} = 0.8365$ (Winner)</li>
                <li><b>Decision Tree Regressor:</b> $R^2 = -0.0283$, $\text{RMSE} = 0.8582$</li>
            </ul>
            <p>The resulting continuous prediction is mapped onto every casualty row as the engineered feature <code>risk_score</code>, providing downstream classifiers with crash-scale severity context.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <span class="flow-pill flow-pill-amber">Stage 4 & 5</span>
            <h4>Meta-Classifier & Headline Evaluation</h4>
            <p><b>Stage 4 (Meta-Learner):</b> A balanced multiclass Logistic Regression meta-model operating on exactly 10 inputs:</p>
            <ul>
                <li>$P_{\text{logreg}}(\text{Fatal}), P_{\text{logreg}}(\text{Serious}), P_{\text{logreg}}(\text{Slight})$</li>
                <li>$P_{\text{knn}}(\text{Fatal}), P_{\text{knn}}(\text{Serious}), P_{\text{knn}}(\text{Slight})$</li>
                <li>$P_{\text{dt}}(\text{Fatal}), P_{\text{dt}}(\text{Serious}), P_{\text{dt}}(\text{Slight})$</li>
                <li>$+\\text{risk\\_score}$ (Stage 2 regression feature)</li>
            </ul>
            <p><b>Stage 5 (Headline Result):</b> Standalone models fail on the critical minority class (Fatal recall: 1.1% for KNN). The multi-stage stacking pipeline achieves <b>62.2% Fatal safety recall</b>, catching 179 out of 288 fatal crashes in held-out testing!</p>
        </div>
        """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECTION 2: Model Theory & Loss Functions
# ---------------------------------------------------------------------------
elif menu == "🔬 Model Theory & Loss Functions":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Mathematical Theory & Loss Functions</div>
        <div class="hero-subtitle">
            Rigorous breakdown of the objective functions, optimization criteria, and algorithmic mechanisms powering each stage of the stacked pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Stage 2: Regression Loss",
        "Stage 3A: Base Classifiers Loss",
        "Stage 4: Meta-Classifier Loss",
        "⚖️ Loss Function Comparison Table"
    ])

    with tab1:
        st.markdown("### Stage 2: Casualty Count Regression")
        st.markdown(r"""
        The goal is to estimate expected casualty intensity $y \in \mathbb{R}^+$ given collision-level road/environmental descriptors $\mathbf{x}$.

        #### 1. Linear Regression (Winner)
        Minimizes the Ordinary Least Squares (OLS) **Mean Squared Error (MSE)** loss:
        """)
        st.latex(r"\mathcal{L}_{\text{MSE}}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \left( y_i - \mathbf{w}^T \mathbf{x}_i \right)^2")
        st.markdown(r"""
        - **Closed-Form Solution:** $\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$
        - **Why it won:** Casualty counts are heavily zero-skewed and noisy ($1.2$ average). Linear regression maintains global convex stability without overfitting spurious interaction splits.
        - **Performance:** $R^2 = +0.0232$, $\text{RMSE} = 0.8365$.

        #### 2. Decision Tree Regressor (Challenger)
        Recursively partitions feature space using variance reduction:
        """)
        st.latex(r"\Delta \sigma^2(S, j, s) = \text{Var}(S) - \frac{|S_L|}{|S|} \text{Var}(S_L) - \frac{|S_R|}{|S|} \text{Var}(S_R)")
        st.markdown(r"""
        - **Why it degraded ($R^2 = -0.0283$):** Axis-aligned cuts created leaf overconfidence on localized noisy outliers, worsening test generalization compared to the global planar estimate.
        """)

    with tab2:
        st.markdown("### Stage 3A: Base Classifiers Optimization")
        
        st.markdown("#### 1. Multiclass Logistic Regression (Balanced Cross-Entropy)")
        st.markdown(r"""
        Minimizes L2-regularized multinomial cross-entropy (log-loss) with inverse class frequency weights $w_c$:
        """)
        st.latex(r"\mathcal{L}_{\text{logreg}}(\mathbf{W}) = -\sum_{i=1}^N \sum_{c=1}^3 w_c \cdot y_{ic} \ln \left( \frac{e^{\mathbf{w}_c^T \mathbf{x}_i}}{\sum_{j=1}^3 e^{\mathbf{w}_j^T \mathbf{x}_i}} \right) + \frac{1}{2C} \sum_{c=1}^3 \|\mathbf{w}_c\|_2^2")
        st.markdown(r"""
        Where class weights counteract the severe $76\% : 22.7\% : 1.3\%$ class imbalance:
        """)
        st.latex(r"w_c = \frac{N}{3 \cdot N_c}")

        st.markdown("---")
        st.markdown("#### 2. K-Nearest Neighbors ($k=15$, Distance-Weighted Voting)")
        st.markdown(r"""
        Non-parametric local metric learning over standardized space $z_j = \frac{x_j - \mu_j}{\sigma_j}$. Distance metric is Minkowski $p=2$ (Euclidean):
        """)
        st.latex(r"d(\mathbf{u}, \mathbf{v}) = \left( \sum_{j=1}^D |u_j - v_j|^2 \right)^{1/2}")
        st.markdown(r"""
        Class probability estimation uses inverse-distance kernel weighting:
        """)
        st.latex(r"w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i) + \epsilon}, \quad P(y=c|\mathbf{x}) = \frac{\sum_{i \in \mathcal{N}_k(\mathbf{x}), y_i=c} w_i}{\sum_{i \in \mathcal{N}_k(\mathbf{x})} w_i}")

        st.markdown("---")
        st.markdown("#### 3. Decision Tree Classifier (Gini Impurity)")
        st.markdown(r"""
        Splits feature space recursively to maximize Gini Impurity Gain under class weighting:
        """)
        st.latex(r"I_G(t) = 1 - \sum_{c=1}^3 \left( \frac{\sum_{i \in t, y_i=c} w_c}{\sum_{i \in t} w_{y_i}} \right)^2")

    with tab3:
        st.markdown("### Stage 4: Stacking Meta-Classifier Optimization")
        st.markdown(r"""
        The meta-classifier acts as an optimal Bayesian probability calibrator. Its input space is the 10-dimensional vector:
        """)
        st.latex(r"\mathbf{z} = \left[ P_{\text{logreg}}^{(1)}, P_{\text{logreg}}^{(2)}, P_{\text{logreg}}^{(3)}, P_{\text{knn}}^{(1)}, P_{\text{knn}}^{(2)}, P_{\text{knn}}^{(3)}, P_{\text{dt}}^{(1)}, P_{\text{dt}}^{(2)}, P_{\text{dt}}^{(3)}, \text{risk\_score} \right]^T")
        st.markdown(r"""
        The meta-learner solves the optimization problem:
        """)
        st.latex(r"\min_{\mathbf{\Theta}} \left\{ -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^3 w_c \cdot y_{ic} \ln \sigma(\mathbf{\theta}_c^T \mathbf{z}_i) + \lambda \|\mathbf{\Theta}\|_F^2 \right\}")
        st.markdown(r"""
        #### Why Stacking Outperforms Simple Voting:
        1. **Learned Reliability:** KNN is notoriously overconfident on majority classes, while Logistic Regression provides smooth probabilities. The meta-learner learns algorithm trust per class.
        2. **Condition Conditioning:** Adding `risk_score` gives the meta-classifier independent crash severity magnitude context that modulates the probability decision boundary.
        3. **Calibrated Thresholds:** Balanced class weighting in the meta-layer forces the final decision boundary to prioritize fatal recall without collapsing slight-class precision.
        """)

    with tab4:
        st.markdown("### 📋 Objective & Optimization Comparison")
        st.markdown("""
        | Stage | Model | Primary Objective / Loss Function | Regularization / Constraint | Role in Ensemble |
        | :--- | :--- | :--- | :--- | :--- |
        | **Stage 2** | Linear Regression | Mean Squared Error (MSE) / OLS | Preprocessor One-Hot Encoding | Continuous risk magnitude generator |
        | **Stage 2** | Decision Tree Regressor | Variance Reduction Criterion | Max Depth = 8 | Benchmark comparison |
        | **Stage 3A** | Logistic Regression | Weighted Multinomial Log-Loss | L2 Penalty ($C=1.0$), Balanced Weights | Linear probabilistic calibration |
        | **Stage 3A** | KNN Classifier | Inverse Distance Weighted Voting | $k=15$, KD-Tree indexed | Local geometric density estimation |
        | **Stage 3A** | Decision Tree | Gini Impurity Gain | Max Depth = 8, Min Leaf = 20 | Non-linear interaction feature modeling |
        | **Stage 4** | Meta-Classifier | Balanced Multinomial Cross-Entropy | Standardized Features + L2 Penalty | Optimal ensemble decision arbitration |
        """)


# ---------------------------------------------------------------------------
# SECTION 3: Model Comparisons & Metrics
# ---------------------------------------------------------------------------
elif menu == "📊 Model Comparisons & Metrics":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Model Performance & Benchmark Analysis</div>
        <div class="hero-subtitle">
            Evaluating standalone base models against the multi-stage stacked ensemble pipeline on the held-out test set (23,528 unseen casualties).
        </div>
    </div>
    """, unsafe_allow_html=True)

    comp_file = ROOT / "models" / "evaluation" / "results" / "model_comparison_table.csv"
    cm_file = ROOT / "models" / "evaluation" / "results" / "meta_classifier_confusion_matrix.csv"

    if comp_file.exists():
        df_comp = pd.read_csv(comp_file)
        # Format numeric columns for clean display
        for col in ["Macro_F1", "Accuracy", "Fatal_F1", "Serious_F1", "Slight_F1"]:
            if col in df_comp.columns:
                df_comp[col] = pd.to_numeric(df_comp[col], errors="coerce")

        # Top KPIs
        k1, k2, k3, k4 = st.columns(4)
        pipeline_row = df_comp[df_comp["Model"].str.contains("Stacking|Pipeline")].iloc[0]
        knn_row = df_comp[df_comp["Model"].str.contains("KNN")].iloc[0]

        k1.metric("Pipeline Macro-F1", f"{pipeline_row['Macro_F1']:.4f}", "+0.032 vs LogReg")
        k2.metric("Pipeline Fatal-F1", f"{pipeline_row['Fatal_F1']:.4f}", f"{(pipeline_row['Fatal_F1']/knn_row['Fatal_F1']):.1f}x higher than KNN")
        k3.metric("Fatal Safety Recall", "62.2%", "179 / 288 Fatalities Caught")
        k4.metric("Test Evaluation Samples", "23,528", "Group-Held Out")

        st.markdown("---")
        st.markdown("### 🏆 Comprehensive Benchmark Table")
        
        display_df = df_comp.copy()
        display_df["Accuracy"] = display_df["Accuracy"].apply(lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%")
        for c in ["Macro_F1", "Fatal_F1", "Serious_F1", "Slight_F1"]:
            display_df[c] = display_df[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 📈 Visual Metric Comparison")

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            fig, ax = plt.subplots(figsize=(8, 5))
            models_list = df_comp["Model"].tolist()
            x = np.arange(len(models_list))
            width = 0.35

            rects1 = ax.bar(x - width/2, df_comp["Macro_F1"], width, label='Macro-F1', color='#0ea5e9')
            rects2 = ax.bar(x + width/2, df_comp["Fatal_F1"] * 5, width, label='Fatal-F1 (scaled 5x)', color='#ef4444')

            ax.set_ylabel('Score')
            ax.set_title('Macro-F1 vs Fatal-F1 (Safety Critical)')
            ax.set_xticks(x)
            ax.set_xticklabels([m.split("(")[0].strip() for m in models_list], rotation=20, ha='right')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)

        with col_plot2:
            if cm_file.exists():
                cm_df = pd.read_csv(cm_file, index_col=0)
                fig_cm, ax_cm = plt.subplots(figsize=(7, 5.2))
                sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                            annot_kws={"size": 13, "weight": "bold"})
                ax_cm.set_title('Meta-Classifier Confusion Matrix (Test Set)', fontsize=13, weight='bold')
                ax_cm.set_xlabel('Predicted Label', fontsize=11)
                ax_cm.set_ylabel('True Label', fontsize=11)
                fig_cm.tight_layout()
                st.pyplot(fig_cm)

        st.markdown("""
        <div class="glass-card">
            <h4>💡 Why the Stacking Pipeline is Better: The Safety-Critical Tradeoff</h4>
            <p>
            In standard machine learning benchmarks, models like <b>KNN</b> appear to have superior accuracy (<b>73.67%</b>). However, this is an <b>accuracy illusion</b> caused by severe class imbalance:
            </p>
            <ul>
                <li>Because 76% of casualties are "Slight", a model predicting only "Slight" easily achieves ~76% accuracy while catching <b>zero</b> fatal casualties.</li>
                <li>KNN standalone only caught <b>1.1%</b> of fatal accidents (Fatal-F1: 0.0116), rendering it completely dangerous for public health and road safety deployment.</li>
                <li>The <b>Full Multi-Stage Stacking Pipeline</b> achieves a balanced <b>62.2% Fatal Recall</b> (catching 179 out of 288 fatalities), ensuring emergency response and traffic safety auditors are alerted to life-threatening risks.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Evaluation results file not found. Run `python pipeline.py` to generate benchmarks.")


# ---------------------------------------------------------------------------
# SECTION 4: Live Single Incident Custom Prediction
# ---------------------------------------------------------------------------
elif menu == "🎯 Single Incident Prediction":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Live Collision Severity Prediction</div>
        <div class="hero-subtitle">
            Configure specific environmental and crash parameters below. The incident will be processed sequentially through Stage 2 Regression, Stage 3A Base Classifiers, and Stage 4 Meta-Classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset scenarios for quick testing
    st.markdown("#### ⚡ Quick Presets")
    sc_col1, sc_col2, sc_col3 = st.columns(3)
    preset_choice = None
    if sc_col1.button("💥 High-Speed Rural Night Crash (Severe)"):
        preset_choice = "high_risk"
    if sc_col2.button("🚗 Urban Daylight Low-Speed Bump (Mild)"):
        preset_choice = "low_risk"
    if sc_col3.button("🌧️ Rainy Motorway Multi-Car Incident (Medium)"):
        preset_choice = "med_risk"

    # Default values or preset overrides
    defaults = {
        "speed_limit": 60 if preset_choice == "high_risk" else (30 if preset_choice == "low_risk" else 70),
        "road_type": 6 if preset_choice == "high_risk" else (6 if preset_choice == "low_risk" else 3),
        "junction_detail": 0 if preset_choice == "high_risk" else 13,
        "light_conditions": 6 if preset_choice == "high_risk" else (1 if preset_choice == "low_risk" else 4),
        "weather_conditions": 1 if preset_choice == "low_risk" else 2,
        "road_surface_conditions": 1 if preset_choice == "low_risk" else 2,
        "casualty_class": 3 if preset_choice == "high_risk" else 1,
        "casualty_type": 0 if preset_choice == "high_risk" else 9,
        "number_of_vehicles": 1 if preset_choice == "high_risk" else (2 if preset_choice == "low_risk" else 3),
        "number_of_casualties": 1 if preset_choice == "low_risk" else 2,
        "day_of_week": 6,
        "first_road_class": 3 if preset_choice != "med_risk" else 1,
        "junction_control": 4,
        "special_conditions_at_site": 0,
        "carriageway_hazards": 0,
        "urban_or_rural_area": 2 if preset_choice == "high_risk" else 1
    }

    with st.form("single_prediction_form"):
        st.markdown("### 1. Environmental & Road Conditions")
        c1, c2, c3 = st.columns(3)

        with c1:
            speed_limit = st.selectbox(
                "Speed Limit (mph):",
                SPEED_LIMIT_OPTIONS,
                index=SPEED_LIMIT_OPTIONS.index(defaults["speed_limit"])
            )
            road_type = st.selectbox(
                "Road Type:",
                list(ROAD_TYPE_MAP.keys()),
                index=list(ROAD_TYPE_MAP.values()).index(defaults["road_type"])
            )
            first_road_class = st.selectbox(
                "Road Classification:",
                list(ROAD_CLASS_MAP.keys()),
                index=list(ROAD_CLASS_MAP.values()).index(defaults["first_road_class"])
            )

        with c2:
            light_conditions = st.selectbox(
                "Lighting Conditions:",
                list(LIGHT_CONDITIONS_MAP.keys()),
                index=list(LIGHT_CONDITIONS_MAP.values()).index(defaults["light_conditions"])
            )
            weather_conditions = st.selectbox(
                "Weather Conditions:",
                list(WEATHER_CONDITIONS_MAP.keys()),
                index=list(WEATHER_CONDITIONS_MAP.values()).index(defaults["weather_conditions"])
            )
            road_surface = st.selectbox(
                "Road Surface:",
                list(ROAD_SURFACE_CONDITIONS_MAP.keys()),
                index=list(ROAD_SURFACE_CONDITIONS_MAP.values()).index(defaults["road_surface_conditions"])
            )

        with c3:
            junction_detail = st.selectbox(
                "Junction Detail:",
                list(JUNCTION_DETAIL_MAP.keys()),
                index=list(JUNCTION_DETAIL_MAP.values()).index(defaults["junction_detail"])
            )
            junction_control = st.selectbox(
                "Junction Control:",
                list(JUNCTION_CONTROL_MAP.keys()),
                index=list(JUNCTION_CONTROL_MAP.values()).index(defaults["junction_control"])
            )
            urban_rural = st.selectbox(
                "Area Type:",
                list(URBAN_RURAL_MAP.keys()),
                index=list(URBAN_RURAL_MAP.values()).index(defaults["urban_or_rural_area"])
            )

        st.markdown("### 2. Casualty & Incident Context")
        c4, c5, c6 = st.columns(3)

        with c4:
            cas_class = st.selectbox(
                "Casualty Role:",
                list(CASUALTY_CLASS_MAP.keys()),
                index=list(CASUALTY_CLASS_MAP.values()).index(defaults["casualty_class"])
            )
            cas_type = st.selectbox(
                "Vehicle / Casualty Type:",
                list(CASUALTY_TYPE_MAP.keys()),
                index=list(CASUALTY_TYPE_MAP.values()).index(defaults["casualty_type"])
            )

        with c5:
            num_vehicles = st.number_input("Number of Vehicles Involved:", min_value=1, max_value=20, value=defaults["number_of_vehicles"])
            num_casualties = st.number_input("Total Casualties in Incident:", min_value=1, max_value=50, value=defaults["number_of_casualties"])

        with c6:
            day_of_week = st.selectbox(
                "Day of Week:",
                list(DAY_OF_WEEK_MAP.keys()),
                index=list(DAY_OF_WEEK_MAP.values()).index(defaults["day_of_week"])
            )
            special_conditions = st.selectbox(
                "Special Site Hazards:",
                list(SPECIAL_CONDITIONS_MAP.keys()),
                index=list(SPECIAL_CONDITIONS_MAP.values()).index(defaults["special_conditions_at_site"])
            )
            carriageway_hazards = st.selectbox(
                "Carriageway Obstacles:",
                list(CARRIAGEWAY_HAZARDS_MAP.keys()),
                index=list(CARRIAGEWAY_HAZARDS_MAP.values()).index(defaults["carriageway_hazards"])
            )

        submitted = st.form_submit_button("🚀 Run Full Pipeline Inference", use_container_width=True)

    if submitted:
        input_record = {
            "casualty_class": CASUALTY_CLASS_MAP[cas_class],
            "casualty_type": CASUALTY_TYPE_MAP[cas_type],
            "number_of_vehicles": int(num_vehicles),
            "number_of_casualties": int(num_casualties),
            "day_of_week": DAY_OF_WEEK_MAP[day_of_week],
            "first_road_class": ROAD_CLASS_MAP[first_road_class],
            "road_type": ROAD_TYPE_MAP[road_type],
            "speed_limit": int(speed_limit),
            "junction_detail": JUNCTION_DETAIL_MAP[junction_detail],
            "junction_control": JUNCTION_CONTROL_MAP[junction_control],
            "light_conditions": LIGHT_CONDITIONS_MAP[light_conditions],
            "weather_conditions": WEATHER_CONDITIONS_MAP[weather_conditions],
            "road_surface_conditions": ROAD_SURFACE_CONDITIONS_MAP[road_surface],
            "special_conditions_at_site": SPECIAL_CONDITIONS_MAP[special_conditions],
            "carriageway_hazards": CARRIAGEWAY_HAZARDS_MAP[carriageway_hazards],
            "urban_or_rural_area": URBAN_RURAL_MAP[urban_rural]
        }

        with st.spinner("Processing through 4-stage pipeline..."):
            result = predict_single(input_record)

        pred_class = result["prediction"]
        pred_label = SEVERITY_LABELS[pred_class]
        probs = result["probabilities"]
        risk_score = result["risk_score"]

        st.markdown("---")
        st.markdown("### 🏆 Final Pipeline Prediction")

        r_col1, r_col2 = st.columns([1.2, 1.8])

        with r_col1:
            badge_class = f"severity-badge-{pred_label.lower()}"
            st.markdown(f"""
            <div class="{badge_class}">
                {SEVERITY_ICONS[pred_class]} {pred_label.upper()} SEVERITY
                <div style="font-size:0.95rem; font-weight:400; margin-top:0.4rem;">
                    Confidence: {probs[pred_class-1]*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Stage 2 Predicted Risk Score</div>
                <div class="metric-val">{risk_score:.2f} casualties</div>
            </div>
            """, unsafe_allow_html=True)

        with r_col2:
            st.markdown("#### Class Probability Distribution (Meta-Learner)")
            prob_df = pd.DataFrame({
                "Severity Class": ["1: Fatal", "2: Serious", "3: Slight"],
                "Probability": [probs[0], probs[1], probs[2]]
            })
            st.bar_chart(prob_df.set_index("Severity Class"), color="#38bdf8")

        # Step-by-Step Transparency Expander
        with st.expander("🔍 Inspect Internal Pipeline Reasoning (Stages 2 -> 3A -> 4)"):
            st.markdown("##### 1. Base Classifiers Probability Outputs (Stage 3A)")
            base_rows = []
            for name, p in result["base_probabilities"].items():
                base_rows.append({
                    "Model": name,
                    "P(Fatal)": f"{p[0]*100:.2f}%",
                    "P(Serious)": f"{p[1]*100:.2f}%",
                    "P(Slight)": f"{p[2]*100:.2f}%",
                    "Model Vote": SEVERITY_LABELS[np.argmax(p) + 1]
                })
            st.table(pd.DataFrame(base_rows))

            st.markdown("##### 2. Exact 10-Feature Vector Fed to Meta-Classifier (Stage 4)")
            st.dataframe(pd.DataFrame([result["meta_features"]]))


# ---------------------------------------------------------------------------
# SECTION 5: Batch CSV / Excel Testing
# ---------------------------------------------------------------------------
elif menu == "📁 Batch CSV / Excel Testing":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Batch Prediction: CSV / Excel Ingestion</div>
        <div class="hero-subtitle">
            Upload custom road accident records in CSV or Excel (.xlsx, .xls) format to execute end-to-end vectorized inference across thousands of incidents simultaneously.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state for batch processing
    if "batch_df" not in st.session_state:
        st.session_state["batch_df"] = None
    if "enriched_df" not in st.session_state:
        st.session_state["enriched_df"] = None

    # Sample batch generator button
    st.markdown("#### ⚡ Don't have a file ready?")
    b_col1, b_col2 = st.columns([1.5, 2.5])
    
    with b_col1:
        if st.button("📥 Load Sample Test Batch (10 Unseen Crashes)"):
            sample_path = ROOT / "dataset" / "processed" / "merged_test.csv"
            if sample_path.exists():
                st.session_state["batch_df"] = pd.read_csv(sample_path, nrows=10)
                st.session_state["enriched_df"] = None
                st.success("Loaded 10 rows from held-out test set!")
            else:
                st.error("Test dataset not found. Please upload a file.")

    uploaded_file = st.file_uploader(
        "Upload Collision Data File (.csv, .xlsx, .xls):",
        type=["csv", "xlsx", "xls"],
        help="Must contain the 16 standard STATS19 feature columns."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state["batch_df"] = pd.read_csv(uploaded_file)
            else:
                st.session_state["batch_df"] = pd.read_excel(uploaded_file)
            st.session_state["enriched_df"] = None
            st.success(f"Successfully loaded '{uploaded_file.name}' with {len(st.session_state['batch_df']):,} rows.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    df_batch = st.session_state.get("batch_df")

    if df_batch is not None:
        st.markdown("### 📋 Uploaded Data Preview")
        st.dataframe(df_batch.head(5), use_container_width=True)

        if st.button("🚀 Process Batch Through Pipeline", use_container_width=True):
            try:
                with st.spinner(f"Running inference on {len(df_batch):,} records..."):
                    res = predict_pipeline(df_batch)

                preds = res["final_predictions"]
                probs = res["final_probabilities"]
                risks = res["risk_scores"]

                # Enrich dataframe
                enriched = df_batch.copy()
                enriched["Predicted_Severity_Code"] = preds
                enriched["Predicted_Severity"] = [SEVERITY_LABELS[p] for p in preds]
                enriched["P_Fatal"] = np.round(probs[:, 0], 4)
                enriched["P_Serious"] = np.round(probs[:, 1], 4)
                enriched["P_Slight"] = np.round(probs[:, 2], 4)
                enriched["Predicted_Risk_Score"] = np.round(risks, 3)
                st.session_state["enriched_df"] = enriched

            except Exception as e:
                st.error(f"Inference error: {e}")
                st.info("Make sure the uploaded file includes all 16 required STATS19 columns.")

    enriched_df = st.session_state.get("enriched_df")
    if enriched_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Batch Prediction Analytics")

        # Summary KPIs
        preds = enriched_df["Predicted_Severity_Code"].values
        fatal_cnt = int(sum(preds == 1))
        serious_cnt = int(sum(preds == 2))
        slight_cnt = int(sum(preds == 3))
        total_cnt = len(preds)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Processed", f"{total_cnt:,}")
        k2.metric("Predicted Fatal (🚨)", f"{fatal_cnt} ({fatal_cnt/total_cnt*100:.1f}%)")
        k3.metric("Predicted Serious (⚠️)", f"{serious_cnt} ({serious_cnt/total_cnt*100:.1f}%)")
        k4.metric("Predicted Slight (🟢)", f"{slight_cnt} ({slight_cnt/total_cnt*100:.1f}%)")

        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Severity Class Breakdown")
            chart_df = pd.DataFrame({
                "Severity": ["Fatal", "Serious", "Slight"],
                "Count": [fatal_cnt, serious_cnt, slight_cnt]
            }).set_index("Severity")
            st.bar_chart(chart_df, color="#f59e0b")

        with c2:
            st.markdown("#### Average Risk Score per Severity")
            avg_risk = enriched_df.groupby("Predicted_Severity")["Predicted_Risk_Score"].mean()
            st.bar_chart(avg_risk, color="#0ea5e9")

        st.markdown("### 📋 Enriched Predictions Table")
        st.dataframe(enriched_df, use_container_width=True)

        # Download buttons (CSV and Excel)
        d1, d2 = st.columns(2)
        with d1:
            csv_data = enriched_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name="collision_severity_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )
        with d2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                enriched_df.to_excel(writer, index=False, sheet_name='Predictions')
            buffer.seek(0)
            st.download_button(
                label="📥 Download Results as Excel (.xlsx)",
                data=buffer,
                file_name="collision_severity_predictions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )


# ---------------------------------------------------------------------------
# SECTION 6: Pipeline Execution Audit Logs
# ---------------------------------------------------------------------------
elif menu == "📋 Pipeline Audit Logs":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Pipeline Execution History & Logs</div>
        <div class="hero-subtitle">
            Direct inspection of <code>pipeline_log.json</code>. Track exact timestamps, execution time per step, input/output file hashes, and console outputs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    log_path = ROOT / "pipeline_log.json"
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pipeline Status", log_data.get("final_status", "UNKNOWN"))
        m2.metric("Total Execution Time", f"{log_data.get('total_duration_seconds', 0):.1f}s")
        m3.metric("Total Steps Logged", len(log_data.get("steps", [])))
        m4.metric("Last Run Timestamp", log_data.get("completed_at", "N/A")[:19].replace("T", " "))

        st.markdown("---")
        st.markdown("### 📜 Stage-by-Stage Detailed Audit")

        for s in log_data.get("steps", []):
            status_color = "green" if s.get("status") == "SUCCESS" else "red"
            with st.expander(f"Step {s['step']}: {s['name']} — [{s['status']}] in {s.get('duration_seconds', 0):.1f}s"):
                st.markdown(f"**Script:** `{s['script']}`")
                st.markdown(f"**Description:** {s['description']}")
                st.markdown(f"**Why:** {s['why']}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Inputs:**")
                    for inp in s.get("inputs", []):
                        st.markdown(f"- `{inp['path']}` ({inp.get('size_human', 'N/A')}, {inp.get('rows', 'N/A')} rows)")
                with c2:
                    st.markdown("**Outputs:**")
                    for outp in s.get("outputs", []):
                        st.markdown(f"- `{outp['path']}` ({outp.get('size_human', 'N/A')}, {outp.get('rows', 'N/A')} rows)")

                if "metrics" in s:
                    st.markdown("**Metrics:**")
                    st.json(s["metrics"])

                if s.get("console_output"):
                    st.markdown("**Console Output:**")
                    st.code(s["console_output"], language="text")
    else:
        st.warning("No pipeline_log.json found. Run `python pipeline.py` first.")
