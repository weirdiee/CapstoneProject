import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import FEATURE_COLS, RAW_FEATURES, add_derived_features, get_loan_features

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Default Risk Modeling",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENCY = "$"  # PKDD'99 is Czech data (CZK); change this label if you prefer
RANDOM_STATE = 42

# ==========================================
# 1. DATA (heavy lifting lives in features.py; run `python features.py` once
#    to build the small loan_features.csv so the 109 MB transaction file is not needed online)
# ==========================================
# "B_or_D": default = status B or D (76 loans, includes running-but-unpaid D loans)
# "A_vs_B": only finished loans (A = paid, B = defaulted). Cleaner label, far fewer rows.
LABEL_MODE = "B_or_D"


@st.cache_data
def load_and_engineer_features():
    df = get_loan_features(BASE_DIR)
    if LABEL_MODE == "A_vs_B":
        df = df[df["status"].isin(["A", "B"])].copy()
        df["is_default"] = (df["status"] == "B").astype(int)
    return add_derived_features(df).reset_index(drop=True)


# ==========================================
# 2. MODEL TRAINING (tuned, cross-validated, calibrated)
# ==========================================
def signed_log1p(x):
    """Tames heavy-tailed money features for Logistic Regression
    (sign-preserving, so negative balances still work)."""
    return np.sign(x) * np.log1p(np.abs(x))


def make_lr_search(cv_folds):
    pipe = Pipeline([
        ("log", FunctionTransformer(signed_log1p, validate=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ])
    grid = {
        "clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "clf__class_weight": [None, "balanced"],
    }
    return GridSearchCV(
        pipe, grid, scoring="roc_auc",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )


def make_rf_search(cv_folds):
    rf = RandomForestClassifier(
        n_estimators=150, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=1,
    )
    grid = {
        "max_depth": [3, 6, None],
        "min_samples_leaf": [3, 8],
        "max_features": ["sqrt"],
    }
    return GridSearchCV(
        rf, grid, scoring="roc_auc",
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )


def nested_oof(make_search, X, y, outer_folds=5, inner_folds=3):
    """Honest evaluation: tune on the training part of each outer fold,
    then score the untouched fold. Returns out-of-fold probabilities."""
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr_idx, te_idx in outer.split(X, y):
        search = make_search(inner_folds)
        search.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = search.predict_proba(X.iloc[te_idx])[:, 1]
        oof[te_idx] = p
        fold_aucs.append(roc_auc_score(y.iloc[te_idx], p))
    return oof, fold_aucs


@st.cache_resource
def train_predictive_models(df_features):
    X = df_features[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_features["is_default"]

    # --- 1. Honest performance estimate (nested 5-fold CV, all 100% of the data used) ---
    oof_lr, folds_lr = nested_oof(make_lr_search, X, y)
    oof_rf, folds_rf = nested_oof(make_rf_search, X, y)
    oof_blend = (oof_lr + oof_rf) / 2

    # --- 2. Final models: tune on all data, then calibrate so PD is a real probability ---
    lr_search = make_lr_search(5).fit(X, y)
    rf_search = make_rf_search(5).fit(X, y)

    lr_cal = CalibratedClassifierCV(lr_search.best_estimator_, method="sigmoid", cv=5).fit(X, y)
    rf_cal = CalibratedClassifierCV(rf_search.best_estimator_, method="sigmoid", cv=5).fit(X, y)

    # --- 3. Temporal sanity check: train on the earliest 70% of loans, test on the latest 30% ---
    order = np.argsort(df_features["loan_date"].values, kind="stable")
    cut = int(len(order) * 0.7)
    tr_i, te_i = order[:cut], order[cut:]
    temporal = {k: np.nan for k in ["Logistic Regression", "Random Forest", "Blend (average)"]}
    if y.iloc[te_i].nunique() == 2 and y.iloc[tr_i].nunique() == 2:
        p = {}
        for name, est in [("Logistic Regression", lr_search.best_estimator_),
                          ("Random Forest", rf_search.best_estimator_)]:
            m = clone(est).fit(X.iloc[tr_i], y.iloc[tr_i])
            p[name] = m.predict_proba(X.iloc[te_i])[:, 1]
            temporal[name] = roc_auc_score(y.iloc[te_i], p[name])
        temporal["Blend (average)"] = roc_auc_score(
            y.iloc[te_i], (p["Logistic Regression"] + p["Random Forest"]) / 2)

    return {
        "temporal": temporal,
        "y": y,
        "oof": {"Logistic Regression": oof_lr, "Random Forest": oof_rf, "Blend (average)": oof_blend},
        "fold_aucs": {"Logistic Regression": folds_lr, "Random Forest": folds_rf},
        "lr_cal": lr_cal,
        "rf_cal": rf_cal,
        "lr_best": lr_search.best_estimator_,
        "rf_best": rf_search.best_estimator_,
        "lr_params": lr_search.best_params_,
        "rf_params": rf_search.best_params_,
    }


with st.spinner("Building features and tuning models (first load only, may take a minute)..."):
    df_features = load_and_engineer_features()
    bundle = train_predictive_models(df_features)

y_all = bundle["y"]
base_rate = float(y_all.mean())
oof = bundle["oof"]
auc = {name: roc_auc_score(y_all, p) for name, p in oof.items()}

# ==========================================
# SIDEBAR: scoring engine and risk tiers
# ==========================================
st.sidebar.header("⚙️ Scoring settings")
engines = ["Random Forest", "Logistic Regression", "Blend (average)"]
engine = st.sidebar.selectbox(
    "Primary scoring engine", engines, index=engines.index(max(engines, key=lambda n: auc[n])),
    help="Defaults to whichever scored the best out-of-fold ROC-AUC.",
)
st.sidebar.caption(f"Portfolio default rate in training data: **{base_rate:.1%}**")
st.sidebar.markdown("**Risk tier cut-offs (PD %)**")
cuts = sorted([
    st.sidebar.number_input("Low → Moderate", 1, 99, 10, 1) / 100,
    st.sidebar.number_input("Moderate → High", 1, 99, 20, 1) / 100,
    st.sidebar.number_input("High → Critical", 1, 99, 40, 1) / 100,
])
st.sidebar.caption(
    "PDs are calibrated, so they are comparable to the portfolio default rate. "
    "Fixed 25/50/75% cut-offs would almost never trigger when the base rate is ~10%."
)

# ==========================================
# 3. STREAMLIT APPLICATION INTERFACE
# ==========================================
st.title("🏦 PKDD Credit Default Risk & Predictive Underwriting")
tab1, tab2 = st.tabs(["📊 Model Performance & Benchmarking", "🎯 Interactive Loan Default Predictor"])

# -----------------------------------------------------------------------------
# TAB 1: MODEL EVALUATION
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Model Evaluation & Discrimination Power")
    st.caption(
        "Scores come from nested 5-fold cross-validation: hyperparameters are tuned inside each "
        "training fold and every loan is scored by a model that never saw it. This is far more "
        "reliable than a single 80/20 split on a small dataset."
    )

    m = st.columns(5)
    m[0].metric("Loans analysed", f"{len(df_features):,}")
    m[1].metric("Default rate", f"{base_rate:.1%}")
    m[2].metric("Logistic Reg. ROC-AUC", f"{auc['Logistic Regression']:.3f}")
    m[3].metric("Random Forest ROC-AUC", f"{auc['Random Forest']:.3f}")
    m[4].metric("Blend ROC-AUC", f"{auc['Blend (average)']:.3f}")

    summary = pd.DataFrame({
        "Pooled ROC-AUC": {k: auc[k] for k in oof},
        "Fold mean ROC-AUC": {
            **{k: np.mean(v) for k, v in bundle["fold_aucs"].items()},
            "Blend (average)": np.nan,
        },
        "Fold std": {
            **{k: np.std(v) for k, v in bundle["fold_aucs"].items()},
            "Blend (average)": np.nan,
        },
        "PR-AUC": {k: average_precision_score(y_all, p) for k, p in oof.items()},
        "Latest-30% loans AUC": bundle["temporal"],
    }).round(3)
    st.dataframe(summary, width='stretch')
    st.caption(
        f"PR-AUC matters more than ROC-AUC when defaults are rare; random guessing scores about "
        f"{base_rate:.2f}. A large fold std means the estimate is noisy. 'Latest-30% loans AUC' trains on "
        f"older loans and tests on newer ones; a big drop versus the cross-validated AUC suggests drift over time."
    )

    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Blend (average)": "#ff7f0e"}
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, p in oof.items():
            fpr, tpr, _ = roc_curve(y_all, p)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc[name]:.2f})", color=colors[name], lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curves (out-of-fold)")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, p in oof.items():
            prec, rec, _ = precision_recall_curve(y_all, p)
            ax.plot(rec, prec, label=name, color=colors[name], lw=2)
        ax.axhline(base_rate, color="k", ls="--", lw=1, label="Random guess")
        ax.set_title("Precision-Recall Curves (out-of-fold)")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)

    with c3:
        imp = pd.Series(bundle["rf_best"].feature_importances_, index=FEATURE_COLS).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        imp.plot(kind="barh", ax=ax, color="#d62728")
        ax.set_title("What the Random Forest relies on")
        ax.set_xlabel("Gini importance (size only, not direction)")
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        coefs = pd.Series(
            bundle["lr_best"].named_steps["clf"].coef_[0], index=FEATURE_COLS
        ).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(coefs.index, coefs.values, color=["#2ca02c" if v < 0 else "#d62728" for v in coefs.values])
        ax.axvline(0, color="k", lw=1)
        ax.set_title("Logistic Regression direction of effect")
        ax.set_xlabel("Standardised coefficient (red = raises default risk)")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("🛠️ Selected hyperparameters"):
        st.write("**Logistic Regression:**", bundle["lr_params"])
        st.write("**Random Forest:**", bundle["rf_params"])

# -----------------------------------------------------------------------------
# TAB 2: INTERACTIVE UNDERWRITING SIMULATOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Underwrite a Prospective Loan Application")
    st.markdown(
        "Adjust loan terms and applicant account history. The **Probability of Default (PD)** "
        "updates live; there is no button to press."
    )

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("**1. Loan Contract Terms**")
        input_amount = st.number_input(
            f"Total Loan Principal ({CURRENCY})", min_value=1000, max_value=600000, value=150000, step=5000
        )
        input_duration = st.selectbox("Term Duration (Months)", [12, 24, 36, 48, 60], index=2)
        annual_rate = st.number_input(
            "Annual Interest Rate (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.5,
            help="PKDD'99 installments are essentially principal ÷ term (0%). "
                 "Enter a rate to use a standard amortised instalment instead.",
        )

        # --- automatic instalment calculation ---
        monthly_rate = annual_rate / 100 / 12
        if monthly_rate == 0:
            auto_payment = input_amount / input_duration
        else:
            auto_payment = input_amount * monthly_rate / (1 - (1 + monthly_rate) ** -input_duration)

        override = st.checkbox("Override instalment manually")
        if override:
            # value changes with the inputs, so the widget refreshes to the new auto value
            input_payment = st.number_input(
                f"Monthly Instalment ({CURRENCY})", min_value=0.0, max_value=100000.0,
                value=float(round(auto_payment, 2)), step=100.0,
            )
        else:
            input_payment = float(auto_payment)
            st.metric(f"Monthly Instalment ({CURRENCY}), auto", f"{input_payment:,.2f}")

    with col_input2:
        st.markdown("**2. Account History & Tenure**")
        input_tenure = st.slider("Account Relationship Tenure (Months)", min_value=1.0, max_value=72.0, value=24.0, step=0.5)
        input_trans_count = st.number_input("Pre-Loan Completed Transactions", min_value=1, max_value=1000, value=120, step=5)
        input_avg_balance = st.number_input(f"Average Pre-Loan Account Balance ({CURRENCY})", min_value=0, max_value=200000, value=35000, step=1000)
        input_min_balance = st.number_input(f"Lowest Pre-Loan Balance Observed ({CURRENCY})", min_value=-50000, max_value=100000, value=2500, step=500)

    with col_input3:
        st.markdown("**3. Historical Cash Flow Velocity**")
        input_total_inflow = st.number_input(f"Total Historical Inflow / Credits ({CURRENCY})", min_value=1000, max_value=2000000, value=280000, step=10000)
        input_total_outflow = st.number_input(f"Total Historical Outflow / Debits ({CURRENCY})", min_value=1000, max_value=2000000, value=245000, step=10000)

    # Same feature function as training, so the inputs are transformed identically
    raw_input = pd.DataFrame([{
        "loan_amount": input_amount,
        "loan_duration": input_duration,
        "monthly_loan_payment": input_payment,
        "account_tenure_months": input_tenure,
        "pre_loan_trans_count": input_trans_count,
        "avg_balance_pre_loan": input_avg_balance,
        "min_balance_pre_loan": input_min_balance,
        "total_inflow_pre_loan": input_total_inflow,
        "total_outflow_pre_loan": input_total_outflow,
    }])
    full_input = add_derived_features(raw_input)
    input_df = full_input[FEATURE_COLS]

    # ---- sanity checks on the inputs ----
    warnings_list = []
    if input_min_balance > input_avg_balance:
        warnings_list.append("Lowest balance is higher than the average balance, which is impossible.")
    out_of_range = [
        c for c in RAW_FEATURES + ["payment_to_inflow_ratio", "loan_to_avg_balance", "inflow_to_outflow_ratio"]
        if c in df_features.columns
        and not (df_features[c].min() <= full_input.at[0, c] <= df_features[c].max())
    ]
    if out_of_range:
        warnings_list.append(
            "These values fall outside the range seen in training, so the prediction is an "
            "extrapolation: " + ", ".join(f"`{c}`" for c in out_of_range)
        )
    for w in warnings_list:
        st.warning(w)

    st.divider()

    # ---- scoring ----
    pd_log = bundle["lr_cal"].predict_proba(input_df)[0, 1]
    pd_rf = bundle["rf_cal"].predict_proba(input_df)[0, 1]
    pd_blend = (pd_log + pd_rf) / 2
    primary_pd = {"Random Forest": pd_rf, "Logistic Regression": pd_log, "Blend (average)": pd_blend}[engine]

    low_c, mod_c, high_c = cuts
    if primary_pd < low_c:
        risk_tier, decision_box = "Low Risk", st.success
        action = "✅ **APPROVE**: Low risk of default relative to the portfolio."
    elif primary_pd < mod_c:
        risk_tier, decision_box = "Moderate Risk", st.info
        action = "⚠️ **MANUAL REVIEW**: Moderate risk. Consider requesting collateral or credit card linkage."
    elif primary_pd < high_c:
        risk_tier, decision_box = "High Risk", st.warning
        action = "🛑 **BORDERLINE**: Elevated risk. Recommend reducing loan principal or extending term."
    else:
        risk_tier, decision_box = "Critical Risk", st.error
        action = "❌ **REJECT**: High probability of payment delinquency."

    r = st.columns(4)
    r[0].metric(f"Primary PD ({engine})", f"{primary_pd:.1%}")
    r[1].metric("Random Forest PD", f"{pd_rf:.1%}")
    r[2].metric("Logistic Regression PD", f"{pd_log:.1%}")
    r[3].metric("Assigned Risk Category", risk_tier)

    if abs(pd_rf - pd_log) > 0.15:
        st.caption("⚠️ The two models disagree by more than 15 percentage points. Treat this case with extra caution.")
    st.caption(f"Portfolio average default rate: {base_rate:.1%}. A PD near this level is an average-risk loan.")

    decision_box(action)

    with st.expander("🔍 View Computed Financial Underwriting Indicators"):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Est. Monthly Inflow", f"{CURRENCY}{full_input.at[0, 'avg_monthly_inflow']:,.2f}")
        e2.metric("Payment-to-Inflow Ratio", f"{full_input.at[0, 'payment_to_inflow_ratio']:.2%}")
        e3.metric("Inflow / Outflow Ratio", f"{full_input.at[0, 'inflow_to_outflow_ratio']:.2f}x")
        e4.metric("Balance Cover (months of instalments)", f"{full_input.at[0, 'balance_cover_months']:.1f}")
