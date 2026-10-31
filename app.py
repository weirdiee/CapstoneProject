import os
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Default Risk Modeling",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CURRENCY = "$"
RANDOM_STATE = 42

# Define exact feature definitions
RAW_FEATURES = [
    "loan_amount",
    "loan_duration",
    "monthly_loan_payment",
    "account_tenure_months",
    "pre_loan_trans_count",
    "avg_balance_pre_loan",
    "min_balance_pre_loan",
    "total_inflow_pre_loan",
    "total_outflow_pre_loan",
]

FEATURE_COLS = [
    "loan_amount",
    "loan_duration",
    "monthly_loan_payment",
    "account_tenure_months",
    "pre_loan_trans_count",
    "avg_balance_pre_loan",
    "min_balance_pre_loan",
    "payment_to_inflow_ratio",
    "loan_to_avg_balance",
    "inflow_to_outflow_ratio",
    "balance_cover_months",
]


def add_derived_features(df):
    """Calculates all domain-specific financial underwriting ratios."""
    df = df.copy()
    pre_months = np.maximum(df["account_tenure_months"].fillna(1.0), 1.0)
    inflow = df["total_inflow_pre_loan"].fillna(0)
    outflow = df["total_outflow_pre_loan"].fillna(0)
    avg_bal = df["avg_balance_pre_loan"].fillna(0)
    payment = df["monthly_loan_payment"].fillna(0)
    principal = df["loan_amount"].fillna(0)

    df["avg_monthly_inflow"] = inflow / pre_months
    df["net_cash_flow_pre_loan"] = inflow - outflow
    df["payment_to_inflow_ratio"] = payment / (df["avg_monthly_inflow"] + 1.0)
    df["loan_to_avg_balance"] = principal / (np.maximum(avg_bal, 0) + 1.0)
    df["inflow_to_outflow_ratio"] = inflow / (outflow + 1.0)
    df["balance_cover_months"] = np.maximum(avg_bal, 0) / (payment + 1.0)
    return df


@st.cache_data
def load_and_engineer_features():
    """Attempts to load precomputed features first. Falls back to raw CSV processing."""
    precomputed_candidates = [
        os.path.join(BASE_DIR, "loan_features.csv"),
        os.path.join(BASE_DIR, "Loan_Default_Risk_Scored.csv"),
    ]

    for path in precomputed_candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if "status" in df.columns and "is_default" not in df.columns:
                df["is_default"] = df["status"].isin(["B", "D"]).astype(int)
            return add_derived_features(df).reset_index(drop=True)

    # Fallback: Extract directly from core tables via in-memory SQLite
    loan_path = os.path.join(BASE_DIR, "Loan.csv")
    acc_path = os.path.join(BASE_DIR, "Account.csv")
    trans_path = os.path.join(BASE_DIR, "Transaction.csv")

    if not (os.path.exists(loan_path) and os.path.exists(acc_path)):
        st.error(f"Missing required data files in `{BASE_DIR}`. Please push Loan.csv and Account.csv to GitHub.")
        st.stop()

    df_loan = pd.read_csv(loan_path).dropna(how="all")
    df_account = pd.read_csv(acc_path).dropna(how="all")

    df_loan["date"] = pd.to_datetime(df_loan["date"], errors="coerce")
    df_account["date"] = pd.to_datetime(df_account["date"], errors="coerce")
    df_loan["is_default"] = df_loan["status"].isin(["B", "D"]).astype(int)

    conn = sqlite3.connect(":memory:")
    df_loan.to_sql("loan", conn, index=False, if_exists="replace")
    df_account.to_sql("account", conn, index=False, if_exists="replace")

    if os.path.exists(trans_path):
        # Read transactions safely
        df_trans = pd.read_csv(trans_path).dropna(how="all").drop_duplicates(subset=["trans_id"])
        df_trans["date"] = pd.to_datetime(df_trans["date"], errors="coerce")
        df_trans["amount"] = pd.to_numeric(df_trans["amount"], errors="coerce")
        df_trans["balance"] = pd.to_numeric(df_trans["balance"], errors="coerce")
        df_trans.to_sql("trans", conn, index=False, if_exists="replace")

        query = """
        SELECT
            l.loan_id,
            l.account_id,
            l.date AS loan_date,
            l.amount AS loan_amount,
            l.duration AS loan_duration,
            l.payments AS monthly_loan_payment,
            l.status,
            l.is_default,
            ROUND((JULIANDAY(l.date) - JULIANDAY(a.date)) / 30.44, 2) AS account_tenure_months,
            COUNT(t.trans_id) AS pre_loan_trans_count,
            COALESCE(AVG(t.balance), 0) AS avg_balance_pre_loan,
            COALESCE(MIN(t.balance), 0) AS min_balance_pre_loan,
            COALESCE(SUM(CASE WHEN t.type LIKE '%Credit%' OR t.type = 'PRIJEM' THEN t.amount ELSE 0 END), 0) AS total_inflow_pre_loan,
            COALESCE(SUM(CASE WHEN t.type LIKE '%Debit%' OR t.type LIKE '%Withdrawal%' OR t.type = 'VYDAJ' THEN t.amount ELSE 0 END), 0) AS total_outflow_pre_loan
        FROM loan l
        JOIN account a ON l.account_id = a.account_id
        LEFT JOIN trans t ON l.account_id = t.account_id AND t.date < l.date
        GROUP BY l.loan_id
        """
    else:
        # Fallback if Transaction.csv is not uploaded due to GitHub's 100MB limit
        query = """
        SELECT
            l.loan_id,
            l.account_id,
            l.date AS loan_date,
            l.amount AS loan_amount,
            l.duration AS loan_duration,
            l.payments AS monthly_loan_payment,
            l.status,
            l.is_default,
            ROUND((JULIANDAY(l.date) - JULIANDAY(a.date)) / 30.44, 2) AS account_tenure_months,
            50 AS pre_loan_trans_count,
            30000 AS avg_balance_pre_loan,
            1000 AS min_balance_pre_loan,
            150000 AS total_inflow_pre_loan,
            120000 AS total_outflow_pre_loan
        FROM loan l
        JOIN account a ON l.account_id = a.account_id
        """

    df_features = pd.read_sql_query(query, conn)
    conn.close()
    return add_derived_features(df_features).reset_index(drop=True)


# ==========================================
# 2. MODEL PIPELINES
# ==========================================
def signed_log1p(x):
    """Preserves negative balances while compressing skewed distributions."""
    return np.sign(x) * np.log1p(np.abs(x))


def get_lr_pipeline():
    return Pipeline([
        ("log", FunctionTransformer(signed_log1p, validate=False)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE)),
    ])


def get_rf_model():
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


@st.cache_resource
def train_predictive_models(df_features):
    X = df_features[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df_features["is_default"]

    # 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_lr = np.zeros(len(y))
    oof_rf = np.zeros(len(y))
    folds_lr = []
    folds_rf = []

    for tr_idx, te_idx in cv.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

        lr_fold = get_lr_pipeline().fit(X_tr, y_tr)
        rf_fold = get_rf_model().fit(X_tr, y_tr)

        p_lr = lr_fold.predict_proba(X_te)[:, 1]
        p_rf = rf_fold.predict_proba(X_te)[:, 1]

        oof_lr[te_idx] = p_lr
        oof_rf[te_idx] = p_rf

        folds_lr.append(roc_auc_score(y_te, p_lr))
        folds_rf.append(roc_auc_score(y_te, p_rf))

    oof_blend = (oof_lr + oof_rf) / 2

    # Fit final calibrated models on 100% of the dataset
    lr_base = get_lr_pipeline().fit(X, y)
    rf_base = get_rf_model().fit(X, y)

    lr_cal = CalibratedClassifierCV(lr_base, method="sigmoid", cv=5).fit(X, y)
    rf_cal = CalibratedClassifierCV(rf_base, method="sigmoid", cv=5).fit(X, y)

    # Temporal split check (earliest 70% vs latest 30%)
    temporal = {k: np.nan for k in ["Logistic Regression", "Random Forest", "Blend (average)"]}
    if "loan_date" in df_features.columns:
        order = np.argsort(pd.to_datetime(df_features["loan_date"]).values, kind="stable")
        cut = int(len(order) * 0.7)
        tr_i, te_i = order[:cut], order[cut:]
        if y.iloc[te_i].nunique() == 2 and y.iloc[tr_i].nunique() == 2:
            p_lr_t = clone(lr_base).fit(X.iloc[tr_i], y.iloc[tr_i]).predict_proba(X.iloc[te_i])[:, 1]
            p_rf_t = clone(rf_base).fit(X.iloc[tr_i], y.iloc[tr_i]).predict_proba(X.iloc[te_i])[:, 1]
            temporal["Logistic Regression"] = roc_auc_score(y.iloc[te_i], p_lr_t)
            temporal["Random Forest"] = roc_auc_score(y.iloc[te_i], p_rf_t)
            temporal["Blend (average)"] = roc_auc_score(y.iloc[te_i], (p_lr_t + p_rf_t) / 2)

    return {
        "temporal": temporal,
        "y": y,
        "oof": {"Logistic Regression": oof_lr, "Random Forest": oof_rf, "Blend (average)": oof_blend},
        "fold_aucs": {"Logistic Regression": folds_lr, "Random Forest": folds_rf},
        "lr_cal": lr_cal,
        "rf_cal": rf_cal,
        "lr_best": lr_base,
        "rf_best": rf_base,
    }


# ==========================================
# 3. INITIALIZE APP & DATA
# ==========================================
with st.spinner("Processing pre-loan temporal data and training models..."):
    df_features = load_and_engineer_features()
    bundle = train_predictive_models(df_features)

y_all = bundle["y"]
base_rate = float(y_all.mean())
oof = bundle["oof"]
auc = {name: roc_auc_score(y_all, p) for name, p in oof.items()}

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("⚙️ Scoring Settings")
engines = ["Random Forest", "Logistic Regression", "Blend (average)"]
engine = st.sidebar.selectbox(
    "Primary scoring engine",
    engines,
    index=engines.index(max(engines, key=lambda n: auc[n])),
    help="Defaults to the model with the highest out-of-fold ROC-AUC.",
)
st.sidebar.caption(f"Portfolio default rate in training data: **{base_rate:.1%}**")
st.sidebar.markdown("**Risk tier cut-offs (PD %)**")
cuts = sorted([
    st.sidebar.number_input("Low → Moderate", 1, 99, 10, 1) / 100,
    st.sidebar.number_input("Moderate → High", 1, 99, 20, 1) / 100,
    st.sidebar.number_input("High → Critical", 1, 99, 40, 1) / 100,
])

# ==========================================
# APP INTERFACE
# ==========================================
st.title("🏦 PKDD Credit Default Risk & Predictive Underwriting")
tab1, tab2 = st.tabs(["📊 Model Performance & Benchmarking", "🎯 Interactive Loan Default Predictor"])

# -----------------------------------------------------------------------------
# TAB 1: MODEL EVALUATION
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Model Evaluation & Discrimination Power")

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
    st.dataframe(summary, use_container_width=True)

    colors = {"Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c", "Blend (average)": "#ff7f0e"}
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        for name, p in oof.items():
            fpr, tpr, _ = roc_curve(y_all, p)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc[name]:.2f})", color=colors[name], lw=2)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title("ROC Curves (Out-of-Fold)")
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
        ax.set_title("Precision-Recall Curves (Out-of-Fold)")
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
        ax.set_title("Key Determinants of Default (Random Forest)")
        ax.set_xlabel("Gini Importance")
        st.pyplot(fig)
        plt.close(fig)

    with c4:
        coefs = pd.Series(
            bundle["lr_best"].named_steps["clf"].coef_[0], index=FEATURE_COLS
        ).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(coefs.index, coefs.values, color=["#2ca02c" if v < 0 else "#d62728" for v in coefs.values])
        ax.axvline(0, color="k", lw=1)
        ax.set_title("Logistic Regression Direction of Effect")
        ax.set_xlabel("Standardised Coefficient (Red = Raises Default Risk)")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------------------------------------------------------
# TAB 2: INTERACTIVE UNDERWRITING SIMULATOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Underwrite a Prospective Loan Application")
    st.markdown("Adjust loan terms and applicant account history. The **Probability of Default (PD)** updates in real time.")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("**1. Loan Contract Terms**")
        input_amount = st.number_input(
            f"Total Loan Principal ({CURRENCY})", min_value=1000, max_value=600000, value=150000, step=5000
        )
        input_duration = st.selectbox("Term Duration (Months)", [12, 24, 36, 48, 60], index=2)
        annual_rate = st.number_input(
            "Annual Interest Rate (%)", min_value=0.0, max_value=30.0, value=0.0, step=0.5,
            help="Default PKDD instalments are principal ÷ term (0% interest). Enter a rate to apply amortization.",
        )

        monthly_rate = annual_rate / 100 / 12
        if monthly_rate == 0:
            auto_payment = input_amount / input_duration
        else:
            auto_payment = input_amount * monthly_rate / (1 - (1 + monthly_rate) ** -input_duration)

        override = st.checkbox("Override instalment manually")
        if override:
            input_payment = st.number_input(
                f"Monthly Instalment ({CURRENCY})", min_value=0.0, max_value=100000.0,
                value=float(round(auto_payment, 2)), step=100.0,
            )
        else:
            input_payment = float(auto_payment)
            st.metric(f"Monthly Instalment ({CURRENCY})", f"{input_payment:,.2f}")

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

    # Input validations
    if input_min_balance > input_avg_balance:
        st.warning("⚠️ Lowest balance is higher than average balance, which is logically invalid.")

    st.divider()

    # Model predictions
    pd_log = bundle["lr_cal"].predict_proba(input_df)[0, 1]
    pd_rf = bundle["rf_cal"].predict_proba(input_df)[0, 1]
    pd_blend = (pd_log + pd_rf) / 2
    primary_pd = {"Random Forest": pd_rf, "Logistic Regression": pd_log, "Blend (average)": pd_blend}[engine]

    low_c, mod_c, high_c = cuts
    if primary_pd < low_c:
        risk_tier, decision_box = "Low Risk", st.success
        action = "✅ **APPROVE**: Low probability of default. Sound cash flow and coverage."
    elif primary_pd < mod_c:
        risk_tier, decision_box = "Moderate Risk", st.info
        action = "⚠️ **MANUAL REVIEW**: Moderate risk. Consider verifying secondary income or requesting guarantees."
    elif primary_pd < high_c:
        risk_tier, decision_box = "High Risk", st.warning
        action = "🛑 **BORDERLINE**: Elevated risk. Recommend reducing principal or extending term."
    else:
        risk_tier, decision_box = "Critical Risk", st.error
        action = "❌ **REJECT**: High probability of delinquency. Cash flow metrics indicate debt stress."

    r = st.columns(4)
    r[0].metric(f"Primary PD ({engine})", f"{primary_pd:.1%}")
    r[1].metric("Random Forest PD", f"{pd_rf:.1%}")
    r[2].metric("Logistic Regression PD", f"{pd_log:.1%}")
    r[3].metric("Assigned Risk Category", risk_tier)

    if abs(pd_rf - pd_log) > 0.15:
        st.caption("⚠️ The models diverge by >15%. Review non-linear payment ratios carefully.")

    decision_box(action)

    with st.expander("🔍 View Computed Financial Underwriting Indicators"):
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Est. Monthly Inflow", f"{CURRENCY}{full_input.at[0, 'avg_monthly_inflow']:,.2f}")
        e2.metric("Payment-to-Inflow Ratio", f"{full_input.at[0, 'payment_to_inflow_ratio']:.2%}")
        e3.metric("Inflow / Outflow Ratio", f"{full_input.at[0, 'inflow_to_outflow_ratio']:.2f}x")
        e4.metric("Balance Cover (Months)", f"{full_input.at[0, 'balance_cover_months']:.1f}")