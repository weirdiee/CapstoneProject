import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Credit Default Risk Modeling",
    page_icon="🏦",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DATA PIPELINE & FEATURE ENGINEERING
# ==========================================
@st.cache_data
def load_and_engineer_features():
    loan_path = os.path.join(BASE_DIR, 'Loan.csv')
    acc_path = os.path.join(BASE_DIR, 'Account.csv')
    trans_path = os.path.join(BASE_DIR, 'Transaction.csv')

    df_loan = pd.read_csv(loan_path).dropna(how='all')
    df_account = pd.read_csv(acc_path).dropna(how='all')
    df_trans = pd.read_csv(trans_path).dropna(how='all').drop_duplicates(subset=['trans_id'])

    # Format dates and target variable
    df_loan['date'] = pd.to_datetime(df_loan['date'], errors='coerce')
    df_account['date'] = pd.to_datetime(df_account['date'], errors='coerce')
    df_trans['date'] = pd.to_datetime(df_trans['date'], errors='coerce')
    df_trans['amount'] = pd.to_numeric(df_trans['amount'], errors='coerce')
    df_trans['balance'] = pd.to_numeric(df_trans['balance'], errors='coerce')

    # B and D indicate loan default
    df_loan['is_default'] = df_loan['status'].isin(['B', 'D']).astype(int)

    # In-memory SQLite join to extract pre-loan financial health
    conn = sqlite3.connect(':memory:')
    df_loan.to_sql('loan', conn, index=False, if_exists='replace')
    df_account.to_sql('account', conn, index=False, if_exists='replace')
    df_trans.to_sql('trans', conn, index=False, if_exists='replace')

    query = """
    SELECT
        l.loan_id,
        l.account_id,
        l.amount AS loan_amount,
        l.duration AS loan_duration,
        l.payments AS monthly_loan_payment,
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
    df_features = pd.read_sql_query(query, conn)
    conn.close()

    # Derived financial ratios
    df_features['pre_loan_months'] = np.maximum(df_features['account_tenure_months'], 1.0)
    df_features['avg_monthly_inflow'] = df_features['total_inflow_pre_loan'] / df_features['pre_loan_months']
    df_features['payment_to_inflow_ratio'] = df_features['monthly_loan_payment'] / (df_features['avg_monthly_inflow'] + 1.0)
    df_features['loan_to_avg_balance'] = df_features['loan_amount'] / (df_features['avg_balance_pre_loan'] + 1.0)
    df_features['inflow_to_outflow_ratio'] = df_features['total_inflow_pre_loan'] / (df_features['total_outflow_pre_loan'] + 1.0)

    return df_features

# ==========================================
# 2. MODEL TRAINING & RESOURCE CACHING
# ==========================================
FEATURE_COLS = [
    'loan_amount', 'loan_duration', 'monthly_loan_payment',
    'account_tenure_months', 'pre_loan_trans_count', 'avg_balance_pre_loan',
    'min_balance_pre_loan', 'payment_to_inflow_ratio', 'loan_to_avg_balance',
    'inflow_to_outflow_ratio'
]

@st.cache_resource
def train_predictive_models(df_features):
    X = df_features[FEATURE_COLS].fillna(0)
    y = df_features['is_default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Logistic Regression
    log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    # 2. Random Forest
    rf_model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)

    return scaler, log_reg, rf_model, X_test, X_test_scaled, y_test

# Load data and train models
with st.spinner("Processing pre-loan temporal data and training models..."):
    df_features = load_and_engineer_features()
    scaler, log_reg, rf_model, X_test, X_test_scaled, y_test = train_predictive_models(df_features)

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

    # Generate test predictions
    y_prob_log = log_reg.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    auc_log = roc_auc_score(y_test, y_prob_log)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Dataset Loans Analyzed", f"{len(df_features):,}")
    mcol2.metric("Logistic Regression ROC-AUC", f"{auc_log:.4f}")
    mcol3.metric("Random Forest ROC-AUC", f"{auc_rf:.4f}")

    col_left, col_right = st.columns(2)

    with col_left:
        # ROC Curves
        fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={auc_log:.2f})', color='#1f77b4', lw=2)
        ax_roc.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.2f})', color='#2ca02c', lw=2)
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_title("ROC Curves Comparison")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)

    with col_right:
        # Feature Importances (Random Forest)
        importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
        fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
        importances.plot(kind='barh', ax=ax_imp, color='#d62728')
        ax_imp.set_title("Key Determinants of Default (Random Forest)")
        ax_imp.set_xlabel("Gini Importance")
        st.pyplot(fig_imp)

# -----------------------------------------------------------------------------
# TAB 2: INTERACTIVE UNDERWRITING SIMULATOR
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Underwrite a Prospective Loan Application")
    st.markdown("Adjust loan terms and applicant account history to compute the real-time **Probability of Default (PD)**.")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        st.markdown("**1. Loan Contract Terms**")
        input_amount = st.number_input("Total Loan Principal ($)", min_value=1000, max_value=600000, value=150000, step=5000)
        input_duration = st.selectbox("Term Duration (Months)", [12, 24, 36, 48, 60], index=2)
        input_payment = st.number_input("Monthly Installment ($)", min_value=100, max_value=20000, value=4500, step=100)

    with col_input2:
        st.markdown("**2. Account History & Tenure**")
        input_tenure = st.slider("Account Relationship Tenure (Months)", min_value=1.0, max_value=72.0, value=24.0, step=0.5)
        input_trans_count = st.number_input("Pre-Loan Completed Transactions", min_value=1, max_value=1000, value=120, step=5)
        input_avg_balance = st.number_input("Average Pre-Loan Account Balance ($)", min_value=0, max_value=200000, value=35000, step=1000)
        input_min_balance = st.number_input("Lowest Pre-Loan Balance Observed ($)", min_value=-50000, max_value=100000, value=2500, step=500)

    with col_input3:
        st.markdown("**3. Historical Cash Flow Velocity**")
        input_total_inflow = st.number_input("Total Historical Inflow / Credits ($)", min_value=1000, max_value=2000000, value=280000, step=10000)
        input_total_outflow = st.number_input("Total Historical Outflow / Debits ($)", min_value=1000, max_value=2000000, value=245000, step=10000)

    # Compute derived ratios dynamically
    calc_monthly_inflow = input_total_inflow / max(input_tenure, 1.0)
    calc_payment_ratio = input_payment / (calc_monthly_inflow + 1.0)
    calc_loan_to_balance = input_amount / (input_avg_balance + 1.0)
    calc_in_out_ratio = input_total_inflow / (input_total_outflow + 1.0)

    input_df = pd.DataFrame([{
        'loan_amount': input_amount,
        'loan_duration': input_duration,
        'monthly_loan_payment': input_payment,
        'account_tenure_months': input_tenure,
        'pre_loan_trans_count': input_trans_count,
        'avg_balance_pre_loan': input_avg_balance,
        'min_balance_pre_loan': input_min_balance,
        'payment_to_inflow_ratio': calc_payment_ratio,
        'loan_to_avg_balance': calc_loan_to_balance,
        'inflow_to_outflow_ratio': calc_in_out_ratio
    }])

    st.divider()

    # Predict button
    if st.button("🚀 Calculate Default Risk & Underwriting Decision", type="primary"):
        # Scale for Logistic Regression
        input_scaled = scaler.transform(input_df)

        pd_log = log_reg.predict_proba(input_scaled)[0, 1]
        pd_rf = rf_model.predict_proba(input_df)[0, 1]

        # Use Random Forest as primary scoring engine
        primary_pd = pd_rf

        # Assign risk tier
        if primary_pd < 0.25:
            risk_tier = "Low Risk"
            decision_box = st.success
            action = "✅ **APPROVE**: Low risk of default. Meets debt-service and tenure requirements."
        elif primary_pd < 0.50:
            risk_tier = "Moderate Risk"
            decision_box = st.info
            action = "⚠️ **MANUAL REVIEW**: Moderate risk. Consider requesting collateral or credit card linkage."
        elif primary_pd < 0.75:
            risk_tier = "High Risk"
            decision_box = st.warning
            action = "🛑 **BORDERLINE**: Elevated risk. Recommend reducing loan principal or extending term."
        else:
            risk_tier = "Critical Risk"
            decision_box = st.error
            action = "❌ **REJECT**: High probability of payment delinquency. Unfavorable payment-to-inflow ratio."

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Random Forest PD", f"{primary_pd:.1%}")
        res_col2.metric("Logistic Regression PD", f"{pd_log:.1%}")
        res_col3.metric("Assigned Risk Category", risk_tier)

        decision_box(action)

        with st.expander("🔍 View Computed Financial Underwriting Indicators"):
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Est. Monthly Inflow", f"${calc_monthly_inflow:,.2f}")
            r_col2.metric("Payment-to-Inflow Ratio (DSCR)", f"{calc_payment_ratio:.2%}")
            r_col3.metric("Inflow / Outflow Ratio", f"{calc_in_out_ratio:.2f}x")