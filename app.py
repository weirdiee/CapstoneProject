import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="PKDD Financial Analytics",
    page_icon="🏦",
    layout="wide"
)

# Plot theme setup
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

# Base directory points directly to the repository folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data
def load_and_clean_data():
    # 1. Transactions
    raw_trans_path = os.path.join(BASE_DIR, 'Transaction.csv')
    df_trans = pd.read_csv(raw_trans_path).dropna(how='all')
    df_trans = df_trans.drop_duplicates(subset=['trans_id'])
    df_trans['date'] = pd.to_datetime(df_trans['date'], errors='coerce')
    df_trans['amount'] = pd.to_numeric(df_trans['amount'], errors='coerce')
    df_trans['balance'] = pd.to_numeric(df_trans['balance'], errors='coerce')

    type_map = {
        'PRIJEM': 'Credit (Income)',
        'VYDAJ': 'Debit (Expense)',
        'VYBER': 'Cash Withdrawal'
    }
    operation_map = {
        'VYBER KE KARTY': 'Credit Card Withdrawal',
        'VKLAD': 'Cash Deposit',
        'PREVOD Z UCTU': 'Remittance from Other Bank',
        'VYBER': 'Cash Withdrawal',
        'PREVOD NA UCET': 'Remittance to Other Bank'
    }
    k_symbol_map = {
        'POJISTNE': 'Insurance Payment',
        'SIPO': 'Household/Utility Payment',
        'UVER': 'Loan Payment',
        'DUCHOD': 'Pension',
        'SLUZBY': 'Service Charge',
        'SANKC. UROK': 'Sanction Interest',
        'LEASING': 'Leasing Payment'
    }
    df_trans['type'] = df_trans['type'].map(type_map).fillna(df_trans['type'])
    df_trans['operation'] = df_trans['operation'].map(operation_map).fillna('Unspecified Operation')
    df_trans['k_symbol'] = df_trans['k_symbol'].map(k_symbol_map).fillna('Uncategorized/Other')
    df_trans['bank'] = df_trans['bank'].fillna('Internal/Cash')
    df_trans['account'] = df_trans['account'].fillna(0).astype(int)

    # 2. Client & Account
    df_client = pd.read_csv(os.path.join(BASE_DIR, 'Client.csv')).dropna(how='all')
    df_client['birth_date'] = pd.to_datetime(df_client['birth_date'], errors='coerce')
    df_client['gender'] = df_client['gender'].str.upper().str.strip()
    df_client['age'] = (pd.to_datetime('1998-12-31') - df_client['birth_date']).dt.days // 365

    df_account = pd.read_csv(os.path.join(BASE_DIR, 'Account.csv')).dropna(how='all')
    df_account['date'] = pd.to_datetime(df_account['date'], errors='coerce')

    # 3. Disp & Card
    df_disp = pd.read_csv(os.path.join(BASE_DIR, 'Disp.csv')).dropna(how='all')
    df_disp['type'] = df_disp['type'].str.upper().str.strip()

    df_card = pd.read_csv(os.path.join(BASE_DIR, 'Card.csv')).dropna(how='all')
    df_card['issued'] = pd.to_datetime(df_card['issued'], errors='coerce')
    df_card['type'] = df_card['type'].str.lower().str.strip()

    # 4. Loan & Order
    df_loan = pd.read_csv(os.path.join(BASE_DIR, 'Loan.csv')).dropna(how='all')
    df_loan['date'] = pd.to_datetime(df_loan['date'], errors='coerce')
    status_map = {
        'A': 'Finished - Paid in Full',
        'B': 'Finished - Defaulted',
        'C': 'Running - Current/Good',
        'D': 'Running - In Debt/Default'
    }
    df_loan['status_description'] = df_loan['status'].map(status_map)
    df_loan['is_default'] = df_loan['status'].isin(['B', 'D']).astype(int)

    df_order = pd.read_csv(os.path.join(BASE_DIR, 'Order.csv')).dropna(how='all')
    df_order['payment_type'] = df_order['k_symbol'].fillna('OTHER')

    # 5. District
    df_district = pd.read_csv(os.path.join(BASE_DIR, 'District.csv')).dropna(how='all')
    district_col_map = {
        'A2': 'district_name',
        'A3': 'region',
        'A4': 'inhabitants',
        'A11': 'average_salary',
        'A12': 'unemployment_rate_95',
        'A13': 'unemployment_rate_96',
        'A14': 'entrepreneurs_per_1000'
    }
    df_district = df_district.rename(columns=district_col_map)
    df_district['unemployment_rate_95'] = df_district['unemployment_rate_95'].fillna(
        df_district.groupby('region')['unemployment_rate_95'].transform('median')
    )

    return df_trans, df_client, df_account, df_disp, df_card, df_loan, df_order, df_district

# Load datasets
with st.spinner("Loading financial datasets from root directory..."):
    try:
        df_trans, df_client, df_account, df_disp, df_card, df_loan, df_order, df_district = load_and_clean_data()
    except Exception as e:
        st.error(f"Error loading files. Ensure your CSV files are uploaded to the root directory. Details: {e}")
        st.stop()

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🏦 PKDD Analytics")
menu = st.sidebar.radio(
    "Select Module:",
    [
        "Executive Overview",
        "1. Credit Default Risk",
        "2. Customer RFM Segmentation",
        "3. Liquidity & Standing Orders",
        "4. Regional Performance",
        "5. Payment Card Synergy"
    ]
)

# ==========================================
# 1. EXECUTIVE OVERVIEW
# ==========================================
if menu == "Executive Overview":
    st.title("📊 PKDD Financial Analytics Dashboard")
    st.markdown("Comprehensive evaluation of portfolio credit risk, customer segmentation, liquidity, and geographic distribution.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Accounts", f"{len(df_account):,}")
    col2.metric("Total Clients", f"{len(df_client):,}")
    col3.metric("Total Loans", f"{len(df_loan):,}")
    col4.metric("Portfolio Default Rate", f"{df_loan['is_default'].mean():.1%}")

    st.divider()
    st.subheader("Data Summary")
    summary_data = {
        "Dataset": ["Transactions", "Accounts", "Clients", "Cards", "Loans", "Orders", "Districts"],
        "Record Count": [len(df_trans), len(df_account), len(df_client), len(df_card), len(df_loan), len(df_order), len(df_district)]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

# ==========================================
# 2. CREDIT DEFAULT RISK
# ==========================================
elif menu == "1. Credit Default Risk":
    st.title("🏦 Credit Default Risk & Repayment Modeling")

    conn = sqlite3.connect(':memory:')
    df_loan.to_sql('loan', conn, index=False, if_exists='replace')
    df_account.to_sql('account', conn, index=False, if_exists='replace')
    df_trans.to_sql('trans', conn, index=False, if_exists='replace')

    query = """
    SELECT
        l.loan_id, l.account_id, l.date AS loan_date, l.amount AS loan_amount,
        l.duration AS loan_duration, l.payments AS monthly_loan_payment,
        l.status, l.is_default, a.date AS account_open_date,
        ROUND((JULIANDAY(l.date) - JULIANDAY(a.date)) / 30.44, 2) AS account_tenure_months,
        COUNT(t.trans_id) AS pre_loan_trans_count,
        COALESCE(AVG(t.balance), 0) AS avg_balance_pre_loan,
        COALESCE(MIN(t.balance), 0) AS min_balance_pre_loan,
        COALESCE(MAX(t.balance), 0) AS max_balance_pre_loan,
        COALESCE(SUM(CASE WHEN t.type LIKE '%Credit%' OR t.type = 'PRIJEM' THEN t.amount ELSE 0 END), 0) AS total_inflow_pre_loan,
        COALESCE(SUM(CASE WHEN t.type LIKE '%Debit%' OR t.type LIKE '%Withdrawal%' OR t.type = 'VYDAJ' THEN t.amount ELSE 0 END), 0) AS total_outflow_pre_loan
    FROM loan l
    JOIN account a ON l.account_id = a.account_id
    LEFT JOIN trans t ON l.account_id = t.account_id AND t.date < l.date
    GROUP BY l.loan_id
    """
    df_feat = pd.read_sql_query(query, conn)
    conn.close()

    df_feat['pre_loan_months'] = np.maximum(df_feat['account_tenure_months'], 1.0)
    df_feat['avg_monthly_inflow'] = df_feat['total_inflow_pre_loan'] / df_feat['pre_loan_months']
    df_feat['payment_to_inflow_ratio'] = df_feat['monthly_loan_payment'] / (df_feat['avg_monthly_inflow'] + 1.0)
    df_feat['loan_to_avg_balance'] = df_feat['loan_amount'] / (df_feat['avg_balance_pre_loan'] + 1.0)
    df_feat['inflow_to_outflow_ratio'] = df_feat['total_inflow_pre_loan'] / (df_feat['total_outflow_pre_loan'] + 1.0)

    feature_cols = [
        'loan_amount', 'loan_duration', 'monthly_loan_payment',
        'account_tenure_months', 'pre_loan_trans_count', 'avg_balance_pre_loan',
        'min_balance_pre_loan', 'payment_to_inflow_ratio', 'loan_to_avg_balance',
        'inflow_to_outflow_ratio'
    ]
    X = df_feat[feature_cols].fillna(0)
    y = df_feat['is_default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_model.fit(X_train, y_train)
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    col1, col2 = st.columns(2)
    col1.metric("Random Forest ROC-AUC", f"{auc_rf:.4f}")
    col2.metric("Total Defaulted Loans", f"{y.sum()} / {len(y)}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    axes[0].plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='#2ca02c')
    axes[0].plot([0, 1], [0, 1], 'k--')
    axes[0].set_title('ROC Curve')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()

    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    importances.plot(kind='barh', ax=axes[1], color='#d62728')
    axes[1].set_title('Feature Importances')
    axes[1].set_xlabel('Gini Importance')
    st.pyplot(fig)

# ==========================================
# 3. RFM SEGMENTATION
# ==========================================
elif menu == "2. Customer RFM Segmentation":
    st.title("👥 Customer RFM Behavioral Segmentation")

    snapshot_date = pd.to_datetime('1998-12-31')
    rfm = df_trans.groupby('account_id').agg(
        recency_days=('date', lambda x: (snapshot_date - x.max()).days),
        frequency_count=('trans_id', 'count'),
        monetary_value=('amount', 'sum')
    ).reset_index()

    df_owner = df_disp[df_disp['type'] == 'OWNER'].merge(df_client, on='client_id', how='left')
    rfm_df = rfm.merge(df_owner[['account_id', 'client_id', 'gender', 'birth_date']], on='account_id', how='left')
    rfm_df['age'] = ((snapshot_date - rfm_df['birth_date']).dt.days / 365.25).round(1)
    rfm_df = rfm_df.dropna(subset=['recency_days', 'frequency_count', 'monetary_value', 'age'])

    rfm_df['log_R'] = np.log1p(rfm_df['recency_days'])
    rfm_df['log_F'] = np.log1p(rfm_df['frequency_count'])
    rfm_df['log_M'] = np.log1p(rfm_df['monetary_value'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_df[['log_R', 'log_F', 'log_M', 'age']])
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_df['cluster'] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.countplot(data=rfm_df, x='cluster', palette='viridis', ax=ax)
    ax.set_title('Customer Count by K-Means Cluster')
    ax.set_xlabel('Cluster ID')
    ax.set_ylabel('Number of Accounts')
    st.pyplot(fig)

    st.subheader("Cluster Profile Metrics (Averages)")
    st.dataframe(rfm_df.groupby('cluster')[['recency_days', 'frequency_count', 'monetary_value', 'age']].