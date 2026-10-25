
+ **Prepared By : Daren Lee**
+ **Dataset : Transactions**

# PKDD Financial
<p align="center">
 <img width="312" height="208" alt="FKDD_Logo" src="https://github.com/user-attachments/assets/6cb3afaa-5501-40b3-a146-14922a3d0595" />
</p>

<h1 align="center">PKDD Financial Database Dataset</h1>

## Description
This repository contains a multi-table relational banking database capturing retail financial activity, customer account movements, and credit usage across individual accounts and regional districts through **January 1993 to December 1998**. The data includes granular transaction ledger records detailing dates, credit/debit operations, transaction amounts, post-transaction balances, counterparty bank codes, and payment purpose tags (**k_symbol**), alongside relational tables for customer demographics (**Client**), account ownership roles (**Disp**), loan repayment histories (**Loan**), payment card issuances (**Card**), recurring standing orders (**Order**), and district-level economic statistics (**District**).

It serves as an excellent resource for exploratory data analysis (EDA), credit default risk modeling, customer profiling and RFM behavioral segmentation, account liquidity monitoring, regional macroeconomic risk analysis, and payment card adoption analytics.

## Problem Statement
How can a retail bank optimize loan portfolio risk, identify high-value and churn-prone customer segments, and detect transactional anomalies by integrating customer demographics, account behavior, standing payment instructions, and regional economic indicators?

## 📋 5 Research Objectives Eamples

1. **Credit Default Risk & Loan Repayment Modeling**
   	> Objective: Evaluate key determinants of loan default and repayment status (status) by analyzing loan terms, payment amounts, and historical account cash flow preceding loan issuance.
    > Tool Stack: Python (Logistic Regression / Random Forest classification for loan default prediction) & SQL / SQLite (joining Loan, Account, and Transaction tables to calculate debt service coverage).
2. **Customer Profiling & RFM Behavioral Segmentation**
	   > Objective: Segment account holders based on transaction Recency, Frequency, and Monetary value (Transaction), cross-analyzed with client age (birth_date), gender, and account ownership role (Disp).
    > Tool Stack: Power BI (interactive DAX measures and RFM dashboard matrix) & Python (K-Means clustering for customer persona generation).
3. **Liquidity Analysis & Standing Payment Order Coverage**
	   > Objective: Assess monthly account liquidity by tracking balance volatility against recurring payment obligations (Order) such as utilities and loans, highlighting accounts at risk of overdraft.
    >	Tool Stack: Power BI (waterfall and line trend visuals for cash flows) & Excel / SQL (recurring transaction aggregation using transaction purpose codes k_symbol).
4. **Regional Macroeconomic Impact on Financial Risk**
	   > Objective: Correlate regional economic indicators—such as district unemployment rates (A12, A13), average salary (A11), and crime rates (A15, A16)—with regional loan failure rates and payment card adoption (Card).
    > Tool Stack: Python (correlation matrices, Seaborn heatmaps) & Power BI (regional shape maps and district demographic comparisons).
5. **Payment Card Adoption & Channel Usage Patterns**
	   > Objective: Analyze payment card issuance trends (Card) across demographics and account opening dates (Account), measuring how carded accounts differ in transaction frequency and balance maintenance compared to non-carded accounts.
    > Tool Stack: Power BI (card product distribution analysis) & SQL (cross-tabulating card types against statement frequencies and average balances).

## 📋 Dataset Overview

* **File Name:** `Transaction.csv`
* **Time Horizon:** January 1993 – December 1998
* **Target Category:** Retail Banking & Financial Services
* **Key Entities:** Customer Accounts (account_id), Credit/Debit Operations (type, operation), Ledger Balances & Amounts (amount, balance), Payment Purpose Tags (k_symbol), Counterparty Banks & Accounts (bank, account)

  Link to Jupyter Notebook in Google Colab:
  https://colab.research.google.com/drive/1xlipoYS-Dwa7UtsnrqJgKSW7RRfsprBm

## 🗂️ Data Dictionary

The dataset consists of the following 10 features:

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| **trans_id** | Integer | Unique identifier assigned to a transaction. | `386012` |
| **account_id** | Integer | Unique identifier for the customer bank account. | `1312` |
| **date** | Date | The transaction date formatted as `DD-MM-YYYY`.  | `10/4/1995` |
| **type** | String | Transaction category indicating credit/income (PRIJEM) or debit/expense (VYDAJ). | `PRIJEM` |
| **operation** | String | Mode of transaction performed (e.g. bank transfer, cash deposit, cash withdrawal). | `PREVOD' 'Z' 'UCTU` |
| **amount** | Float | Transaction monetary value. | `5735.00` |
| **balance** | Float | Account balance remaining immediately following the transaction. | `31973.00` |
| **k_symbol** | String | Standardized payment purpose code tag (e.g. pension DUCHOD, utility SIPO, loan payment UVER). | `DUCHOD` |
| **bank** | String | Two-letter counterparty bank identification code. | `QR` |
| **account** | Integer | Counterparty account number involved in the transaction transfer. | `82899385` |

## 🔍 Data Quality & Cleaning Notes

If you are using this dataset for data cleaning projects, please note that it intentionally contains realistic real-world data anomalies:
* **Missing Values (NaNs) :** Features like payment purpose tags (k_symbol) in Order.csv and Transaction.csv, counterparty identifiers (bank and account for cash-based transactions), as well as regional economic indicators (A12 unemployment rate and A15 crime statistics in District.csv) contain scattered missing fields (e.g., unclassified standing orders or cash withdrawals lacking counterparty bank codes).
* **Domain & Czech Term Standardization :** Key categorical features utilize legacy Czech financial codes and terms—such as transaction type type (PRIJEM for credit, VYDAJ for debit), purpose tags k_symbol (SIPO for household utilities, UVER for loan payments, POJISTNE for insurance, DUCHOD for pensions), and account statement frequencies (POPLATEK MESICNE). These must be mapped to English terms for standardized Power BI and Python dashboard reporting.
* **Multi-Part Ingestion & Deduplication :** The core transaction ledger is split across 36 separate multi-part CSV files (Transaction_part_1 through Transaction_part_36). Data pipelines must programmatically concatenate these files into a unified dataframe or SQLite schema while enforcing unique key constraints on trans_id to eliminate duplicate records.
* **Date Parsing & Temporal Alignment :** Dates across tables (date in Transaction_part_xx, Account, and Loan, issued in Card, and birth_date in Client) appear in varying text date formats. All fields must be converted into standard ISO dates (YYYY-MM-DD) to enable contiguous Date Table relationships and DAX time-intelligence functions in Power BI.
* **Referential Integrity & Conditional Imputation :** Relational links between account_id, client_id, disp_id, and district_id require key validation across the eight relational tables [1–10]. Additionally, unassigned k_symbol values in transaction records can be conditionally imputed by matching recurring amounts and counterparty accounts against standing instructions in Order.csv.

## 🛠️ Usage Example (Python & Pandas)

Get started analyzing this dataset quickly using excel (due to the huge datasets of one million transactions) and Python (data cleansing) :
import pandas as pd
import glob

### 1. Load and concatenate all multi-part transaction CSV files
file_paths = glob.glob('Transaction_part_*.csv')
df_list = [pd.read_csv(file).dropna(how='all') for file in file_paths]
df = pd.concat(df_list, ignore_index=True)

### 2. Convert date column to standardized datetime object
df['date'] = pd.to_datetime(df['date'])

### 3. Map legacy Czech financial codes to English business terms
type_mapping = {
    'PRIJEM': 'Credit/Income',
    'VYDAJ': 'Debit/Expense',
    'VYBER': 'Cash Withdrawal'
}
k_symbol_mapping = {
    'POJISTNE': 'Insurance Payment',
    'SIPO': 'Household/Utility Payment',
    'UVER': 'Loan Payment',
    'DUCHOD': 'Pension',
    'SLUZBY': 'Service Charge',
    'SANKC. UROK': 'Sanction Interest'
}

df['type'] = df['type'].map(type_mapping).fillna(df['type'])
df['k_symbol'] = df['k_symbol'].map(k_symbol_mapping).fillna('Uncategorized/Other')

### 4. Fill missing counterparty attributes for cash transactions
df['bank'] = df['bank'].fillna('Internal/Cash')
df['account'] = df['account'].fillna(0).astype(int)

### 5. Quick view of total transaction volume by payment purpose tag
transaction_summary = df.groupby('k_symbol')['amount'].agg(['count', 'sum']).rename(columns={'count': 'Transaction_Count', 'sum': 'Total_Amount'}).sort_values(by='Total_Amount', ascending=False)
print("Transaction Summary by Purpose:\n", transaction_summary)

## 💡 Potential Projects & Analysis Directions
1. **Credit Default Risk & Loan Repayment Modeling :** Build predictive classification models in Python (Logistic Regression / Random Forest) and interactive Power BI risk dashboards to score borrower default risk using historical cash flow indicators from Transaction_part_xx and Loan repayment statuses.
2. **Customer Profiling & RFM Behavioral Segmentation :** Implement K-Means clustering in Python and DAX segmentation matrices in Power BI to classify account holders by Recency, Frequency, and Monetary (RFM) metrics derived from Transaction_part_xx, Client demographics, and Disp ownership roles.
3. **Liquidity Analysis & Standing Order Coverage :** Construct cash flow trend models and overdraft risk tracking in Power BI by comparing running ledger balances in Transaction_part_xx against recurring monthly commitments in Order categorized by k_symbol purpose tags.
4. **Regional Macroeconomic Impact Analysis :** Perform spatial heatmap analysis and cross-sectional correlations mapping district economic metrics (average salary, unemployment, and crime rates from District) against localized transaction volumes and regional default rates.
5. **Payment Card Adoption & Channel Usage Studies :** Evaluate ledger transaction frequency and balance changes in Transaction_part_xx pre- and post-card issuance (issued in Card), analyzing how card tiers (classic, junior, gold) influence electronic payment behavior.



<img width="320" height="160" alt="Master_Executive_View" src="https://github.com/user-attachments/assets/7af8658d-d7bf-4ccc-bc83-4b04d756bb2b" />
<img width="320" height="160" alt="Research_PBI_1" src="https://github.com/user-attachments/assets/d6ffc2ff-73ab-44d0-90fe-e52e87b6c808" />
<img width="320" height="160" alt="Research_PBI_2" src="https://github.com/user-attachments/assets/9e071a5b-54e1-4dee-93e9-f901092340ad" />
<img width="320" height="160" alt="Research_PBI_3" src="https://github.com/user-attachments/assets/9f332f7d-9d94-4451-bd30-f9faa8bf430f" />
<img width="320" height="160" alt="Research_PBI_4" src="https://github.com/user-attachments/assets/fe1b73cb-7a00-4dba-afdf-1d9241aee734" />
<img width="320" height="160" alt="Research_PBI_5" src="https://github.com/user-attachments/assets/fd5a3ec1-2157-4526-a9fa-c6a27b4cde8a" />






