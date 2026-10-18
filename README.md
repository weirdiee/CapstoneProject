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





# Electronics Retail Sales Dataset 📊

<p align="center">
 <img width="266" height="205" alt="image_1e16842e_edited" src="https://github.com/user-attachments/assets/699034ae-2410-4f06-ae9d-66dd5003594f" />
</p>

<h1 align="center">Electronics Retail Sales Dataset</h1>

This repository contains a transactional dataset capturing retail sales performance for consumer electronics products throughout the first half of **2026**. The data includes granular details on order parameters, product categories, geographic regions, salesperson mapping, and promotional discounts.

It serves as an excellent resource for exploratory data analysis (EDA), sales performance forecasting, regional market analysis, and missing data imputation practice.

<img width="1317" height="742" alt="image" src="https://github.com/user-attachments/assets/fd2d0b2f-b7d0-4953-8c8b-868132c2485a" />

## 📋 5 Research Objectives Eamples

1. **Regional Sales Performance & Geographic Market Analysis**
   	> Evaluate total revenue, transaction volume, and average order value across different geographic regions (Region) to identify high-performing markets and underperforming zones.
2. **Impact of Discount Strategies on Net Revenue and Volume**
	> Investigate the relationship between Discount percentages and the resulting Quantity sold versus total net Sales to determine whether discounting successfully stimulates demand or negatively impacts revenue margins.
3. **Product Demand and Revenue Concentration Analysis**
	> Identify which specific Product lines dominate total revenue generation and sales volume to pinpoint core revenue drivers versus low-performing inventory
4. **Salesperson Productivity and Performance Benchmarking**
	> Assess individual sales representatives (Salesperson) based on total sales generated, transaction frequency, and average order size to establish performance benchmarks
5. **Temporal Sales Trends and Demand Seasonality**
	> Examine periodic and monthly variations in sales transactions over time (Date) to detect peak purchasing windows and seasonal demand fluctuations.

## 📋 Dataset Overview

* **File Name:** `product_sales_practice.csv`
* **Time Horizon:** January 2026 – June 2026
* **Target Category:** Electronics
* **Key Entities:** Products, Sales Team, Regional Markets, Revenue Metrics

  Link to Jupyter Notebook in Google Colab:
  https://colab.research.google.com/drive/1zdPR2ig9qFp6Q1gPrZvky3Ac7UITW3Mv?usp=sharing
   
## 🗂️ Data Dictionary

The dataset consists of the following 10 features:

| Column Name | Data Type | Description | Example Value |
| :--- | :--- | :--- | :--- |
| **trans_id** | Integer | Unique identifier assigned to a transaction. | `ORD078` |
| **account_id** | Integer | The transaction date formatted as `YYYY-MM-DD`. | `2026-04-09` |
| **date** | Date | The name of the electronic item sold. | `Laptop`, `Mouse`, `Webcam` |
| **type** | String | The high-level category classification. | `Electronics` |
| **operation** | String | The geographic sales territory. | `North`, `East`, `West`, `South` |
| **amount** | Float | The name of the account representative. | `Alice`, `Bob`, `Charlie`, `Diana`, `Ethan` |
| **balance** | Float | The total units purchased in the order. | `7.0` |
| **k_symbol** | String | The retail price per single unit of the product. | `997.67` |
| **bank** | String | The promotional deduction percentage applied (0.0 to 0.15). | `0.15` (equivalent to 15%) |
| **account** | Integer | The final calculated gross revenue earned from the order. | `6983.69` |

## 🔍 Data Quality & Cleaning Notes

If you are using this dataset for data cleaning projects, please note that it intentionally contains realistic real-world data anomalies:
* **Missing Values (NaNs):** Features like `Region`, `Salesperson`, `Unit_Price`, `Discount`, and `Sales` contain scattered missing fields (e.g., missing revenue values or unattributed regional boundaries).
* **Calculated Columns:** The `Sales` column follows the general formula:  
  \[\text{Sales} = (\text{Quantity} \times \text{Unit\_Price}) \times (1 - \text{Discount})\]
  This relationship can be exploited to programmatically restore missing metrics.
* **Empty Rows:** The raw structure contains blank structural divider rows (`,,,,,,,,,`) that should be filtered out during preprocessing.

## 🛠️ Usage Example (Python & Pandas)

Get started analyzing this dataset quickly using Python:

```python
import pandas as pd

# Load the dataset and drop completely empty structural lines
df = pd.read_csv('sales_data.csv').dropna(how='all')

# Convert date column to datetime object
df['Date'] = pd.to_datetime(df['Date'])

# Quick view of total sales revenue by product
revenue_by_product = df.groupby('Product')['Sales'].sum().sort_values(ascending=False)
print("Revenue by Product:\n", revenue_by_product)
```

## 💡 Potential Projects & Analysis Directions
1. **Sales Performance Dashboards:** Create interactive Tableau or Power BI visualizations mapping regional performance by salesperson.
2. **Predictive Modeling:** Build machine learning regression models to predict expected order values or customer quantity demands.
3. **Discount Impact Studies:** Assess whether offering a $10\%$ vs $15\%$ discount yields a statistically significant increase in total items sold.
