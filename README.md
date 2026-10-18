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
| **Order_ID** | String | A unique alphanumeric identifier for each transaction. | `ORD078` |
| **Date** | Date | The transaction date formatted as `YYYY-MM-DD`. | `2026-04-09` |
| **Product** | String | The name of the electronic item sold. | `Laptop`, `Mouse`, `Webcam` |
| **Category** | String | The high-level category classification. | `Electronics` |
| **Region** | String | The geographic sales territory. | `North`, `East`, `West`, `South` |
| **Salesperson** | String | The name of the account representative. | `Alice`, `Bob`, `Charlie`, `Diana`, `Ethan` |
| **Quantity** | Integer | The total units purchased in the order. | `7.0` |
| **Unit_Price** | Float | The retail price per single unit of the product. | `997.67` |
| **Discount** | Float | The promotional deduction percentage applied (0.0 to 0.15). | `0.15` (equivalent to 15%) |
| **Sales** | Float | The final calculated gross revenue earned from the order. | `6983.69` |

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
