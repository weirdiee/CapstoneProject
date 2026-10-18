# Capstone Project : PKDD Financial
# Problem Statement
How can a retail bank optimize loan portfolio risk, identify high-value and churn-prone customer segments, and detect transactional anomalies by integrating customer demographics, account behavior, standing payment instructions, and regional economic indicators?

# Investigation Points
1.	Credit Default Risk & Loan Repayment Modeling
o	Objective: Evaluate key determinants of loan default and repayment status (status) by analyzing loan terms, payment amounts, and historical account cash flow preceding loan issuance.
o	Tool Stack: Python (Logistic Regression / Random Forest classification for loan default prediction) & SQL / SQLite (joining Loan, Account, and Transaction tables to calculate debt service coverage).
2.	Customer Profiling & RFM Behavioral Segmentation
o	Objective: Segment account holders based on transaction Recency, Frequency, and Monetary value (Transaction), cross-analyzed with client age (birth_date), gender, and account ownership role (Disp).
o	Tool Stack: Power BI (interactive DAX measures and RFM dashboard matrix) & Python (K-Means clustering for customer persona generation).
3.	Liquidity Analysis & Standing Payment Order Coverage
o	Objective: Assess monthly account liquidity by tracking balance volatility against recurring payment obligations (Order) such as utilities and loans, highlighting accounts at risk of overdraft.
o	Tool Stack: Power BI (waterfall and line trend visuals for cash flows) & Excel / SQL (recurring transaction aggregation using transaction purpose codes k_symbol).
4.	Regional Macroeconomic Impact on Financial Risk
o	Objective: Correlate regional economic indicators—such as district unemployment rates (A12, A13), average salary (A11), and crime rates (A15, A16)—with regional loan failure rates and payment card adoption (Card).
o	Tool Stack: Python (correlation matrices, Seaborn heatmaps) & Power BI (regional shape maps and district demographic comparisons).
5.	Payment Card Adoption & Channel Usage Patterns
o	Objective: Analyze payment card issuance trends (Card) across demographics and account opening dates (Account), measuring how carded accounts differ in transaction frequency and balance maintenance compared to non-carded accounts.
o	Tool Stack: Power BI (card product distribution analysis) & SQL (cross-tabulating card types against statement frequencies and average balances).
