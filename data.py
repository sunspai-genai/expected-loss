variable_info = {
    "Variable": [
        "Contract_ID", "Equipment_Type", "Credit_Score", "Equipment_Age", "Contract_Term",
        "Loan_to_Value", "Down_Payment_Pct", "Usage_Hours_Per_Week", "Maintenance_Score",
        "Region_Risk_Score", "Customer_Tenure_Years", "Macroeconomic_Index", "Revenue_Last_Year",
        "Operating_Margin", "Prior_Defaults", "Default", "LGD", "EAD"
    ],
    "Description": [
        "Unique ID per lease contract",
        "Type of equipment leased (e.g., Crane)",
        "Customer creditworthiness (300–850 scale)",
        "Age of equipment in years",
        "Lease contract duration in months",
        "Ratio of loan to asset value",
        "Portion paid upfront by customer",
        "Weekly equipment usage",
        "Quality of asset maintenance (0–1)",
        "Risk level of customer's region (0–1)",
        "How long the customer has been with you",
        "Index capturing economic conditions",
        "Customer revenue in last fiscal year",
        "Profitability ratio (0–1)",
        "Whether the customer has defaulted before",
        "Target: whether the contract defaulted",
        "Target: % of exposure lost if defaulted",
        "Target: exposure at the point of default"
    ],
    "Reason_for_Inclusion": [
        "Identifier",
        "Equipment influences asset value and loss behavior",
        "Key indicator of default risk",
        "Older assets may depreciate faster, affecting LGD",
        "Affects EAD and payment stress",
        "High LTV increases risk and LGD",
        "Higher down payments reduce EAD",
        "Indicates wear & tear and potential EAD",
        "Poor maintenance raises LGD risk",
        "Regional economic risk",
        "Experienced customers are less risky",
        "Affects default and recovery risk",
        "Proxy for business health",
        "Weak margins suggest higher risk",
        "Strong signal for PD",
        "Target for PD model",
        "Target for LGD model",
        "Target for EAD model"
    ]
}

#######################

report_dict = {
    "Non-Default": {"precision": 0.83, "recall": 0.84, "f1-score": 0.84, "support": 651},
    "Default": {"precision": 0.70, "recall": 0.68, "f1-score": 0.69, "support": 349},
    "accuracy": 0.78,
    "macro avg": {"precision": 0.76, "recall": 0.76, "f1-score": 0.76, "support": 1000},
    "weighted avg": {"precision": 0.78, "recall": 0.78, "f1-score": 0.78, "support": 1000}
}

# --- Summary metrics dictionary ---
summary_metrics = {
    "Accuracy": 0.78,
    "Precision": 0.70,
    "Recall": 0.68,
    "F1 Score": 0.69,
    "ROC AUC": 0.81
}


##########################################################################################

lgd_variables = [
    {
        "Variable Name": "ContractID",
        "Description": "Unique identifier for the contract",
        "Business Hypothesis": "Identification and indexing purposes"
    },
    {
        "Variable Name": "CreditScoreCategory",
        "Description": "Risk classification of borrower's creditworthiness",
        "Business Hypothesis": "Lower scores are expected to be associated with higher LGD"
    },
    {
        "Variable Name": "Industry",
        "Description": "Industry sector of the borrower",
        "Business Hypothesis": "Some industries may have higher loss rates due to volatility"
    },
    {
        "Variable Name": "Region",
        "Description": "Geographical location of the borrower",
        "Business Hypothesis": "Regional economic conditions can influence recovery"
    },
    {
        "Variable Name": "ContractType",
        "Description": "Type of financing (e.g., Loan or Lease)",
        "Business Hypothesis": "Different contract types have different recovery implications"
    },
    {
        "Variable Name": "CollateralValue",
        "Description": "Value of collateral backing the contract",
        "Business Hypothesis": "Higher collateral typically leads to lower LGD"
    },
    {
        "Variable Name": "RecoveryRate",
        "Description": "Proportion of exposure expected to be recovered",
        "Business Hypothesis": "Directly used in LGD calculation (LGD = 1 - RecoveryRate)"
    },
    {
        "Variable Name": "LGD",
        "Description": "Loss Given Default - target variable",
        "Business Hypothesis": "Target output for model prediction"
    },
    {
        "Variable Name": "% Change in Revenue",
        "Description": "Revenue growth/decline from previous period",
        "Business Hypothesis": "Financial health impacts severity of default loss"
    },
    {
        "Variable Name": "% Change in Net Profit",
        "Description": "Profit growth/decline from previous period",
        "Business Hypothesis": "Indicates risk and ability to recover"
    },
    {
        "Variable Name": "DebtToEquity",
        "Description": "Ratio of total liabilities to equity",
        "Business Hypothesis": "Higher leverage may lead to greater losses"
    },
    {
        "Variable Name": "InterestCoverage",
        "Description": "Earnings before interest / interest expense",
        "Business Hypothesis": "Lower coverage implies greater risk of default loss"
    },
    {
        "Variable Name": "Equipment_Age",
        "Description": "Age of financed equipment in years",
        "Business Hypothesis": "Older equipment depreciates more, affecting recovery"
    },
    {
        "Variable Name": "Lease-to-Value Ratio",
        "Description": "Lease amount divided by collateral value",
        "Business Hypothesis": "Higher ratio means more exposure, less collateral protection"
    },
    {
        "Variable Name": "Maintenance_Score",
        "Description": "Condition rating of equipment (0–100)",
        "Business Hypothesis": "Better maintained assets retain value, improve recovery"
    },
    {
        "Variable Name": "Operating_Margin",
        "Description": "Operating income divided by revenue",
        "Business Hypothesis": "Reflects operational efficiency; weak margin increases LGD"
    },
]
######################################################################################################

ead_variables = [
    {
        "Variable Name": "ContractID",
        "Description": "Unique identifier for the contract",
        "Business Hypothesis": "Identification and indexing purposes"
    },
    {
        "Variable Name": "Industry",
        "Description": "Sector of the borrower",
        "Business Hypothesis": "Influences risk exposure and credit usage"
    },
    {
        "Variable Name": "Region",
        "Description": "Geographical location",
        "Business Hypothesis": "Impacts business stability and exposure patterns"
    },
    {
        "Variable Name": "CompanySize",
        "Description": "Classification as Small, Medium, or Large enterprise",
        "Business Hypothesis": "Smaller firms tend to draw more credit, affecting EAD"
    },
    {
        "Variable Name": "CompanyAge",
        "Description": "Age of the company in years",
        "Business Hypothesis": "Younger companies may have riskier profiles"
    },
    {
        "Variable Name": "ContractTermMonths",
        "Description": "Duration of the credit contract in months",
        "Business Hypothesis": "Longer terms may increase drawn amounts"
    },
    {
        "Variable Name": "CommittedAmount",
        "Description": "Total credit committed by the lender",
        "Business Hypothesis": "Forms upper bound of potential EAD"
    },
    {
        "Variable Name": "CreditUsageRate",
        "Description": "Portion of committed credit already used",
        "Business Hypothesis": "Direct factor in current exposure at default"
    },
    {
        "Variable Name": "EAD",
        "Description": "Exposure at Default - target variable",
        "Business Hypothesis": "Target output for model prediction"
    },
    {
        "Variable Name": "% Change in Revenue",
        "Description": "Revenue growth/decline",
        "Business Hypothesis": "Reflects ability to service credit"
    },
    {
        "Variable Name": "% Change in Net Profit",
        "Description": "Profitability trends",
        "Business Hypothesis": "Indicates sustainability of credit usage"
    },
    {
        "Variable Name": "DebtToEquity",
        "Description": "Leverage ratio",
        "Business Hypothesis": "Higher values may imply greater risk and usage"
    },
    {
        "Variable Name": "CurrentRatio",
        "Description": "Current assets / current liabilities",
        "Business Hypothesis": "Indicates short-term liquidity"
    },
    {
        "Variable Name": "DownPaymentPct",
        "Description": "Portion paid upfront by borrower",
        "Business Hypothesis": "Higher down payments reduce future exposure"
    },
    {
        "Variable Name": "UsageHoursPerWeek",
        "Description": "Utilization of financed equipment per week",
        "Business Hypothesis": "High usage suggests productivity but also wear and tear"
    },
    {
        "Variable Name": "EquipmentType",
        "Description": "Type of financed asset (e.g., Excavator, Crane)",
        "Business Hypothesis": "Different asset types have different risk and exposure patterns"
    },
]
######################################################################################################

pd_variables = [
    {
        "Variable Name": "ContractID",
        "Description": "Unique identifier for the contract",
        "Business Hypothesis": "Identification and indexing purposes"
    },
    
    {
        "Variable Name": "PD",
        "Description": "Probability of Default - target variable",
        "Business Hypothesis": "Target output for model prediction"
    },
        {"Variable Name":"CompanySize",
        "Description": "Size of the company, typically measured in terms of revenue or number of employees.",
        "Business Hypothesis": "Larger companies tend to be more stable and diversified, thus less likely to default."
    },
    {"Variable Name":"Industry",
        "Description": "Sector or type of business activity the company is engaged in.",
        "Business Hypothesis": "Some industries have higher inherent risks (e.g., construction vs. utilities), impacting default probability."
    },
    {"Variable Name":"Region",
        "Description": "Geographical location of the company’s operations or headquarters.",
        "Business Hypothesis": "Economic and political conditions vary by region and can influence credit risk."
    },
    {"Variable Name":"CreditScore",
        "Description": "A numerical score representing the creditworthiness of a company.",
        "Business Hypothesis": "A lower credit score generally correlates with higher default risk."
    },
    {"Variable Name":"PriorDefault",
        "Description": "Indicator of whether the company has defaulted previously (binary variable).",
        "Business Hypothesis": "Past defaults strongly predict future credit behavior and risk of default."
    },
    {"Variable Name":"RegionRiskScore",
        "Description": "Quantitative measure of the economic risk associated with the region.",
        "Business Hypothesis": "High regional risk scores indicate higher external risk pressures affecting company performance."
    },
    {"Variable Name":"MacroEconomicIndex",
        "Description": "Composite index reflecting the overall health of the macroeconomy.",
        "Business Hypothesis": "A weak macroeconomic environment increases the likelihood of financial stress and default."
    },
    {"Variable Name":"PctChangeRevenue",
        "Description": "Percentage change in revenue over a specific period.",
        "Business Hypothesis": "Declining revenue may signal deteriorating financial health and increased default probability."
    },
    {"Variable Name":"PctChangeNetProfit",
        "Description": "Percentage change in net profit over a specific period.",
        "Business Hypothesis": "Shrinking or negative profits can indicate financial distress and a higher risk of default."
    },
    {"Variable Name":"CurrentRatio",
        "Description": "Ratio of current assets to current liabilities.",
        "Business Hypothesis": "Low current ratios may suggest liquidity problems, increasing the likelihood of default."
    },
    {"Variable Name":"DebtToEquity",
        "Description": "Ratio of total debt to shareholders' equity.",
        "Business Hypothesis": "Higher leverage increases financial risk, making companies more vulnerable to default."
    },
    {"Variable Name":"InterestCoverage",
        "Description": "Ratio of earnings before interest and taxes (EBIT) to interest expenses.",
        "Business Hypothesis": "Low interest coverage indicates difficulty in meeting debt obligations, raising default risk."
    }
]
######################################################################################################