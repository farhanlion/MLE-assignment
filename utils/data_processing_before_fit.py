import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_features(input_df):
    
    # Replace "_" with NaN
    string_cols = input_df.select_dtypes(include="object").columns.tolist()
    for c in string_cols:
        input_df[c] = input_df[c].replace("_", np.nan)

    # One-hot encode categoricals
    onehot_cols = ["Credit_Mix", "Payment_Behaviour", "Occupation"]
    input_df = pd.get_dummies(input_df, columns=onehot_cols, drop_first=False)

    # Convert boolean columns to 0/1
    bool_cols = input_df.select_dtypes(include="bool").columns
    input_df[bool_cols] = input_df[bool_cols].astype(int)

    # Normalize numeric cols
    scaler = StandardScaler()
    num_cols = [
    "Annual_Income","Monthly_Inhand_Salary","Num_Bank_Accounts","Num_Credit_Card",
    "Interest_Rate","Num_of_Loan","Delay_from_due_date","Num_of_Delayed_Payment",
    "Changed_Credit_Limit","Num_Credit_Inquiries","Credit_Utilization_Ratio",
    "Credit_History_Age",
    "fe_1_5m_avg","fe_2_5m_avg","fe_3_5m_avg","fe_4_5m_avg","fe_5_5m_avg",
    "fe_6_5m_avg","fe_7_5m_avg","fe_8_5m_avg","fe_9_5m_avg","fe_10_5m_avg",
    "fe_11_5m_avg","fe_12_5m_avg","fe_13_5m_avg","fe_14_5m_avg","fe_15_5m_avg",
    "fe_16_5m_avg","fe_17_5m_avg","fe_18_5m_avg","fe_19_5m_avg","fe_20_5m_avg"
    ]


    input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

    return input_df