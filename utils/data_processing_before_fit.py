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
        "fe_1","fe_2","fe_3","fe_4","fe_5","fe_6","fe_7","fe_8","fe_9","fe_10",
        "fe_11","fe_12","fe_13","fe_14","fe_15","fe_16","fe_17","fe_18","fe_19","fe_20"
    ]

    input_df[num_cols] = scaler.fit_transform(input_df[num_cols])

    return input_df