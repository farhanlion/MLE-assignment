import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse
import re


from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

_DATE_IN_NAME = re.compile(r".*?(\d{4})_(\d{2})_(\d{2})\.csv$")

def _infer_snapshot_from_name(filename: str):
    m = _DATE_IN_NAME.match(filename)
    if not m:
        return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"

def process_silver_table_attr(bronze_attr_directory, silver_attr_directory, spark):

    bronze_files = [f for f in os.listdir(bronze_attr_directory) if f.lower().endswith(".csv")]
    if not bronze_files:
        print(f"[SILVER/ATTR] No CSVs found in {bronze_attr_directory}")
        return None

    bronze_files.sort()
    print(f"[SILVER/ATTR] Found {len(bronze_files)} bronze attribute partitions...")


    for fname in bronze_files:
        bronze_path = os.path.join(bronze_attr_directory, fname)

        # Try to infer snapshot date from filename
        snapshot_date_str = _infer_snapshot_from_name(fname)

        # Read bronze CSV
        df_attr = spark.read.csv(bronze_path, header=True, inferSchema=True)
        
         # SSN: replace the bad token with a formatted placeholder
        df_attr = df_attr.withColumn(
            "SSN",
            F.when(F.col("SSN") == "#F%$D@*&8", "000-00-0000").otherwise(F.col("SSN"))
        )

        # Occupation: replace placeholder with "Unemployed"
        df_attr = df_attr.withColumn(
            "Occupation",
            F.when(F.col("Occupation") == "_______", "Unemployed").otherwise(F.col("Occupation"))
        )

        df_attr = df_attr.withColumn("Age", F.regexp_replace(F.col("Age"), "_", ""))
        df_attr = df_attr.withColumn("Age", F.col("Age").cast(IntegerType()))
        df_attr = df_attr.withColumn(
            "Age",
            F.when((F.col("Age") >= 0) & (F.col("Age") <= 100), F.col("Age")).otherwise(F.lit(None))
        )

        column_type_map = {
            "Customer_ID": StringType(),
            "Name": StringType(),
            "Age": IntegerType(),
            "SSN": StringType(),
            "Occupation": StringType(),
            "snapshot_date": DateType(),
        }
        
        for column, new_type in column_type_map.items():
                df_attr = df_attr.withColumn(column, col(column).cast(new_type))
                 
        # Write to silver with consistent naming: silver_attr_YYYY_MM_DD.parquet
        out_name = f"silver_attr_{snapshot_date_str.replace('-', '_')}.parquet"
        out_path = os.path.join(silver_attr_directory, out_name)

        df_attr.write.mode("overwrite").parquet(out_path)
    
    print(f"[SILVER/ATTR] Finished writing to {silver_attr_directory}")
    return True       

def process_silver_table_lms(bronze_lms_directory, silver_loan_daily_directory, spark):

    bronze_files = [f for f in os.listdir(bronze_lms_directory) if f.lower().endswith(".csv")]
    # keep only expected LMS files
    bronze_files = [f for f in bronze_files if _infer_snapshot_from_name(f)]
    if not bronze_files:
        print(f"[SILVER/LMS] No LMS CSVs found in {bronze_lms_directory}")
        return None

    bronze_files.sort()
    print(f"[SILVER/LMS] Found {len(bronze_files)} bronze LMS partitions...")

    for fname in bronze_files:
        snapshot_date_str = _infer_snapshot_from_name(fname)
        bronze_path = os.path.join(bronze_lms_directory, fname)

        # Read bronze CSV
        df = spark.read.csv(bronze_path, header=True, inferSchema=True)

        # ---- Enforce schema / types ----
        column_type_map = {
            "loan_id": StringType(),
            "Customer_ID": StringType(),
            "loan_start_date": DateType(),
            "tenure": IntegerType(),
            "installment_num": IntegerType(),
            "loan_amt": FloatType(),
            "due_amt": FloatType(),
            "paid_amt": FloatType(),
            "overdue_amt": FloatType(),
            "balance": FloatType(),
            "snapshot_date": DateType(),
        }
        for column, new_type in column_type_map.items():
            if column in df.columns:
                df = df.withColumn(column, col(column).cast(new_type))

        # ---- Augmentations ----
        # month-on-book
        df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
        # installments missed (ceil(overdue/due)), first missed date, DPD
        df = df.withColumn(
            "installments_missed",
            F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
        ).fillna({"installments_missed": 0})
        df = df.withColumn(
            "first_missed_date",
            F.when(col("installments_missed") > 0,
                   F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
            ).cast(DateType())
        )
        df = df.withColumn(
            "dpd",
            F.when(col("overdue_amt") > 0.0,
                   F.datediff(col("snapshot_date"), col("first_missed_date")))
             .otherwise(F.lit(0))
             .cast(IntegerType())
        )

        # ---- Write Silver parquet per snapshot ----
        out_name = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
        out_path = os.path.join(silver_loan_daily_directory, out_name)
        df.write.mode("overwrite").parquet(out_path)

    print(f"[SILVER/LMS] Finished writing to {silver_loan_daily_directory}")
    return True


def clean_numeric_column(df, column_name):
    """
    Cleans a numeric column in a PySpark DataFrame:
    1. Removes stray underscores from numeric values.
    2. Casts the column to FloatType.
    3. Uses IQR to detect outliers.
    4. Replaces outliers with None.
    
    Args:
        df (DataFrame): Input PySpark DataFrame
        column_name (str): Name of the numeric column to clean

    Returns:
        DataFrame: Cleaned DataFrame with outliers replaced by None
    """
    # 1️⃣ Remove underscores and cast to float
    dtype_lookup = dict(df.dtypes).get(column_name)
    if dtype_lookup == "string":
        df = df.withColumn(column_name, F.regexp_replace(F.col(column_name), "_", ""))

    df = df.withColumn(column_name, F.col(column_name).cast(FloatType()))

    # 2️⃣ Compute IQR boundaries
    q1, q3 = df.approxQuantile(column_name, [0.25, 0.75], 0.01)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # 3️⃣ Replace outliers with None
    df = df.withColumn(
        column_name,
        F.when((F.col(column_name) >= lower) & (F.col(column_name) <= upper),
               F.col(column_name))
         .otherwise(F.lit(None))
    )

    return df

def process_silver_table_fin(bronze_fin_directory, silver_fin_directory, spark):
    # ensure output dir exists
    os.makedirs(silver_fin_directory, exist_ok=True)

    # find bronze FIN csvs
    bronze_files = [f for f in os.listdir(bronze_fin_directory) if f.lower().endswith(".csv")]
    if not bronze_files:
        print(f"[SILVER/FIN] No FIN CSVs found in {bronze_fin_directory}")
        return None

    bronze_files.sort()
    print(f"[SILVER/FIN] Found {len(bronze_files)} bronze FIN partitions...")

    for fname in bronze_files:
        # infer snapshot date from file name like *_YYYY_MM_DD.csv
        snapshot_date_str = _infer_snapshot_from_name(fname)
        if not snapshot_date_str:
            print(f"[SILVER/FIN][SKIP] Cannot infer snapshot from '{fname}'")
            continue

        fin_path = os.path.join(bronze_fin_directory, fname)
        df_fin = spark.read.csv(fin_path, header=True, inferSchema=True)

        # ---- numeric cleaning via your helper (underscore -> float + IQR -> None) ----
        # string numerics
        for c in [
            "Annual_Income",
            "Num_of_Loan",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Outstanding_Debt",
            "Amount_invested_monthly",
            "Monthly_Balance",
        ]:
            if c in df_fin.columns:
                df_fin = clean_numeric_column(df_fin, c)

        # numeric columns (already numeric, underscore step is a no-op; still IQR clean)
        for c in [
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Delay_from_due_date",
            "Num_Credit_Inquiries",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
        ]:
            if c in df_fin.columns:
                df_fin = clean_numeric_column(df_fin, c)

        # ---- Credit_History_Age -> total months (int) ----
        if "Credit_History_Age" in df_fin.columns:
            df_fin = df_fin.withColumn(
                "Credit_History_Age",
                F.regexp_replace(F.col("Credit_History_Age"), "_", "")  # remove underscores if any
            )
            df_fin = df_fin.withColumn(
                "Years",  F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast(IntegerType())
            ).withColumn(
                "Months", F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast(IntegerType())
            )
            df_fin = df_fin.withColumn(
                "Credit_History_Age_Months",
                (F.col("Years") * F.lit(12) + F.coalesce(F.col("Months"), F.lit(0))).cast(IntegerType())
            ).drop("Years", "Months", "Credit_History_Age") \
             .withColumnRenamed("Credit_History_Age_Months", "Credit_History_Age")

        # ---- Payment_of_Min_Amount normalization ----
        if "Payment_of_Min_Amount" in df_fin.columns:
            df_fin = df_fin.withColumn("Payment_of_Min_Amount", F.trim(F.lower(F.col("Payment_of_Min_Amount"))))
            df_fin = df_fin.withColumn(
                "Payment_of_Min_Amount",
                F.when(F.col("Payment_of_Min_Amount").isin("yes", "y"), "Yes")
                 .when(F.col("Payment_of_Min_Amount").isin("no", "n"), "No")
                 .when(F.col("Payment_of_Min_Amount").isin("nm", "not mentioned", "na", "none", ""), None)
                 .otherwise(F.col("Payment_of_Min_Amount"))
            )

        # ---- Payment_Behaviour: mark junk token as 'Unknown' (per your request) ----
        if "Payment_Behaviour" in df_fin.columns:
            df_fin = df_fin.withColumn(
                "Payment_Behaviour",
                F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown")
                 .otherwise(F.col("Payment_Behaviour"))
            )

        # ---- cast key types & snapshot ----
        if "snapshot_date" in df_fin.columns:
            df_fin = df_fin.withColumn("snapshot_date", F.col("snapshot_date").cast(DateType()))

        # ---- write one Silver parquet per snapshot ----
        out_name = f"silver_fin_{snapshot_date_str.replace('-', '_')}.parquet"
        out_path = os.path.join(silver_fin_directory, out_name)
        df_fin.write.mode("overwrite").parquet(out_path)

    print(f"[SILVER/FIN] Finished writing to {silver_fin_directory}")
    return True




    



