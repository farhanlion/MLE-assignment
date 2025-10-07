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

_DATE_IN_NAME = re.compile(r".*?(\d{4})_(\d{2})_(\d{2})\.(?:parquet|csv)$", re.IGNORECASE)

def _infer_snapshot_from_name(filename: str):
    m = _DATE_IN_NAME.match(filename)
    if not m:
        return None
    y, mth, d = m.groups()
    return f"{y}-{mth}-{d}"


def process_labels_gold_table(silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    files = [f for f in os.listdir(silver_loan_daily_directory) if f.lower().endswith(".parquet")]
    if not files:
        print(f"[GOLD/LABEL] No parquet files found in {silver_loan_daily_directory}")
        return None

    files.sort()
    print(f"[GOLD/LABEL] Found {len(files)} silver LMS partitions...")

    for fname in files:
        snapshot_date_str = _infer_snapshot_from_name(fname)
        if not snapshot_date_str:
            print(f"[GOLD/LABEL][SKIP] Cannot infer snapshot from '{fname}'")
            continue
    
       
        in_path = os.path.join(silver_loan_daily_directory, fname)
        df = spark.read.parquet(in_path)
    
        # get customer at mob
        df = df.filter(col("mob") == mob)
    
        # get label
        df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
        df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))
    
        # select columns to save
        df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")
    
        # save gold table - IRL connect to database to write
        partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
        filepath = gold_label_store_directory + partition_name
        df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
    
    print(f"[GOLD/LABEL] Finished writing to {gold_label_store_directory}")
    return True

def process_features_gold_table(silver_attr_directory, silver_fin_directory, bronze_clickstream_directory, gold_feature_store_directory, spark):
        parquet_files = glob.glob(os.path.join(silver_attr_directory, "*.parquet"))
        df_attr = spark.read.parquet(*parquet_files)
        df_attr = df_attr.drop("Age", "Name", "SSN")
    
        
        parquet_files = glob.glob(os.path.join(silver_fin_directory, "*.parquet"))
        df_fin = spark.read.parquet(*parquet_files)
        df_fin = df_fin.drop("Type_of_Loan")
        # Calculate null percentages
        rows = df_fin.count()
        nulls = df_fin.select([
            (F.count(F.when(F.col(c).isNull(), c)) / rows * 100).alias(c)
            for c in df_fin.columns
        ])
        
        # Collect null percentages into a dict
        nulls_dict = nulls.collect()[0].asDict()
        
        # Get columns to drop
        cols_to_drop = [col for col, pct in nulls_dict.items() if pct > 5]
        
        # Drop columns from DataFrame
        df_fin = df_fin.drop(*cols_to_drop)
        print("Dropped columns:", cols_to_drop)
    
        # Drop the rest of the NAs
        df_fin = df_fin.dropna()    
        
        
        df_click = spark.read.csv(bronze_clickstream_directory, header=True, inferSchema=True)
    
        # 1️⃣ Join finance + attribute (both share the same snapshot_date)
        df_fin_attr = df_fin.join(df_attr, ["Customer_ID", "snapshot_date"], "inner")
        
        # 2️⃣ Drop their snapshot_date — we’ll use the one from clickstream
        df_fin_attr = df_fin_attr.drop("snapshot_date")
        
        # 3️⃣ Join with clickstream (keeping all customers from df_click)
        merged_df = df_fin_attr.join(df_click, ["Customer_ID"], "right")
    
        # --- Write by snapshot_date ---
        snapshot_dates = [r["snapshot_date"] for r in merged_df.select("snapshot_date").distinct().collect()]
        print(f"[GOLD/FE] Found {len(snapshot_dates)} snapshot dates to save.")
    
        for s_date in snapshot_dates:
            s_date_str = s_date.strftime("%Y_%m_%d")
            out_path = os.path.join(gold_feature_store_directory, f"gold_feature_store_{s_date_str}.parquet")
            print(f"[GOLD/FE] Writing snapshot {s_date_str} to {out_path}")
    
            df_snapshot = merged_df.filter(F.col("snapshot_date") == s_date)
            df_snapshot.write.mode("overwrite").parquet(out_path)
    
        print(f"[GOLD/FE] Finished writing feature gold tables to {gold_feature_store_directory}")
        return merged_df
        

    
    