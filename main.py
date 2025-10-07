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

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_gold_table
import importlib


# Initialize SparkSession (local dev)
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Reduce Spark logs to errors
spark.sparkContext.setLogLevel("ERROR")


# === Bronze layer directories ===
bronze_lms_directory = "datamart/bronze/lms/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"
bronze_attr_directory = "datamart/bronze/attributes/"
bronze_fin_directory = "datamart/bronze/fin/"

# Ensure Bronze dirs exist
for directory in [
    bronze_lms_directory,
    bronze_clickstream_directory,
    bronze_attr_directory,
    bronze_fin_directory
]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory (if missing): {directory}")


# Ingest raw CSVs -> Bronze tables (split by snapshot date)
utils.data_processing_bronze_table.process_bronze_table("data/lms_loan_daily.csv",bronze_lms_directory, spark)
utils.data_processing_bronze_table.process_bronze_table("data/feature_clickstream.csv",bronze_clickstream_directory, spark)
utils.data_processing_bronze_table.process_bronze_table("data/features_attributes.csv",bronze_attr_directory, spark)
utils.data_processing_bronze_table.process_bronze_table("data/features_financials.csv",bronze_fin_directory, spark)


# === Silver layer directories ===
silver_loan_daily_directory = "datamart/silver/loan_daily/"
silver_fin_directory = "datamart/silver/fin/"
silver_attr_directory = "datamart/silver/attributes/"

# Ensure Silver dirs exist
for directory in [
    silver_loan_daily_directory,
    silver_fin_directory,
    silver_attr_directory
]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory (if missing): {directory}")

# Transform Bronze -> curated Silver tables
utils.data_processing_silver_table.process_silver_table_lms(bronze_lms_directory, silver_loan_daily_directory, spark)
utils.data_processing_silver_table.process_silver_table_attr(bronze_attr_directory, silver_attr_directory, spark)
utils.data_processing_silver_table.process_silver_table_fin(bronze_fin_directory, silver_fin_directory, spark)


# === Gold layer: label store ===
gold_label_store_directory = "datamart/gold/label_store/"

# Ensure label store dir exists
if not os.path.exists(gold_label_store_directory):
    os.makedirs(gold_label_store_directory)

# Build supervised labels (e.g., 90+ DPD within 7 MOB)
utils.data_processing_gold_table.process_labels_gold_table(silver_loan_daily_directory, gold_label_store_directory, spark, dpd = 90, mob = 7)


# === Gold layer: feature store ===
gold_feature_store_directory = "datamart/gold/feature_store/"
silver_attr_directory = "datamart/silver/attributes/"
silver_fin_directory = "datamart/silver/fin/"
bronze_clickstream_directory = "datamart/bronze/clickstream/"

# NOTE: gold_label_feature_store_directory is referenced but not defined above.
# Ensure dir exists (as written; may raise if var undefined at runtime)
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# Build feature store (aggregations/joins across sources)
utils.data_processing_gold_table.process_features_gold_table(silver_attr_directory, silver_fin_directory, bronze_clickstream_directory, gold_feature_store_directory, spark)
