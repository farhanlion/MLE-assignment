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
import sys

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
import data_processing_silver_table

# Initialize SparkSession (local dev)
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Reduce Spark logs to errors
spark.sparkContext.setLogLevel("ERROR")


# === Bronze layer directories ===
bronze_lms_directory = "/app/datamart/bronze/lms/"
bronze_clickstream_directory = "/app/datamart/bronze/clickstream/"
bronze_attr_directory = "/app/datamart/bronze/attributes/"
bronze_fin_directory = "/app/datamart/bronze/fin/"

# === Silver layer directories ===
silver_loan_daily_directory = "/app/datamart/silver/loan_daily/"
silver_fin_directory = "/app/datamart/silver/fin/"
silver_attr_directory = "/app/datamart/silver/attributes/"

# Ensure Silver dirs exist
for directory in [
    silver_loan_daily_directory,
    silver_fin_directory,
    silver_attr_directory
]:
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory (if missing): {directory}")

# Transform Bronze -> curated Silver tables
data_processing_silver_table.process_silver_table_lms(bronze_lms_directory, silver_loan_daily_directory, spark)
data_processing_silver_table.process_silver_table_attr(bronze_attr_directory, silver_attr_directory, spark)
data_processing_silver_table.process_silver_table_fin(bronze_fin_directory, silver_fin_directory, spark)
