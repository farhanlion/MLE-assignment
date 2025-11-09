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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
import data_processing_gold_table

# Initialize SparkSession (local dev)
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Reduce Spark logs to errors
spark.sparkContext.setLogLevel("ERROR")

# === Gold layer: feature store ===
gold_feature_store_directory = "/app/datamart/gold/feature_store/"
silver_attr_directory = "/app/datamart/silver/attributes/"
silver_fin_directory = "/app/datamart/silver/fin/"
bronze_clickstream_directory = "/app/datamart/bronze/clickstream/"

# NOTE: gold_label_feature_store_directory is referenced but not defined above.
# Ensure dir exists (as written; may raise if var undefined at runtime)
if not os.path.exists(gold_feature_store_directory):
    os.makedirs(gold_feature_store_directory)

# Build feature store (aggregations/joins across sources)
data_processing_gold_table.process_features_gold_table(silver_attr_directory, silver_fin_directory, bronze_clickstream_directory, gold_feature_store_directory, spark)
