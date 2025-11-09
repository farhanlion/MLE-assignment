# %%
import os
import glob
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pprint
import pyspark

from datetime import datetime, timedelta
from pyspark.sql.functions import col
from sklearn.preprocessing import StandardScaler
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from data_processing_before_fit import process_features
# %%
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

#Pyspark remove warnings
spark.sparkContext.setLogLevel("ERROR")

# %%
# config
snapshot_date_str = "2024-11-01"
model_name = "xgb_credit_model_2024_09_01.pkl"

config = {}
config["snapshot_date_str"] = snapshot_date_str
config["snapshot_date"] = datetime.strptime(config["snapshot_date_str"], "%Y-%m-%d")
config["model_name"] = model_name
config["model_bank_directory"] = "/app/model_bank/"
config["model_artefact_filepath"] = config["model_bank_directory"] + config["model_name"]

pprint.pprint(config)

# %%
# Load model from pickle file
with open(config["model_artefact_filepath"], 'rb') as file:
    model_artefact = pickle.load(file)

print("Model loaded successfully! " + config["model_artefact_filepath"])

# %%

# Load Features
gold_feature_directory = "/app/datamart/gold/feature_store/"
df_features = spark.read.option("header", "true").parquet(gold_feature_directory)

# filter features for snapshot_date
features_sdf = df_features.filter((col("snapshot_date") == config["snapshot_date"]))
print("extracted features_sdf", features_sdf.count(), config["snapshot_date"])

# %%
# Preprocess features

features_sdf = features_sdf.dropna(how="any")


features_pdf = features_sdf.toPandas()
X_inference = process_features(features_pdf)\
    .drop(["Customer_ID", "snapshot_date"], axis=1)


# %%

# Inference

# load model
model = model_artefact["model"]

# predict with model
y_inference = model.predict_proba(X_inference)[:, 1]

# prepare output
y_inference_pdf = features_pdf[["Customer_ID","snapshot_date"]].copy()
y_inference_pdf["model_name"] = config["model_name"]
y_inference_pdf["model_predictions"] = y_inference

# %%
# save model inference to a gold table, model_predictions
gold_directory = f"/app/datamart/gold/model_predictions/"
print(gold_directory)
if not os.path.exists(gold_directory):
    os.makedirs(gold_directory)

# save gold table - IRL connect to database to write
partition_name = config["model_name"][:-4] + "_predictions_" + snapshot_date_str.replace('-','_') + '.parquet'
filepath = gold_directory + partition_name
spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
print('saved to:', filepath)


