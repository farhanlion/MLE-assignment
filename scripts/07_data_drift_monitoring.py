# %%
import os
import glob
import numpy as np
import pyspark
import pyspark.sql.functions as F

# %%
# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# %%
# Load predictions
predictions_directory = "/app/datamart/gold/model_predictions/"
files_list = [predictions_directory+os.path.basename(f) for f in glob.glob(os.path.join(predictions_directory, '*'))]
df_predictions = spark.read.option("header", "true").parquet(*files_list)
print("predictions:")
df_predictions.show()
# Load Gold Features
gold_label_directory = "/app/datamart/gold/label_store/"
files_list = [gold_label_directory+os.path.basename(f) for f in glob.glob(os.path.join(gold_label_directory, '*'))]
label_df = spark.read.option("header", "true").parquet(*files_list)
print("True labels:")
label_df.show()
# Load Features
gold_feature_directory = "/app/datamart/gold/feature_store/"
df_features = spark.read.option("header", "true").parquet(gold_feature_directory)
print("Features:")
df_features.show()
# %%
label_df.groupBy("snapshot_date") \
        .agg(F.count("*").alias("count")) \
        .orderBy("snapshot_date") \
        .show(truncate=False)

# %%
merged_df = df_predictions.alias("p").join(
    df_features.alias("x"),
    on=["Customer_ID","snapshot_date"],
    how="inner"
).join(
    label_df.select("Customer_ID","snapshot_date","label").alias("y"),
    on=["Customer_ID","snapshot_date"],
    how="inner"   # labels may be missing
)

# %% [markdown]
# # Data Drift (feature distribution shift)
# 
# compare feature distributions now vs a baseline snapshot
# 
# Kolmogorov–Smirnov test per numeric feature
# 
# p < 0.05 → feature has drifted
# 
# KS stat closer to 1 → heavy drift

from scipy.stats import ks_2samp

baseline = merged_df.filter(F.col("snapshot_date") == "2024-11-01").toPandas()
current  = merged_df.filter(F.col("snapshot_date") == "2024-12-01").toPandas()



for col in ["Annual_Income","Num_of_Loan","Credit_History_Age","fe_1_5m_avg","fe_2_5m_avg","fe_3_5m_avg"]:
    stat, p = ks_2samp(baseline[col].dropna(), current[col].dropna())
    print(col, "KS stat:", stat, "p-value:", p)



# %% [markdown]
# # Concept Drift (model behavior or learned relationship changed)

# %%
df = merged_df.withColumn("prediction_error", F.abs(F.col("label") - F.col("model_predictions")))

# %%
import numpy as np

def psi(expected, actual, bins=10):
    e_perc, _ = np.histogram(expected, bins=bins)
    a_perc, _ = np.histogram(actual, bins=bins)
    e_perc = e_perc/len(expected)
    a_perc = a_perc/len(actual)
    psi_val = sum((e - a) * np.log(e/a) for e, a in zip(e_perc, a_perc) if e!=0 and a!=0)
    return psi_val

base = baseline["model_predictions"]
curr = current["model_predictions"]
print("PSI:", psi(base, curr))