#!/usr/bin/env python
# coding: utf-8

# In[90]:


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


# In[91]:


# Initialize SparkSession
spark = pyspark.sql.SparkSession.builder \
    .appName("dev") \
    .master("local[*]") \
    .getOrCreate()

# Set log level to ERROR to hide warnings
spark.sparkContext.setLogLevel("ERROR")


# In[92]:


# create bronze datalake
bronze_lms_directory = "datamart/bronze/lms/"

if not os.path.exists(bronze_lms_directory):
    os.makedirs(bronze_lms_directory)


# In[145]:


# ----------------------------
# 1️⃣ Read lms_loan_daily.csv
# ----------------------------
lms_path = "data/lms_loan_daily.csv"
df_lms = spark.read.csv(lms_path, header=True, inferSchema=True)
print("✅ LMS Loan Daily Data:")
df_lms.show(5, truncate=False)
df_lms.printSchema()

# ----------------------------
# 2️⃣ Read features_attributes.csv
# ----------------------------
attr_path = "data/features_attributes.csv"
df_attr = spark.read.csv(attr_path, header=True, inferSchema=True)
print("\n✅ Customer Attributes Data:")
df_attr.show(5, truncate=False)
df_attr.printSchema()

# ----------------------------
# 3️⃣ Read features_financials.csv
# ----------------------------
fin_path = "data/features_financials.csv"
df_fin = spark.read.csv(fin_path, header=True, inferSchema=True)
print("\n✅ Financial Features Data:")
df_fin.show(5, truncate=False)
df_fin.printSchema()

# ----------------------------
# 4️⃣ Read feature_clickstream.csv
# ----------------------------
click_path = "data/feature_clickstream.csv"
df_click = spark.read.csv(click_path, header=True, inferSchema=True)
print("\n✅ Clickstream Features Data:")
df_click.show(5, truncate=False)
df_click.printSchema()

# ----------------------------
# Optional: Count rows for a quick sanity check
# ----------------------------
print(f"""
Row Counts:
  LMS: {df_lms.count()}
  ATTR: {df_attr.count()}
  FIN: {df_fin.count()}
  CLICK: {df_click.count()}
""")



# In[157]:


df_lms.count()


# In[158]:


from pyspark.sql import functions as F

df_lms.select(
    F.min("snapshot_date").alias("min_snapshot_date"),
    F.max("snapshot_date").alias("max_snapshot_date"),
    F.min("loan_start_date").alias("min_loan_start_date"),
    F.max("loan_start_date").alias("max_loan_start_date")
).show()


# In[159]:


df_lms_sorted_desc = df_lms.orderBy(F.col("snapshot_date").desc())
df_lms_sorted_desc.show(10, truncate=False)


# In[160]:


from pyspark.sql import functions as F

df_lms.select("snapshot_date") \
    .distinct() \
    .orderBy("snapshot_date") \
    .show(1000, truncate=False)


# In[161]:


dates_str_lst = (
    df_lms.select("snapshot_date")
    .distinct()
    .orderBy("snapshot_date")
    .rdd.flatMap(lambda x: x)
    .collect()
)

dates_str_lst


# In[ ]:





# In[162]:


snapshot_dates = (
        df_lms
        .select(F.col("snapshot_date").cast("string"))
        .distinct()
        .orderBy("snapshot_date")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
snapshot_dates 


# In[163]:


snapshot_dates = (
        df_attr
        .select(F.col("snapshot_date").cast("string"))
        .distinct()
        .orderBy("snapshot_date")
        .rdd.flatMap(lambda x: x)
        .collect()
    )
snapshot_dates 


# In[164]:


df_attr_202302 = df_attr.filter(F.col("snapshot_date") == "2023-02-01")
df_attr_202302.show(10, truncate=False)


# In[165]:


df_attr.show()


# In[166]:


df_attr.filter(F.col("SSN") == "#F%$D@*&8").count()


# In[167]:


df_attr = df_attr.withColumn(
    "SSN",
    F.when(F.col("SSN") == "#F%$D@*&8", "000-00-0000")
     .otherwise(F.col("SSN"))
)


# In[168]:


df_attr.show()


# In[169]:


df_attr.select("Occupation") \
    .distinct() \
    .orderBy("Occupation") \
    .show(100, truncate=False)


# In[170]:


df_attr.filter(F.col("Occupation") == "_______").count()


# In[171]:


df_attr = df_attr.withColumn(
    "Occupation",
    F.when(F.col("Occupation") == "_______", "Unemployed")
     .otherwise(F.col("Occupation"))
)


# In[172]:


df_attr.select("Occupation") \
    .distinct() \
    .orderBy("Occupation") \
    .show(100, truncate=False)


# In[173]:


df_attr.printSchema()


# In[174]:


column_type_map = {
    "Customer_ID": StringType(),
    "Name": StringType(),
    "Age": IntegerType(),
    "SSN": StringType(),
    "Occupation": StringType(),
    "snapshot_date": DateType(),
}


# In[175]:


df_attr.filter(~F.col("Age").rlike("^[0-9]+$")).select("Customer_ID", "Name", "Age").show(50, truncate=False)


# In[176]:


df_attr.filter((F.col("Age") < 0) | (F.col("Age") > 100)) \
    .select("Age") \
    .distinct() \
    .orderBy("Age") \
    .show()


# In[177]:


df_attr = df_attr.withColumn(
    "Age",
    F.when((F.col("Age") >= 0) & (F.col("Age") <= 100), F.col("Age"))
     .otherwise(None)
)


# In[178]:


df_attr.filter((F.col("Age") < 0) | (F.col("Age") > 100)) \
    .select("Age") \
    .distinct() \
    .orderBy("Age") \
    .show()


# In[179]:


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


# In[180]:


df_attr.dtypes


# # DF FIN

# In[242]:


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

    print(f"[CLEAN] Column '{column_name}' cleaned. IQR bounds: ({lower:.2f}, {upper:.2f})")

    return df


# In[236]:


df_fin = spark.read.csv(fin_path, header=True, inferSchema=True)


# In[243]:


df_fin.dtypes


# In[244]:


df_fin = clean_numeric_column(df_fin, "Annual_Income")
df_fin = clean_numeric_column(df_fin, "Monthly_Inhand_Salary")
df_fin = clean_numeric_column(df_fin, "Num_Bank_Accounts")
df_fin = clean_numeric_column(df_fin, "Num_Credit_Card")
df_fin = clean_numeric_column(df_fin, "Interest_Rate")
df_fin = clean_numeric_column(df_fin, "Num_of_Loan")
df_fin = clean_numeric_column(df_fin, "Delay_from_due_date")
df_fin = clean_numeric_column(df_fin, "Num_of_Delayed_Payment")
df_fin = clean_numeric_column(df_fin, "Changed_Credit_Limit")
df_fin = clean_numeric_column(df_fin, "Num_Credit_Inquiries")
df_fin = clean_numeric_column(df_fin, "Outstanding_Debt")
df_fin = clean_numeric_column(df_fin, "Credit_Utilization_Ratio")
df_fin = clean_numeric_column(df_fin, "Total_EMI_per_month")
df_fin = clean_numeric_column(df_fin, "Amount_invested_monthly")
df_fin = clean_numeric_column(df_fin, "Monthly_Balance")



# In[245]:


df_fin = df_fin.withColumn(
    "Credit_History_Age",
    F.regexp_replace(F.col("Credit_History_Age"), "_", "")  # remove underscores if any
)

df_fin = df_fin.withColumn(
    "Years",
    F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Years", 1).cast("int")
).withColumn(
    "Months",
    F.regexp_extract(F.col("Credit_History_Age"), r"(\d+)\s+Months", 1).cast("int")
)

df_fin = df_fin.withColumn(
    "Credit_History_Age_Months",
    (F.col("Years") * 12 + F.col("Months"))
)

# Drop the old columns if you want
df_fin = df_fin.drop("Years", "Months", "Credit_History_Age")

# (Optional) rename for simplicity
df_fin = df_fin.withColumnRenamed("Credit_History_Age_Months", "Credit_History_Age")

df_fin.select("Customer_ID", "Credit_History_Age").show(10, truncate=False)


# In[246]:


df_fin = df_fin.withColumn(
    "Payment_of_Min_Amount",
    F.trim(F.lower(F.col("Payment_of_Min_Amount")))  # normalize case
)

df_fin = df_fin.withColumn(
    "Payment_of_Min_Amount",
    F.when(F.col("Payment_of_Min_Amount").isin("yes", "y"), "Yes")
     .when(F.col("Payment_of_Min_Amount").isin("no", "n"), "No")
     .when(F.col("Payment_of_Min_Amount").isin("nm", "not mentioned", "na", "none"), None)
     .otherwise(F.col("Payment_of_Min_Amount"))  # keep valid ones
)

df_fin.select("Payment_of_Min_Amount").distinct().show()


# In[248]:


df_fin = df_fin.withColumn(
    "Payment_Behaviour",
    F.when(F.col("Payment_Behaviour") == "!@9#%8", "Unknown")
     .otherwise(F.col("Payment_Behaviour"))
)


# In[222]:


df_fin.filter(F.col("Num_Credit_Card") > 10).select().show()


# In[223]:


df_fin = df_fin.withColumn(
    "Num_Bank_Accounts",
    F.when(F.col("Num_Bank_Accounts") > 10, 15)
     .otherwise(F.col("Num_Bank_Accounts"))
)


# In[224]:


df_fin = df_fin.withColumn(
    "Num_Credit_Card",
    F.when(F.col("Num_Credit_Card") > 10, 10)
     .otherwise(F.col("Num_Credit_Card"))
)


# In[225]:


df_fin.groupBy("Num_Credit_Card") \
      .count() \
      .orderBy("Num_Credit_Card") \
      .show(100)


# In[226]:


q1, q3 = df_fin.approxQuantile("Interest_Rate", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df_outliers = df_fin.filter((F.col("Interest_Rate") < lower) | (F.col("Interest_Rate") > upper))
print(df_outliers.select("Interest_Rate").show(100))

df_inliers = df_fin.filter((F.col("Interest_Rate") > lower) & (F.col("Interest_Rate") < upper))
print(df_inliers.select("Interest_Rate").show(100))


# In[227]:


upper


# In[228]:


df_fin.select("Interest_Rate").describe().show()


# In[229]:


q1, q3 = df_fin.approxQuantile("Interest_Rate", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df_outliers = df_fin.filter((F.col("Interest_Rate") < lower) | (F.col("Interest_Rate") > upper))
print(df_outliers.select("Interest_Rate").show(100))

df_inliers = df_fin.filter((F.col("Interest_Rate") > lower) & (F.col("Interest_Rate") < upper))
print(df_inliers.select("Interest_Rate").show(100))


# In[230]:


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

    print(f"[CLEAN] Column '{column_name}' cleaned. IQR bounds: ({lower:.2f}, {upper:.2f})")

    return df


# In[231]:


df_fin = clean_numeric_column(df_fin, "Interest_Rate")


# In[232]:


df_fin = clean_numeric_column(df_fin, "Num_of_Loan")


# In[ ]:





# In[233]:


df_fin.select("Num_of_Loan") \
      .distinct() \
      .orderBy("Num_of_Loan") \
      .show(100)


# In[234]:


numeric_cols = [field.name for field in df_fin.schema.fields 
                if field.dataType.typeName() in ["integer", "double", "float", "long"]]

df_fin.select(numeric_cols).describe().show()


# ## Load all silver parquet files

# In[200]:


parquet_files = glob.glob(os.path.join("datamart/silver/attributes", "*.parquet"))

df_attr = spark.read.parquet(*parquet_files)
df_attr


# In[201]:


df_attr.count()


# In[202]:


# check for null values

df_attr.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c) 
    for c in df_attr.columns
]).show()


# drop age, name, ssn,
df_attr = df_attr.drop("Age", "Name", "SSN")  # drop columns with too many nulls
df_attr.show()


# In[203]:


parquet_files = glob.glob(os.path.join("datamart/silver/fin", "*.parquet"))

df_fin = spark.read.parquet(*parquet_files)
pd_df = df_fin.limit(5).toPandas()
pd_df



# In[204]:


# remove type_of_loan

df_fin = df_fin.drop("Type_of_Loan")  # drop useless col

df_fin.count()


# In[205]:


null_counts = df_fin.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in df_fin.columns
]).toPandas()

print(null_counts.T)  


# In[206]:


rows = df_fin.count()
nulls = df_fin.select([
    (F.count(F.when(F.col(c).isNull(), c)) / rows * 100).alias(c)
    for c in df_fin.columns
])

# Convert to Pandas and transpose for readability
nulls_pdf = nulls.toPandas().T
nulls_pdf.columns = ['% Null']
print(nulls_pdf)



# In[207]:


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


# In[208]:


df_fin = df_fin.dropna()


# In[209]:


rows = df_fin.count()
nulls = df_fin.select([
    (F.count(F.when(F.col(c).isNull(), c)) / rows * 100).alias(c)
    for c in df_fin.columns
])

# Convert to Pandas and transpose for readability
nulls_pdf = nulls.toPandas().T
nulls_pdf.columns = ['% Null']
print(nulls_pdf)


# In[210]:


clickstream_directory = "datamart/bronze/clickstream/"
df_click = spark.read.csv(clickstream_directory, header=True, inferSchema=True)
df_click.count()


# In[211]:


# Count total rows
rows = df_click.count()

# Calculate % of nulls per column
nulls = df_click.select([
    (F.count(F.when(F.col(c).isNull(), c)) / rows * 100).alias(c)
    for c in df_click.columns
])

# Display neatly
nulls_df = nulls.toPandas().T
nulls_df.columns = ['% Null']
print(nulls_df)


# In[212]:


merged_df = (
    df_fin
    .join(df_attr, ["Customer_ID", "snapshot_date"], "inner")
    .join(df_click, ["Customer_ID", "snapshot_date"], "inner")
)



# In[213]:


merged_df


# In[214]:


#read all files from gold feature store and label store
# directories
gold_feature_store_directory = "datamart/gold/feature_store/"
gold_label_store_directory = "datamart/gold/label_store/"

# read all feature files
feature_files = glob.glob(os.path.join(gold_feature_store_directory, "*.parquet"))
df_features = spark.read.parquet(*feature_files)

# read all label files
label_files = glob.glob(os.path.join(gold_label_store_directory, "*.parquet"))
df_label = spark.read.parquet(*label_files)

print("Features rows:", df_features.count())
print("Labels rows:", df_label.count())


# In[215]:


df_label.show()


# In[216]:


# Count distinct vs total
total_count = df_label.count()
unique_count = df_label.select("Customer_ID").distinct().count()

if total_count == unique_count:
    print("✅ Customer_ID is unique in df_label.")
else:
    print(f"⚠️ Customer_ID is NOT unique. Total rows: {total_count}, Unique IDs: {unique_count}")


# In[217]:


# Check uniqueness
total_features = df_features.count()
unique_features = df_features.select("Customer_ID").distinct().count()

if total_features == unique_features:
    print("✅ Customer_ID is unique in df_features.")
else:
    print(f"⚠️ Customer_ID is NOT unique. Total rows: {total_features}, Unique IDs: {unique_features}")


# In[218]:


df_label.filter(F.col("Customer_ID") == "CUS_0x1690").show(truncate=False)


# In[219]:


df_features.filter(F.col("Customer_ID") == "CUS_0x1690").show(truncate=False)


# In[220]:


df_joined = (
    df_features.alias("f")
    .join(
        df_label.alias("y"),
        on=F.col("f.Customer_ID") == F.col("y.Customer_ID"),
        how="inner"
    )
    .filter(F.col("f.snapshot_date") < F.col("y.snapshot_date"))
)


# In[221]:


df_joined.show(5, truncate=False)


# In[222]:


df_joined.filter(F.col("f.Customer_ID") == "CUS_0x1690").show(truncate=False)


# In[223]:


clean = (
    df_joined
    .select(
        # features (prefix f_)
        *[F.col(f"f.{c}").alias(c if c not in ["Customer_ID","snapshot_date"] else f"f_{c}")
          for c in df_features.columns],

        # labels (prefix y_)
        F.col("y.loan_id"),
        F.col("y.Customer_ID").alias("y_Customer_ID"),
        F.col("y.label"),
        F.col("y.label_def"),
        F.col("y.snapshot_date").alias("label_snapshot_date"),
    )
)

clean.show(truncate=False)


# In[224]:


# 1) Collect fe_* columns dynamically from `clean`
fe_cols = [c for c in clean.columns if c.startswith("fe_")]

# 2) Keep rows in the 3-month lookback window: [label-3M, label-1 day]
df_3m = clean.where(
    (F.col("f_snapshot_date") >= F.add_months(F.col("label_snapshot_date"), -3)) &
    (F.col("f_snapshot_date") <  F.col("label_snapshot_date"))
)

# 1) Collect fe_* columns dynamically
fe_cols = [c for c in clean.columns if c.startswith("fe_")]

# 2) Filter rows to last 3 months before label snapshot
df_3m = clean.filter(
    (F.col("f_snapshot_date") >= F.add_months(F.col("label_snapshot_date"), -3)) &
    (F.col("f_snapshot_date") <  F.col("label_snapshot_date"))
)

# 3) Compute mean for each fe_* column per loan
agg_exprs = [F.mean(c).alias(f"{c}_mean_3m") for c in fe_cols]

df_mean = (
    df_3m.groupBy("loan_id", "y_Customer_ID", "label", "label_def", "label_snapshot_date")
         .agg(*agg_exprs)
)

# 4) Show result
df_mean.show(truncate=False)


# In[225]:


from pyspark.sql import functions as F

m = (df_mean
     .withColumnRenamed("y_Customer_ID", "Customer_ID")
     .withColumnRenamed("label_snapshot_date", "snapshot_date"))

final_features_df = (
    df_fin
    .join(df_attr, ["Customer_ID", "snapshot_date"], "inner")
    .join(m, ["Customer_ID"], "inner")
)


# In[226]:


final_features_df.show()


# In[230]:


final_features_df = final_features_df.drop("Customer_ID", "snapshot_date", "loan_id", "label_def")


# In[231]:


final_features_df.show()


# In[232]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Define categorical columns
categorical_cols = ["Credit_Mix", "Payment_Behaviour", "Occupation"]

# String indexers
indexers = [StringIndexer(inputCol=col, outputCol=col + "_index") for col in categorical_cols]

# One-hot encoder
encoder = OneHotEncoder(
    inputCols=[col + "_index" for col in categorical_cols],
    outputCols=[col + "_encoded" for col in categorical_cols]
)

# Build pipeline
pipeline = Pipeline(stages=indexers + [encoder])
encoded_df = pipeline.fit(final_features_df).transform(final_features_df)

# Drop the original categorical columns if you want
encoded_df = encoded_df.drop(*categorical_cols)


# In[234]:


# 1️⃣ Identify categorical encoded and numeric columns
categorical_encoded = ["Credit_Mix_encoded", "Payment_Behaviour_encoded", "Occupation_encoded"]

numeric_cols = [
    "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card", 
    "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment", 
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Credit_Utilization_Ratio", 
    "Credit_History_Age"
] + [f"fe_{i}_mean_3m" for i in range(1, 21)]

# 2️⃣ Assemble numeric columns
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")

# 3️⃣ Standardize numeric features
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_features")

# 4️⃣ Pipeline
pipeline = Pipeline(stages=[assembler, scaler])
scaled_df = pipeline.fit(encoded_df).transform(encoded_df)


# In[235]:


scaled_df.show()


# In[238]:


# === 0) Imports
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# === 1) Combine scaled numerics + encoded categoricals into one features vector
# Make sure these columns exist in your scaled_df from your previous pipeline:
# - "scaled_features" (StandardScaler output)
# - "Credit_Mix_encoded", "Payment_Behaviour_encoded", "Occupation_encoded" (OneHotEncoder outputs)
assembler_final = VectorAssembler(
    inputCols=[
        "scaled_features",
        "Credit_Mix_encoded",
        "Payment_Behaviour_encoded",
        "Occupation_encoded",
    ],
    outputCol="features",
    handleInvalid="keep"
)

final_df = assembler_final.transform(scaled_df).select("features", "label")

# === 2) Train/Test split
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# === 3) Train models
# 3a) Logistic Regression
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
lr_model = lr.fit(train_df)
lr_pred = lr_model.transform(test_df)

# 3b) Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=200, maxDepth=8, seed=42)
rf_model = rf.fit(train_df)
rf_pred = rf_model.transform(test_df)

# === 4) Evaluate
bin_eval_roc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
bin_eval_pr  = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")
f1_eval      = MulticlassClassificationEvaluator(labelCol="label", metricName="f1")

def evaluate(name, pred):
    print(f"\n=== {name} ===")
    print("AUC-ROC:", bin_eval_roc.evaluate(pred))
    print("AUC-PR :", bin_eval_pr.evaluate(pred))
    print("F1     :", f1_eval.evaluate(pred))
    print("Confusion matrix:")
    (pred
     .groupBy("label", "prediction")
     .count()
     .orderBy("label", "prediction")
     .show())

evaluate("Logistic Regression", lr_pred)
evaluate("Random Forest", rf_pred)

# === 5) (Optional) Save the better model
# lr_model.save("/path/to/save/lr_model")
# rf_model.save("/path/to/save/rf_model")


# In[ ]:




