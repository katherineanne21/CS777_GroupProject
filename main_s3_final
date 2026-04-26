#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess

# Install packages inside the job so we do NOT need bootstrap actions
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "xgboost",
    "scikit-learn"
])

from pyspark.sql import SparkSession
import time

from models import create_log_reg_model, split_data, xgboost
from data_cleaning import feature_eng, cleaning_flight_data
from evaluation import evaluate_predictions, confusion_matrix_counts, evaluate_baseline


spark = SparkSession.builder \
    .appName("TermProject") \
    .config("spark.network.timeout", "1000s") \
    .config("spark.executor.heartbeatInterval", "20s") \
    .config("spark.task.maxFailures", "8") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")
print("\n" * 5)


# Step 1: Prep data
start_time = time.perf_counter()
print("Starting Step 1: Prep Data")

cleaning_flight_data(spark)

filename = "s3://777-termproject/processed/cleaned_preprocessed_2019/"
df = spark.read.parquet(filename)

print("Columns in processed dataset:")
print(df.columns)

end_time = time.perf_counter()
print(f"Prepped Data: {end_time - start_time:.4f} seconds")


# Step 2: Data Visualization placeholder
start_time = time.perf_counter()
print("Starting Step 2: Data Visualization")

end_time = time.perf_counter()
print(f"Data Visualization: {end_time - start_time:.4f} seconds")


# Step 3: One Hot Encoding/Scaling
start_time = time.perf_counter()
print("Starting Step 3: One Hot Encoding and Scaling")

train_df, test_df, val_df = split_data(df)

feature_pipeline = feature_eng()
fitted_pipeline = feature_pipeline.fit(train_df)

train_features = fitted_pipeline.transform(train_df)
test_features = fitted_pipeline.transform(test_df)
val_features = fitted_pipeline.transform(val_df)

end_time = time.perf_counter()
print(f"One Hot Encoding and Scaling: {end_time - start_time:.4f} seconds")


# Step 4: Logistic Regression
start_time = time.perf_counter()
print("Starting Step 4: Logistic Regression Model")

log_reg_model = create_log_reg_model(train_features)

end_time = time.perf_counter()
print(f"Logistic Regression Model: {end_time - start_time:.4f} seconds")


# Step 5: Evaluate Logistic Regression
start_time = time.perf_counter()
print("Starting Step 5: Evaluate Logistic Regression Model")

test_predictions = log_reg_model.transform(test_features)

print("Logistic Regression Test Metrics:")
evaluate_predictions(test_predictions)
confusion_matrix_counts(test_predictions)
evaluate_baseline(test_features)

val_predictions = log_reg_model.transform(val_features)

print("Logistic Regression Validation Metrics:")
evaluate_predictions(val_predictions)
confusion_matrix_counts(val_predictions)

end_time = time.perf_counter()
print(f"Evaluate Logistic Regression Model: {end_time - start_time:.4f} seconds")


# Step 6: K-Fold Cross Validation
print("Skipping K-Fold Cross Validation for this run.")


# Step 7: XGBoost
start_time = time.perf_counter()
print("Starting Step 7: XGBoost")

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Smaller sample to avoid EMR memory issues
xgb_train = train_features.sample(withReplacement=False, fraction=0.02, seed=42).limit(50000)
xgb_test = test_features.sample(withReplacement=False, fraction=0.02, seed=42).limit(20000)
xgb_val = val_features.sample(withReplacement=False, fraction=0.02, seed=42).limit(20000)

print("Training XGBoost on sampled data:")
print(f"XGBoost train count: {xgb_train.count()}")
print(f"XGBoost test count: {xgb_test.count()}")
print(f"XGBoost validation count: {xgb_val.count()}")

xgb_model = xgboost(xgb_train)

test_pd = xgb_test.select("features", "DEP_DEL15").toPandas()
X_test = np.array([x.toArray() for x in test_pd["features"]])
y_test = test_pd["DEP_DEL15"].values

test_preds = xgb_model.predict(X_test)

print("XGBoost Test Metrics:")
print("Accuracy:", accuracy_score(y_test, test_preds))
print("Precision:", precision_score(y_test, test_preds))
print("Recall:", recall_score(y_test, test_preds))
print("F1:", f1_score(y_test, test_preds))

val_pd = xgb_val.select("features", "DEP_DEL15").toPandas()
X_val = np.array([x.toArray() for x in val_pd["features"]])
y_val = val_pd["DEP_DEL15"].values

val_preds = xgb_model.predict(X_val)

print("XGBoost Validation Metrics:")
print("Accuracy:", accuracy_score(y_val, val_preds))
print("Precision:", precision_score(y_val, val_preds))
print("Recall:", recall_score(y_val, val_preds))
print("F1:", f1_score(y_val, val_preds))

end_time = time.perf_counter()
print(f"XGBoost: {end_time - start_time:.4f} seconds")


spark.stop()
