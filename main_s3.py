#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
import time
from models import create_log_reg_model, split_data
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
print('\n' * 5)

# %% Step 1: Prep data
start_time = time.perf_counter()
print('Starting Step 1: Prep Data')

# Clean Data
cleaning_flight_data(spark)

# Read Cleaned Data
filename = "s3://777-termproject/processed/cleaned_preprocessed_2019/"
df = spark.read.parquet(filename)

print("Columns in processed dataset:")
print(df.columns)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Prepped Data: {elapsed:.4f} seconds')

# %% Step 2: Data Visualization
start_time = time.perf_counter()
print('Starting Step 2: Data Visualization')

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Data Visualization: {elapsed:.4f} seconds')

# %% Step 3: One Hot Encoding/Scaling
start_time = time.perf_counter()
print('Starting Step 3: One Hot Encoding and Scaling')

# Split data up
train_df, test_df, val_df = split_data(df)

# Clean and scale data
feature_pipeline = feature_eng()
fitted_pipeline = feature_pipeline.fit(train_df)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'One Hot Encoding and Scaling: {elapsed:.4f} seconds')

# %% Step 4: Logistic Regression Model
start_time = time.perf_counter()
print('Starting Step 4: Logistic Regression Model')

# Transform features
train_features = fitted_pipeline.transform(train_df)

# Create model
model = create_log_reg_model(train_features)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Logistic Regression Model: {elapsed:.4f} seconds')

# %% Step 5: Evaluate Model
start_time = time.perf_counter()
print('Starting Step 5: Evaluate Model')

# Transform testing data
test_features = fitted_pipeline.transform(test_df)

# Get predictions
predictions = model.transform(test_df)

# Print Metrics
evaluate_predictions(predictions)
confusion_matrix_counts(predictions)
evaluate_baseline(test_features)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Evaluate Model: {elapsed:.4f} seconds')

spark.stop()
