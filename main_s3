#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
@author: tasneem-abbasi
@author: katherine_rein

Features: MONTH, DAY_OF_MONTH, DAY_OF_WEEK, OP_UNIQUE_CARRIER, ORIGIN_AIRPORT_ID,
DEST_AIRPORT_ID, AIR_TIME, DISTANCE

"""

# %% Prep Workspace

# Import Modules
from pyspark.sql import SparkSession
import sys
import time
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.feature import Tokenizer, CountVectorizer, IDF
from logistic_regression import create_log_reg_model, evaluate_log_reg_model, split_data
import data_vis
from data_cleaning import cleaning_flight_data, feature_eng
from evaluation import evaluate_predictions, confusion_matrix_counts, evaluate_baseline

# Create Spark Session
spark = SparkSession.builder \
    .appName("HW6") \
    .config("spark.network.timeout", "1000s") \
    .config("spark.executor.heartbeatInterval", "20s") \
    .config("spark.task.maxFailures", "8") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")
print('\n' * 5)

### SET WITH ACCURATE FILENAME FOR RUNNING ON YOUR OWN MACHINE
filename = "s3://777-termproject/csv/flights_2019_full.csv"

# %% Step 1: Prep data

start_time = time.perf_counter()
print('Starting Step 1: Prep Data')

#loading the dataset
df = spark.read.csv(filename, header= True, inferSchema=True)
#df.show(5)

#clean data
df = cleaning_flight_data(df)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Prepped Data: {elapsed:.4f} seconds')

# %% Step 2: Data Visualization

'''
Options:
    - Bar Chart: Delay Count by Month
    - Scatter Chart: Day of the Week vs Delay Count
    - Scatter Chart: Day of the Month vs Delay Count
    - Boxplot: Distance vs Delay
    - Boxplot: Air Time vs Delay
    - Bar Chart: Delay Count by Origin Airport/State
    - Bar Chart: Delay Count by Destination Airport/State
'''

start_time = time.perf_counter()
print('Starting Step 2: Data Visualization')


end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Data Visualization: {elapsed:.4f} seconds')

# %% Step 3: One Hot Encoding/Scaling

'''
Features to Encode:
    - OP_UNIQUE_CARRIER
    - ORIGIN_AIRPORT_ID
    - ORIGIN_STATE_ABR
    - DEST_AIRPORT_ID
    - DEST_STATE_ABR
'''

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

'''
Metrics:
    - F1 Score
    - AUC-ROC
'''

start_time = time.perf_counter()
print('Starting Step 5: Evaluate Model')

# Transform testing data
test_features = fitted_pipeline.transform(test_df)

# Get predictions
predictions = evaluate_log_reg_model(test_features, model)
metrics = evaluate_predictions(predictions)
cm = confusion_matrix_counts(predictions)

baseline_predictions, baseline_metrics, baseline_cm = evaluate_baseline(test_features)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Evaluate Model: {elapsed:.4f} seconds')
