#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: 
@author: 
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
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

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


# %% Step 1: Prep data

start_time = time.perf_counter()
print('Starting Step 1: Prep Data')

'''
Substeps:
    - Read in data
    - Remove year column
    - Clean data
    - Split data
'''

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Prepped Data: {elapsed:.4f} seconds')

# %% Step 2: Data Visualization

start_time = time.perf_counter()
print('Starting Step 2: Data Visualization')

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

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Data Visualization: {elapsed:.4f} seconds')

# %% Step 3: One Hot Encoding

start_time = time.perf_counter()
print('Starting Step 3: One Hot Encoding')

'''
Features to Encode:
    - OP_UNIQUE_CARRIER
    - ORIGIN_AIRPORT_ID
    - ORIGIN_STATE_ABR
    - DEST_AIRPORT_ID
    - DEST_STATE_ABR
'''

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'One Hot Encoding: {elapsed:.4f} seconds')

# %% Step 4: Logistic Regression Model

start_time = time.perf_counter()
print('Starting Step 4: Logistic Regression Model')

# Select Features


# Create model
lr = LogisticRegression(
    featuresCol = 'features',
    labelCol = 'label',
    regParam = 0.5,
    elasticNetParam = 0.0,
    maxIter = 100,
    tol = 1e-6
)

model = lr.fit(train_df)

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Logistic Regression Model: {elapsed:.4f} seconds')

# %% Step 5: Evaluate Model

start_time = time.perf_counter()
print('Starting Step 5: Evaluate Model')

predictions = model.transform(test_df)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol = 'label',
    predictionCol = 'prediction',
    metricName = 'f1',
    metricLabel = 1.0
)

evaluator_roc = BinaryClassificationEvaluator(
    labelCol = 'label', 
    rawPredictionCol = 'probability',
    metricName = 'areaUnderROC'
)

f1 = evaluator_f1.evaluate(predictions)
auc_roc = evaluator_roc.evaluate(predictions)

print(f'F1 score: {f1}')
print(f'AUC_ROC: {auc_roc}')

'''
Metrics:
    - F1 Score
    - AUC-ROC
'''

end_time = time.perf_counter()
elapsed = end_time - start_time
print(f'Evaluate Model: {elapsed:.4f} seconds')
