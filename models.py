#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import modules
from pyspark.ml.classification import LogisticRegression

def split_data(df):
    #splitting the data (Train,Test , Validation)
    #It randomly splits the dataset into 70% training and 30% temporary data,
    #then further splits that 30% into 15% validation and 15% test data for model tuning and evaluation
    train_data, temp_data = df.randomSplit([0.7, 0.3], seed=42)
    val_data, test_data = temp_data.randomSplit([0.5, 0.5], seed=42)
    return train_data, test_data, val_data

def create_log_reg_model(train_df):
    
    lr = LogisticRegression(
        featuresCol = 'features',
        labelCol = 'DEP_DEL15',
        regParam = 0.5,
        elasticNetParam = 0.0,
        maxIter = 100,
        tol = 1e-6
    )

    model = lr.fit(train_df)
    
    return model
    
    
    