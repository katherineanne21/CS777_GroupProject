#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import modules
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

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

def evaluate_log_reg_model(test_df, model):
    
    predictions = model.transform(test_df)

    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol = 'DEP_DEL15',
        predictionCol = 'prediction',
        metricName = 'f1',
        metricLabel = 1.0 # 1 = yes delayed
    )

    evaluator_roc = BinaryClassificationEvaluator(
        labelCol = 'DEP_DEL15', 
        rawPredictionCol = 'probability',
        metricName = 'areaUnderROC'
    )

    f1 = evaluator_f1.evaluate(predictions)
    auc_roc = evaluator_roc.evaluate(predictions)
    
    print(f'F1 score: {f1}')
    print(f'AUC_ROC: {auc_roc}')
    
    return predictions
    
    
    