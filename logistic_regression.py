#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import modules
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def create_log_reg_model(train_df):
    
    lr = LogisticRegression(
        featuresCol = 'features',
        labelCol = 'label',
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
    
    return predictions
    
    
    