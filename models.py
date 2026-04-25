#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import modules
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from xgboost.spark import SparkXGBRegressor
from pyspark.sql.functions import col

def split_data(df):
    #splitting the data (Train,Test , Validation)
    #It randomly splits the dataset into 70% training and 30% temporary data,
    #then further splits that 30% into 15% validation and 15% test data for model tuning and evaluation
    train_data, temp_data = df.randomSplit([0.7, 0.3], seed=42)
    val_data, test_data = temp_data.randomSplit([0.5, 0.5], seed=42)
    
    # Undersample train falses and over sample train trues
    true_train = train_data.filter(col("label") == True)
    false_train = train_data.filter(col("label") == False)

    true_balanced = true_train.sample(withReplacement = True, fraction = 3.0, seed = 42)
    false_balanced = false_train.sample(withReplacement=False, fraction = 0.3, seed = 42)

    balanced_train = true_balanced.unionByName(false_balanced)
    
    # Also balance validation set
    true_val = val_data.filter(col("label") == True)
    false_val = val_data.filter(col("label") == False)

    true_balanced_val = true_val.sample(withReplacement = True, fraction = 3.0, seed = 42)
    false_balanced_val = false_val.sample(withReplacement=False, fraction = 0.3, seed = 42)

    balanced_val = true_balanced_val.unionByName(false_balanced_val)
    
    return balanced_train, test_data, balanced_val

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
    
def kfold_cross_log_reg(val_df, train_df):
    
    # Create a basic logistic regression
    lr = LogisticRegression(
        featuresCol = 'features',
        labelCol = 'DEP_DEL15',
        tol = 1e-6
    )
    
    # Build a parameter grid for the folds
    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 1.0])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .addGrid(lr.maxIter, [100,500,1000])
        .build()
    )
    
    # Choose the evalator function
    evaluator = BinaryClassificationEvaluator(
        labelCol="DEP_DEL15",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )
    
    # Find the best model
    cv = CrossValidator(
        estimator = lr,
        estimatorParamMaps = paramGrid,
        evaluator = evaluator,
        numFolds = 5,
        seed = 12345
    )
    
    cv_model = cv.fit(val_df)
    
    best_params = cv_model.bestModel.extractParamMap()
    
    print("Best hyperparameter combo:")
    for param, value in best_params.items():
        print(f"  {param.name}: {value}")
    
    # Using those parameters, train on the training set
    lr_final = LogisticRegression(
        featuresCol = 'features',
        labelCol = 'DEP_DEL15',
        tol = 1e-6
    )
    
    for param, value in best_params.items():
        lr_final.set(param, value)
        
    final_model = lr_final.fit(train_df)

    return final_model

def xgboost(train_df):
    
    regressor = SparkXGBRegressor(
        features_col = 'features',
        label_col = 'DEP_DEL15'
    )
    
    model = regressor.fit(train_df)
    
    return model
