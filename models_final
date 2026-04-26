#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col


def split_data(df):
    train_data, temp_data = df.randomSplit([0.7, 0.3], seed=42)
    val_data, test_data = temp_data.randomSplit([0.5, 0.5], seed=42)

    # Balance ONLY training set
    true_train = train_data.filter(col("DEP_DEL15") == 1)
    false_train = train_data.filter(col("DEP_DEL15") == 0)

    true_balanced = true_train.sample(withReplacement=True, fraction=3.0, seed=42)
    false_balanced = false_train.sample(withReplacement=False, fraction=0.3, seed=42)

    balanced_train = true_balanced.unionByName(false_balanced)

    return balanced_train, test_data, val_data


def create_log_reg_model(train_df):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="DEP_DEL15",
        regParam=0.5,
        elasticNetParam=0.0,
        maxIter=100,
        tol=1e-6
    )
    return lr.fit(train_df)


def kfold_cross_log_reg(val_df, train_df):
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="DEP_DEL15",
        tol=1e-6
    )

    paramGrid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, [0.01, 0.1, 1.0])
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
        .addGrid(lr.maxIter, [100, 500, 1000])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="DEP_DEL15",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=5,
        seed=12345
    )

    cv_model = cv.fit(val_df)
    best_model = cv_model.bestModel

    print("Best hyperparameter combo:")
    print(f"  regParam: {best_model.getRegParam()}")
    print(f"  elasticNetParam: {best_model.getElasticNetParam()}")
    print(f"  maxIter: {best_model.getMaxIter()}")

    final_lr = LogisticRegression(
        featuresCol="features",
        labelCol="DEP_DEL15",
        tol=1e-6,
        regParam=best_model.getRegParam(),
        elasticNetParam=best_model.getElasticNetParam(),
        maxIter=best_model.getMaxIter()
    )

    return final_lr.fit(train_df)


def xgboost(train_df):
    # USE NORMAL XGBOOST (NOT Spark version)
    from xgboost import XGBClassifier
    import numpy as np

    # Convert Spark DF → Pandas
    pdf = train_df.select("features", "DEP_DEL15").toPandas()

    X = np.array([x.toArray() for x in pdf["features"]])
    y = pdf["DEP_DEL15"].values

    model = XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        objective="binary:logistic",
        use_label_encoder=False,
        eval_metric="logloss"
    )

    model.fit(X, y)

    return model
