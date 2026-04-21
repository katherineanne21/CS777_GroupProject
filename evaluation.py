#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def evaluate_predictions(predictions):
    # accuracy
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="DEP_DEL15",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)

    # precision
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="DEP_DEL15",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)

    # recall
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="DEP_DEL15",
        predictionCol="prediction",
        metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)

    # F1
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="DEP_DEL15",
        predictionCol="prediction",
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def confusion_matrix_counts(predictions):
    tp = predictions.filter((col("DEP_DEL15") == 1) & (col("prediction") == 1)).count()
    tn = predictions.filter((col("DEP_DEL15") == 0) & (col("prediction") == 0)).count()
    fp = predictions.filter((col("DEP_DEL15") == 0) & (col("prediction") == 1)).count()
    fn = predictions.filter((col("DEP_DEL15") == 1) & (col("prediction") == 0)).count()

    print("\nConfusion Matrix Counts:")
    print(f"TP: {tp}")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")


def evaluate_baseline(test_df):
    # baseline model: always predict on-time (0)
    baseline_predictions = test_df.withColumn(
        "prediction",
        when(col("DEP_DEL15").isNotNull(), 0.0)
    )

    print("\nBaseline Model Results (always predict on-time):")
    evaluate_predictions(baseline_predictions)
    confusion_matrix_counts(baseline_predictions)
