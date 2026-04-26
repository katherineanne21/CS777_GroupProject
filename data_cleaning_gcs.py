from pyspark.sql.functions import col, sum
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from datetime import datetime


def cleaning_flight_data(spark, filename):

    # ---------------- LOADING  DATA ----------------
    df = spark.read.format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("sep", ",") \
        .option("mode", "DROPMALFORMED") \
        .load(filename)

    print("Data Loaded Successfully")

    # filtering garbage rows
    df = df.filter(col("YEAR") == 2019)

    # then checking MONTH
    print("Checking MONTH values AFTER filtering:")
    df.select("MONTH").distinct().show()

    # ---------------- CHECK ----------------
    df.show(5)

    # ---------------- SELECT REQUIRED COLUMNS ----------------
    df = df.select(
        "MONTH",
        "DAY_OF_MONTH",
        "DAY_OF_WEEK",
        "OP_UNIQUE_CARRIER",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID",
        "AIR_TIME",
        "DISTANCE",
        "DEP_DEL15"
    )

    # ---------------- MISSING VALUES ----------------
    print("Missing values check:")
    df.select([
        sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).show()

    # ---------------- CLEANING ----------------
    df = df.dropna(subset=[
        "DEP_DEL15",
        "AIR_TIME",
        "DISTANCE",
        "OP_UNIQUE_CARRIER",
        "ORIGIN_AIRPORT_ID",
        "DEST_AIRPORT_ID"
    ])

    df = df.filter((col("DEP_DEL15") == 0) | (col("DEP_DEL15") == 1))
    df = df.filter((col("AIR_TIME") > 0) & (col("DISTANCE") > 0))

    # ---------------- TYPE CASTING (VERY IMPORTANT) ----------------
    df = df.withColumn("DISTANCE", col("DISTANCE").cast("double"))
    df = df.withColumn("AIR_TIME", col("AIR_TIME").cast("double"))
    df = df.withColumn("DEP_DEL15", col("DEP_DEL15").cast("integer"))
    df = df.withColumn("MONTH", col("MONTH").cast("integer"))
    df = df.withColumn("DAY_OF_MONTH", col("DAY_OF_MONTH").cast("integer"))
    df = df.withColumn("DAY_OF_WEEK", col("DAY_OF_WEEK").cast("integer"))
    df = df.withColumn("OP_UNIQUE_CARRIER", col("OP_UNIQUE_CARRIER").cast("string"))
    df = df.withColumn("ORIGIN_AIRPORT_ID", col("ORIGIN_AIRPORT_ID").cast("string"))
    df = df.withColumn("DEST_AIRPORT_ID", col("DEST_AIRPORT_ID").cast("string"))

    # Cache for performance
    df = df.cache()

    print("After Cleaning")

    # ---------------- CLASS IMBALANCE ----------------
    print("Class Distribution (Imbalance Check):")
    df.groupBy("DEP_DEL15").count().show()

    # ---------------- SAVE OUTPUT ----------------
    print("Saving Processed Data to S3...")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    gcs_filename = f'gs://cs777_katherine_a_rein/TermProject/cleaned_preprocessed_2019_{timestamp}/'

    df.write.mode("overwrite").parquet(gcs_filename)

    print("DONE SUCCESSFULLY")
    
    return gcs_filename


def feature_eng():

    carrier_indexer = StringIndexer(
        inputCol="OP_UNIQUE_CARRIER",
        outputCol="carrier_index",
        handleInvalid="keep"
    )

    origin_indexer = StringIndexer(
        inputCol="ORIGIN_AIRPORT_ID",
        outputCol="origin_index",
        handleInvalid="keep"
    )

    dest_indexer = StringIndexer(
        inputCol="DEST_AIRPORT_ID",
        outputCol="dest_index",
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=["carrier_index", "origin_index", "dest_index"],
        outputCols=["carrier_vec", "origin_vec", "dest_vec"]
    )

    num_assembler = VectorAssembler(
        inputCols=["DISTANCE", "AIR_TIME"],
        outputCol="num_features"
    )

    scaler = StandardScaler(
        inputCol="num_features",
        outputCol="scaled_num_features"
    )

    final_assembler = VectorAssembler(
        inputCols=[
            "MONTH",
            "DAY_OF_MONTH",
            "DAY_OF_WEEK",
            "scaled_num_features",
            "carrier_vec",
            "origin_vec",
            "dest_vec"
        ],
        outputCol="features"
    )

    pipeline = Pipeline(stages=[
        carrier_indexer,
        origin_indexer,
        dest_indexer,
        encoder,
        num_assembler,
        scaler,
        final_assembler
    ])

    print("Running Pipeline...")

    return pipeline
