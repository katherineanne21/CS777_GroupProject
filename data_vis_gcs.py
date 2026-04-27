from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import os

def visualizations(filename):

    spark = SparkSession.builder.appName("FinalFlightAnalysis").getOrCreate()
    bucket_path = "gs://cs777_katherine_a_rein/TermProject/plots/"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    
    # =========================
    # LOADING CLEANED FULL DATA
    # =========================
    
    df = spark.read.parquet(filename)
    
    print("DATA LOADED")
    
    # =========================
    # SELECT REQUIRED COLUMNS
    # =========================
    df = df.select("MONTH", "DAY_OF_WEEK", "DISTANCE", "DEP_DEL15")
    
    # =========================
    # ENSURING CORRECT TYPES
    # =========================
    df = df.withColumn("MONTH", col("MONTH").cast("int"))
    df = df.withColumn("DAY_OF_WEEK", col("DAY_OF_WEEK").cast("int"))
    df = df.withColumn("DEP_DEL15", col("DEP_DEL15").cast("int"))
    df = df.withColumn("DISTANCE", col("DISTANCE").cast("double"))
    
    # =========================
    # FILTERING VALID MONTHS
    # =========================
    df = df.filter((col("MONTH") >= 1) & (col("MONTH") <= 12))
    
    # =========================
    # DEBUGGING to check
    # =========================
    df.groupBy("MONTH").count().orderBy("MONTH").show()
    
    # =========================
    # 1. DELAY BY MONTH
    # =========================
    monthly = df.filter(col("DEP_DEL15") == 1) \
                .groupBy("MONTH") \
                .count() \
                .orderBy("MONTH")
    
    pdf_month = monthly.toPandas()
    
    plt.figure()
    plt.bar(pdf_month["MONTH"], pdf_month["count"])
    plt.xticks(range(1,13))
    plt.xlabel("Month")
    plt.ylabel("Number of Delays")
    plt.title("Delayed Flights by Month")
    
    local_month = f"/tmp/delay_by_month_{timestamp}.png"
    plt.savefig(local_month)
    os.system(f"gcloud storage cp {local_month} {bucket_path}delay_by_month_{timestamp}.png")
    os.remove(local_month)
    
    print("Saved delay_by_month.png")
    
    # =========================
    # 2. DELAY BY DAY
    # =========================
    dow = df.filter(col("DEP_DEL15") == 1) \
            .groupBy("DAY_OF_WEEK") \
            .count() \
            .orderBy("DAY_OF_WEEK")
    
    pdf_dow = dow.toPandas()
    
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    
    plt.figure()
    plt.bar(pdf_dow["DAY_OF_WEEK"], pdf_dow["count"])
    plt.xticks([1,2,3,4,5,6,7], day_labels)
    plt.title("Delayed Flights by Day of Week")
    
    local_dow = f"/tmp/delay_by_day_{timestamp}.png"
    plt.savefig(local_dow)
    os.system(f"gcloud storage cp {local_dow} {bucket_path}delay_by_day_{timestamp}.png")
    os.remove(local_dow)
    
    print("Saved delay_by_day.png")
    
    # =========================
    # 3. DISTANCE VS DELAY
    # =========================
    dist = df.groupBy("DEP_DEL15").avg("DISTANCE")
    
    pdf_dist = dist.toPandas()
    
    plt.figure()
    plt.bar(pdf_dist["DEP_DEL15"], pdf_dist["avg(DISTANCE)"])
    plt.xticks([0,1], ["No Delay", "Delay"])
    plt.title("Average Distance vs Delay")
    
    local_dist = f"/tmp/distance_vs_delay_{timestamp}.png"
    plt.savefig(local_dist)
    os.system(f"gcloud storage cp {local_dist} {bucket_path}distance_vs_delay_{timestamp}.png")
    os.remove(local_dist)
    
    print("Saved distance_vs_delay.png")

