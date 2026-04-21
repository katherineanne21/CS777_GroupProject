#These visualizations are of RAW DATA

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("FinalVisualizations").getOrCreate()

# =========================
# LOADING DATA (RAW CSV)
# =========================
df = spark.read.csv(
    "s3://777-termproject/csv/flights_2019_full.csv",
    header=True,
    inferSchema=True
)

# =========================
# CLEAN DATA 
# =========================
df = df.select("MONTH", "DAY_OF_WEEK", "DISTANCE", "DEP_DEL15")
df = df.withColumn("DISTANCE", col("DISTANCE").cast("double"))

df = df.dropna()
df = df.filter((col("DEP_DEL15") == 0) | (col("DEP_DEL15") == 1))

print("DATA READY")

# =========================
# 1. CLASS IMBALANCE
# =========================
class_dist = df.groupBy("DEP_DEL15").count()
pdf = class_dist.toPandas()

plt.figure()
plt.bar(pdf["DEP_DEL15"], pdf["count"])
plt.xticks([0,1], ["No Delay", "Delay"])
plt.title("Class Imbalance")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("/home/hadoop/class_imbalance.png")

# =========================
# 2. DELAY BY MONTH
# =========================
monthly = df.filter(col("DEP_DEL15") == 1) \
            .groupBy("MONTH") \
            .count()

pdf_month = monthly.toPandas()

plt.figure()
plt.bar(pdf_month["MONTH"], pdf_month["count"])
plt.title("Delayed Flights by Month")
plt.xlabel("Month")
plt.ylabel("Number of Delays")
plt.savefig("/home/hadoop/delay_by_month.png")

# =========================
# 3. DELAY BY DAY 
# =========================
dow = df.filter(col("DEP_DEL15") == 1) \
        .groupBy("DAY_OF_WEEK") \
        .count()

pdf_dow = dow.toPandas()

# Sort by day number 
pdf_dow = pdf_dow.sort_values("DAY_OF_WEEK")

day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

plt.figure()
plt.bar(pdf_dow["DAY_OF_WEEK"], pdf_dow["count"])
plt.xticks([1,2,3,4,5,6,7], day_labels)
plt.title("Delayed Flights by Day of Week")
plt.xlabel("Day")
plt.ylabel("Number of Delays")
plt.savefig("/home/hadoop/delay_by_day.png")

# =========================
# 4. DISTANCE VS DELAY
# =========================
dist = df.groupBy("DEP_DEL15").avg("DISTANCE")

pdf_dist = dist.toPandas()

plt.figure()
plt.bar(pdf_dist["DEP_DEL15"], pdf_dist["avg(DISTANCE)"])
plt.xticks([0,1], ["No Delay", "Delay"])
plt.title("Average Distance vs Delay")
plt.ylabel("Distance")
plt.savefig("/home/hadoop/distance_vs_delay.png")

spark.stop()

