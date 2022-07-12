#%%
from tokenize import String
import pyspark
from pyspark.sql import SparkSession
import os
import dotenv
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, StructType, StructField, TimestampType

dotenv.load_dotenv()
#%%

_user = os.getenv("MONGO_USER")
_pass = os.getenv("MONGO_PASS")

spark = (
    SparkSession.builder.appName("PyApp")
    .config(
        "spark.mongodb.input.uri",
        f"mongodb://{_user}:{_pass}@localhost:27017/thesis.prod_tweet?authSource=admin",
    )
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1")
    .getOrCreate()
)


df = (
    spark.read.format("com.mongodb.spark.sql.DefaultSource")
    .schema(
        StructType(
            [
                StructField("_id", StringType()),
                StructField("author_id", StringType()),
                StructField("text", StringType()),
                StructField("created_at", StringType()),
            ]
        )
    )
    .load()
)

df.show(5)

#%%
df.groupBy("author_id").agg(F.count("_id").alias("n_posts")).orderBy(
    "n_posts", ascending=False
).take(25)

#%%

df.withColumn("created_at", F.to_timestamp("created_at")).groupBy(
    F.date_format("created_at", "yyyy-MM-d").alias("date")
).agg(F.count("_id")).take(10)

#%%
df.filter(F.col("text").like("%AMZN%")).show()

#%%
df.withColumn("words", F.split("text", " ")).withColumn(
    "words", F.size(F.filter("words", lambda x: x != "Unusual"))
).take(1)
