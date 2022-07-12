#%%
from pyspark.sql import SparkSession
import seaborn as sns

spark = SparkSession.builder.getOrCreate()

#%%
df = spark.createDataFrame(sns.load_dataset("tips"))

#%%
df = df.withColumn("tip_pct", df.tip / df.total_bill)
#%%
df.groupBy("day").agg({"tip_pct": "avg"}).collect()

#%%
df.select("*").where("tip_pct > 0.15").collect()