from pyspark.sql import SparkSession
from pyspark.sql.functions import desc

# sesion
spark = SparkSession.builder.appName("DataFrame-Demo").getOrCreate()

#separar, inicializar, agrupar y ordenar
rdd = spark.sparkContext.textFile("./data/data.txt")
result_rdd = rdd.flatMap(lambda line: line.split(" ")).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False)

#mostrar 10 mayores
print(result_rdd.take(10))

#cargar texto
df = spark.read.text("./data/data.txt")

#lo mismo, separar palabras, agrupar contando y ordenar
result_df = df.selectExpr("explode(split(value, ' ')) as word") \
    .groupBy("word").count().orderBy(desc("count"))

#mostrar 10 mayores
print(result_df.take(10))

spark.stop()