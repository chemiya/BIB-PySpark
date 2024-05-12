from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("RDD-Demo").getOrCreate()

# crear rdd de tuplas
data = [("Alice", 25), ("Bob", 30), ("Charlie", 35), ("Alice", 40)]
rdd = spark.sparkContext.parallelize(data)

# ver sus datos
print("All elements of the rdd: ", rdd.collect())

# contar elementos
count = rdd.count()
print("The total number of elements in rdd: ", count)

# obtener primer elemento
first_element = rdd.first()
print("The first element of the rdd: ", first_element)

# coger 2 primeros
taken_elements = rdd.take(2)
print("The first two elements of the rdd: ", taken_elements)

# Foreach para cada uno imprimir
rdd.foreach(lambda x: print(x))

# convierte nombre a mayusculas
mapped_rdd = rdd.map(lambda x: (x[0].upper(), x[1]))

#imprimir
result = mapped_rdd.collect()
print("rdd with uppercease name: ", result)

# filtra los de edades mayores a 30
filtered_rdd = rdd.filter(lambda x: x[1] > 30)
print(filtered_rdd.collect())

# calcula suma de edades por nombre
reduced_rdd = rdd.reduceByKey(lambda x, y: x + y)
print(reduced_rdd.collect())

# ordena por edad descendente
sorted_rdd = rdd.sortBy(lambda x: x[1], ascending=False)
print(sorted_rdd.collect())

# guardar en txt
rdd.saveAsTextFile("output.txt")

# leer de text
rdd_text = spark.sparkContext.textFile("output.txt")
rdd_text.collect()

#parar sesion
spark.stop()