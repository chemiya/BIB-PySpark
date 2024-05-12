from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Create-DataFrame").getOrCreate()

# leer csv
csv_file_path = "./data/products.csv"
df = spark.read.csv(csv_file_path, header=True)


# ver esquema
df.printSchema()

# ver contenido
df.show(5)

# importar tipos
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

# definir esquema
schema = StructType([
    StructField(name="id", dataType=IntegerType(), nullable=True),
    StructField(name="name", dataType=StringType(), nullable=True),
    StructField(name="category", dataType=StringType(), nullable=True),
    StructField(name="quantity", dataType=IntegerType(), nullable=True),
    StructField(name="price", dataType=DoubleType(), nullable=True)
])

# leer csv
csv_file_path = "./data/products.csv"
df = spark.read.csv(csv_file_path, header=True, schema=schema)


# ver esquema
df.printSchema()

# ver contenido
df.show(5)


#json, cada elemento en una linea
json_file_path = "./data/products_singleline.json"
df = spark.read.json(json_file_path)

# ver esquema
df.printSchema()

# ver contenido
df.show(5)

#en array con json
json_file_path = "./data/products_multiline.json"
df = spark.read.json(json_file_path, multiLine=True)

# ver esquema
df.printSchema()

# ver contenido
df.show(5)


spark.stop()