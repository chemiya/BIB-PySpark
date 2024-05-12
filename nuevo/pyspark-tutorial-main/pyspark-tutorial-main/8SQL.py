from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("DataFrameSQL").getOrCreate()


data_file_path = "./data/persons.csv"
df = spark.read.csv(data_file_path, header=True, inferSchema=True)

df.printSchema()

print("Initial DataFrame:")
df.show(10)


# crear tabla temporal
df.createOrReplaceTempView("my_table")


# seleccionar filas con edad superior a 25
result = spark.sql("SELECT * FROM my_table WHERE age > 25")

result.show()


# calculamos salario medio por genero
avg_salary_by_gender = spark.sql("SELECT gender, AVG(salary) as avg_salary FROM my_table GROUP BY gender")
avg_salary_by_gender.show()


# creamos vista temporal
df.createOrReplaceTempView("people")


# buscamos mayores de 25
result = spark.sql("SELECT * FROM people WHERE age > 25")

result.show()

# comprobamos si existe
view_exists = spark.catalog.tableExists("people")
view_exists

# la eliminamos
spark.catalog.dropTempView("people")


# comprobamos si existe
view_exists = spark.catalog.tableExists("people")
view_exists


# creamos dataframes
employee_data = [
    (1, "John"), (2, "Alice"), (3, "Bob"), (4, "Emily"),
    (5, "David"), (6, "Sarah"), (7, "Michael"), (8, "Lisa"),
    (9, "William")
]
employees = spark.createDataFrame(employee_data, ["id", "name"])

salary_data = [
    ("HR", 1, 60000), ("HR", 2, 55000), ("HR", 3, 58000),
    ("IT", 4, 70000), ("IT", 5, 72000), ("IT", 6, 68000),
    ("Sales", 7, 75000), ("Sales", 8, 78000), ("Sales", 9, 77000)
]
salaries = spark.createDataFrame(salary_data, ["department", "id", "salary"])

employees.show()

salaries.show()

# creamos vistas temporales
employees.createOrReplaceTempView("employees")
salaries.createOrReplaceTempView("salaries")


# buscamos empleados cuyo sueldo esta debajo de la media
result = spark.sql("""
    SELECT name
    FROM employees
    WHERE id IN (
        SELECT id
        FROM salaries
        WHERE salary > (SELECT AVG(salary) FROM salaries)
    )
""")

result.show()


from pyspark.sql.window import Window
from pyspark.sql import functions as F


#hacemos join
employee_salary = spark.sql("""
    select  salaries.*, employees.name
    from salaries 
    left join employees on salaries.id = employees.id
""")

employee_salary.show()


# creamos ventana por departamentos
window_spec = Window.partitionBy("department").orderBy(F.desc("salary"))


# ranking por departamento basandose en el salario
employee_salary.withColumn("rank", F.rank().over(window_spec)).show()

# parar sesion
spark.stop()