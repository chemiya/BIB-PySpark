# Importar las bibliotecas necesarias
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator


spark = SparkSession.builder \
    .appName("Titanic SVM") \
    .getOrCreate()

# Cargar datos
titanic_df = spark.read.csv("titanic.csv", header=True, inferSchema=True)

# Seleccionar las columnas 
selected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
titanic_df = titanic_df.select(selected_columns)

# Eliminar filas con  nulos
titanic_df = titanic_df.dropna()

# Convertir la columna 'Sex' y 'Embarked' a numeros
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ['Sex', 'Embarked']]
titanic_indexed = titanic_df
for indexer in indexers:
    titanic_indexed = indexer.transform(titanic_indexed)

# ensamblador
assembler = VectorAssembler(inputCols=['Pclass', 'Sex_index', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_index'], outputCol='features')
assembled_data = assembler.transform(titanic_indexed)

# Dividir los datos
(training_data, test_data) = assembled_data.randomSplit([0.7, 0.3], seed=123)

# Entrenar 
svm = LinearSVC(labelCol='Survived', featuresCol='features')
svm_model = svm.fit(training_data)

# Realizar predicciones 
predictions = svm_model.transform(test_data)

# Evaluar el rendimiento 
evaluator = BinaryClassificationEvaluator(labelCol='Survived')
accuracy = evaluator.evaluate(predictions)

# Mostrar la precisión 
print("Accuracy:", accuracy)

# Detener la sesión de Spark
spark.stop()
