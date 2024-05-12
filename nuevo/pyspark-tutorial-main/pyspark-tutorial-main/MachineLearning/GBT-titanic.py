# Importar las bibliotecas necesarias
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


spark = SparkSession.builder \
    .appName("GBTClassifier Optimization") \
    .getOrCreate()

# Cargar datos
titanic_df = spark.read.csv("titanic.csv", header=True, inferSchema=True)

# Seleccionar las columnas 
selected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']
titanic_df = titanic_df.select(selected_columns)

# Eliminar filas con nulos
titanic_df = titanic_df.dropna()

# Convertir la columna con textos a numericas
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df) for column in ['Sex', 'Embarked']]
titanic_indexed = titanic_df
for indexer in indexers:
    titanic_indexed = indexer.transform(titanic_indexed)

# Crear un ensamblador 
assembler = VectorAssembler(inputCols=['Pclass', 'Sex_index', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_index'], outputCol='features')
assembled_data = assembler.transform(titanic_indexed)

# Dividir el conjunto de datos en 3
(training_data, validation_data, test_data) = assembled_data.randomSplit([0.6, 0.2, 0.2], seed=123)

# Crear el modelo
gbt = GBTClassifier(labelCol='Survived', featuresCol='features')

# Definir la cuadrícula 
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5, 7]) \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .build()

# Definir el evaluador
evaluator = BinaryClassificationEvaluator(labelCol='Survived')

# Definir el TrainValidationSplit
tvs = TrainValidationSplit(estimator=gbt,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator,
                           trainRatio=0.8)

# Ajustar  modelo
gbt_model = tvs.fit(training_data)

# Hacer predicciones 
validation_predictions = gbt_model.transform(validation_data)

# Calcular la precisión 
validation_accuracy = evaluator.evaluate(validation_predictions)
print("Validation Accuracy:", validation_accuracy)

# Hacer predicciones 
test_predictions = gbt_model.transform(test_data)

# Calcular la precisión 
test_accuracy = evaluator.evaluate(test_predictions)
print("Test Accuracy:", test_accuracy)

# Detener la sesión de Spark
spark.stop()
