from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline



spark = SparkSession.builder \
    .appName("Titanic Decision Tree Classifier with Cross Validation") \
    .getOrCreate()

# Cargar datos
titanic_df = spark.read.csv("titanic.csv", header=True, inferSchema=True)

titanic_df = titanic_df.dropna(subset=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Convertir variables categóricas a numéricas
indexers = [
    StringIndexer(inputCol=column, outputCol=column+"_index").fit(titanic_df)
    for column in ["Sex", "Embarked"]
]
pipeline = Pipeline(stages=indexers)
titanic_df = pipeline.fit(titanic_df).transform(titanic_df)

# Seleccionar las características para el modelo
feature_columns = ['Pclass', 'Sex_index', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_index']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
data = assembler.transform(titanic_df).select("features", "Survived")

# Dividir los datos en 3
train_data, test_data, val_data = data.randomSplit([0.6, 0.2, 0.2], seed=42)

# modelo
dt = DecisionTreeClassifier(featuresCol='features', labelCol='Survived')

# cuadrícula 
param_grid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [3, 5, 7]) \
    .addGrid(dt.minInfoGain, [0.0, 0.1, 0.2]) \
    .build()

# evaluador
evaluator = BinaryClassificationEvaluator(labelCol='Survived')

# validación cruzada
crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=param_grid,
                          evaluator=evaluator,
                          numFolds=5)

# Entrenar el modelo 
cv_model = crossval.fit(train_data)

# Hacer predicciones
predictions = cv_model.transform(test_data)

# Evaluar el rendimiento 
accuracy = evaluator.evaluate(predictions)

print("Accuracy:", accuracy)


# Detener la sesión 
spark.stop()
