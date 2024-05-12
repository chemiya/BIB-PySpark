from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql.functions import col


spark = SparkSession.builder \
    .appName("Boston Housing Regression Advanced") \
    .getOrCreate()

# Cargar datos
housing_df = spark.read.csv("HousingData.csv", header=True, inferSchema=True)


columns = housing_df.columns
# Convertir todas las columnas a tipo Double
for col_name in columns:
    housing_df = housing_df.withColumn(col_name, col(col_name).cast('double'))



# Eliminar nulos
housing_df = housing_df.dropna()

# Seleccionar las características 
feature_columns = housing_df.columns[:-1]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="raw_features")
data = assembler.transform(housing_df).select("raw_features", "medv")

# estandarizar 
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
scaler_model = scaler.fit(data)
scaled_data = scaler_model.transform(data).select("features", "medv")

# Dividir los datos en 3 conjuntos
train_data, val_data, test_data = scaled_data.randomSplit([0.6, 0.2, 0.2], seed=42)

# Crear modelo
rf = RandomForestRegressor(featuresCol='features', labelCol='medv')

# Definir una cuadrícula 
param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100, 150]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

# evaluador
evaluator = RegressionEvaluator(labelCol='medv', predictionCol='prediction', metricName='rmse')

# Configurar la división 
tvs = TrainValidationSplit(estimator=rf,
                           estimatorParamMaps=param_grid,
                           evaluator=evaluator,
                           trainRatio=0.8)

# Entrenar el modelo 
tvs_model = tvs.fit(train_data)

# Hacer predicciones 
predictions = tvs_model.transform(test_data)

# Calcular métricas 
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)

# Detener la sesión de Spark
spark.stop()
