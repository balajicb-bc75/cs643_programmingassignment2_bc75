from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import RandomForestClassifier

# Create a SparkSession
spark = SparkSession.builder.appName("WineQuality").getOrCreate()

# Define the schema for the DataFrame
schema = StructType([
    StructField("fixed acidity", DoubleType(), True),
    StructField("volatile acidity", DoubleType(), True),
    StructField("citric acid", DoubleType(), True),
    StructField("residual sugar", DoubleType(), True),
    StructField("chlorides", DoubleType(), True),
    StructField("free sulfur dioxide", IntegerType(), True),
    StructField("total sulfur dioxide", IntegerType(), True),
    StructField("density", DoubleType(), True),
    StructField("pH", DoubleType(), True),
    StructField("sulphates", DoubleType(), True),
    StructField("alcohol", DoubleType(), True),
    StructField("quality", IntegerType(), True)
])

# Read the data file into a DataFrame
data_path = "s3://cs643-programming-assignment-2/TrainingDataset.csv"  # Replace with the actual path to your data file
training_data = spark.read.csv(data_path, header=True, schema=schema, sep=";")

# Show the DataFrame
training_data.show()

# Rename the 'quality' column to 'label' for the target variable
training_data = training_data.withColumnRenamed("quality", "label")

# List of feature columns (excluding the target column)
feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"]

# Create the VectorAssembler to combine features into a feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
training_data = assembler.transform(training_data)

# Initialize the Logistic Regression model
lr_model = LogisticRegression(featuresCol="features", labelCol="label")

# Initialize the Random Forest model
rf_model = RandomForestClassifier(featuresCol="features", labelCol="label")

# Initialize the evaluator with F1 score metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

# Create a ParamGrid for cross-validation
param_grid = ParamGridBuilder() \
    .addGrid(lr_model.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr_model.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Create a ParamGrid for Random Forest
param_grid_rf = ParamGridBuilder() \
    .addGrid(rf_model.maxDepth, [5, 10, 20]) \
    .addGrid(rf_model.numTrees, [10, 20, 30]) \
    .build()


# Initialize CrossValidator
cross_validator_lf = CrossValidator(estimator=lr_model,
                                estimatorParamMaps=param_grid,
                                evaluator=evaluator,
                                numFolds=5,  # Number of folds for cross-validation
                                seed=42)

# Initialize CrossValidator
cross_validator_rf = CrossValidator(estimator=rf_model,
                                estimatorParamMaps=param_grid_rf,
                                evaluator=evaluator,
                                numFolds=5,  # Number of folds for cross-validation
                                seed=42)

# Run cross-validation and get the best model
cv_model = cross_validator_lf.fit(training_data)

# Get the best model from cross-validation
best_model = cv_model.bestModel
lr_model_path = "s3://cs643-programming-assignment-2/lr_model"
best_model.save(lr_model_path)

# Access other model attributes (e.g., intercepts)
coefficient_matrix = best_model.coefficientMatrix
intercept_vector = best_model.interceptVector
num_classes = best_model.numClasses

# Calculate the F1 score on the training dataset
predictions = best_model.transform(training_data)
f1_score_lm = evaluator.evaluate(predictions)

# Run cross-validation and get the best model
cv_model_rf = cross_validator_rf.fit(training_data)

# Get the best model from cross-validation
best_model_rf = cv_model_rf.bestModel
rf_model_path = "s3://cs643-programming-assignment-2/rf_model"
best_model_rf.save(rf_model_path)


# Access other model attributes (e.g., intercepts)
feature_importances = best_model_rf.featureImportances
num_classes_rf = best_model_rf.numClasses

# Calculate the F1 score on the training dataset
predictions_rf = best_model_rf.transform(training_data)
f1_score_rf = evaluator.evaluate(predictions_rf)

# Print the F1 score
print("F1 Score on Training Data using Logistic Regression: {:.4f}".format(f1_score_lm))
print("F1 Score on Training Data using Random Forest: {:.4f}".format(f1_score_rf))

spark.stop()