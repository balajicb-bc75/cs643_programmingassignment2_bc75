import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# from pyspark.ml.classification import RandomForestClassifier

# Getting the file from the argument
if len(sys.argv)< 2:
    print("Please provide the dataset path in the argument")
    sys.exit(1)

data_path = sys.argv[1]
# Create a SparkSession
spark = SparkSession.builder \
    .appName("MyApp") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

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
# data_path = "s3://cs643-programming-assignment-2/TrainingDataset.csv"  # Replace with the actual path to your data file
validation_data = spark.read.csv(data_path, header=True, schema=schema, sep=";")

# Show the DataFrame
validation_data.show()

# Rename the 'quality' column to 'label' for the target variable
validation_data = validation_data.withColumnRenamed("quality", "label")

# List of feature columns (excluding the target column)
feature_cols = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide",
                "density", "pH", "sulphates", "alcohol"]

# Create the VectorAssembler to combine features into a feature vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
validation_data = assembler.transform(validation_data)

# Load the Logistic Regression model
lr_model_path = "lr_model"
load_lr_model = LogisticRegressionModel.load(lr_model_path)

# Load the Random Forest model
rf_model_path = "rf_model"
load_rf_model = RandomForestClassificationModel.load(rf_model_path)

# Initialize the evaluator with F1 score metric
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")



# Calculate the F1 score on the training dataset
predictions = load_lr_model.transform(validation_data)
f1_score_lm = evaluator.evaluate(predictions)


# Calculate the F1 score on the training dataset
predictions_rf = load_rf_model.transform(validation_data)
f1_score_rf = evaluator.evaluate(predictions_rf)

# Print the F1 score
print("F1 Score on Validation Data using Logistic Regression: {:.4f}".format(f1_score_lm))
print("F1 Score on Validation Data using Random Forest: {:.4f}".format(f1_score_rf))

spark.stop()