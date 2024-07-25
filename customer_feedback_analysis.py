import pandas as pd
from faker import Faker
import random
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import FloatType
from textblob import TextBlob
import nltk

# Initialize Faker and Spark
fake = Faker()
spark = SparkSession.builder.appName("CustomerFeedbackAnalysis").getOrCreate()

# Step 1: Generate synthetic data
data = {
    'CustomerID': [i for i in range(1, 101)],
    'Name': [fake.name() for _ in range(100)],
    'FeedbackDate': [fake.date_this_decade() for _ in range(100)],
    'FeedbackText': [fake.text() for _ in range(100)],
    'FeedbackScore': [random.randint(1, 5) for _ in range(100)]
}

df = pd.DataFrame(data)
df.to_csv('customer_feedback.csv', index=False)

# Step 2: Load data into PySpark
file_location = "customer_feedback.csv"
file_type = "csv"

df_spark = spark.read.format(file_type) \
    .option("inferSchema", "true") \
    .option("header", "true") \
    .option("sep", ",") \
    .load(file_location)

# Step 3: Data cleaning
df_spark = df_spark.filter(col("CustomerID").rlike("^[0-9]+$"))
df_spark.createOrReplaceTempView("customer_feedback")

# Step 4: Basic data analysis using SQL
average_feedback_score = spark.sql("""
SELECT round(AVG(FeedbackScore), 3) AS Average_Feedback_Score
FROM customer_feedback
""")
average_feedback_score.show()

highest_feedback_customers = spark.sql("""
SELECT Name, FeedbackScore 
FROM customer_feedback 
WHERE FeedbackScore = 5
""")
highest_feedback_customers.show()

most_feedback_customers = spark.sql("""
SELECT Name, COUNT(FeedbackText) AS FeedbackCount
FROM customer_feedback
GROUP BY Name
ORDER BY FeedbackCount DESC
""")
most_feedback_customers.show()

# Step 5: Install necessary libraries for sentiment analysis
nltk.download('brown')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Step 6: Sentiment analysis function
def analyze_sentiment(feedback):
    analysis = TextBlob(feedback)
    return analysis.sentiment.polarity

# Register the function as a UDF
analyze_sentiment_udf = udf(analyze_sentiment, FloatType())

# Calculate sentiment score and add a new column
df_spark = df_spark.withColumn("SentimentScore", analyze_sentiment_udf(df_spark.FeedbackText))

# Categorize sentiment
df_spark = df_spark.withColumn(
    "Emotion",
    when(df_spark.SentimentScore < -0.3, "Unhappy")
    .when((df_spark.SentimentScore >= -0.3) & (df_spark.SentimentScore <= 0.3), "Neutral")
    .when(df_spark.SentimentScore > 0.3, "Happy")
)

df_spark.createOrReplaceTempView("customer_feedback")

# Display the DataFrame with sentiment analysis
df_spark.select("CustomerID", "Name", "SentimentScore", "Emotion").show()

# Step 7: Visualize sentiment analysis results using SQL
emotion_count = spark.sql("""
SELECT Emotion, COUNT(Emotion) AS Count 
FROM customer_feedback
GROUP BY Emotion
""")
emotion_count.show()
