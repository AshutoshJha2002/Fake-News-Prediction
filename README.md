# Fake News Prediction

This project implements a machine learning model to classify news articles as either "Real" or "Fake" using text data. The model is built using Logistic Regression 
and TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization.

# Table of Contents

1.Dataset

2.Dependencies

3.Project Structure

4.Preprocessing

5.Model Training

6.Evaluation

7.Usage

# Dataset

The dataset used in this project is a CSV file that contains the following columns:

author: The author of the news article

title: The title of the news article

text: The full text of the news article

label: The label for the news article (0 for Real, 1 for Fake)

# Dependencies

To run the project, you need the following libraries:

Python 3.x

NumPy

Pandas

scikit-learn

NLTK (Natural Language Toolkit)

You can install the required libraries using the following command:

pip install numpy pandas scikit-learn nltk

# Project Structure

train.csv: The dataset file containing the news articles.

fake_news_detection.py: The main Python script for training and evaluating the model.

# Preprocessing

1.Handling Missing Values: Any missing values in the dataset are replaced with an empty string.

2.Text Preprocessing: The author's name and title of the news articles are combined into a single column content. The text data is then preprocessed by:

Removing non-alphabetic characters.

Converting text to lowercase.

Removing stopwords (commonly used words such as "the", "is", etc.).

Stemming words to their root form using the Porter Stemmer.

# Model Training

TF-IDF Vectorization: The preprocessed text data is converted into numerical data using TF-IDF Vectorizer.

Logistic Regression: The Logistic Regression model is trained on the vectorized data.

Train-Test Split: The dataset is split into training and testing sets with stratification to maintain the distribution of labels.

# Evaluation

The model's performance is evaluated using accuracy scores on both the training and test data. The accuracy scores are printed after training.

# Usage

To use the model for predicting whether a news article is real or fake, follow these steps:

Run the fake_news_detection.py script.

The script will output the accuracy scores for both the training and test data.

A sample prediction is made on a test data point, and the result is displayed as either "Real" or "Fake".

# Example output:

Accuracy score of the training data :  0.986

Accuracy score of the test data :  0.955

Prediction for a sample test article: Fake
