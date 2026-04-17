# Machine Learning Analysis of COVID-19 Spread in Indonesia

This project analyzes the spread of COVID-19 in Indonesia using machine learning techniques, combining unsupervised learning for regional risk segmentation and supervised learning for high-case prediction. The project also includes an interactive Streamlit dashboard for symptom-based prediction and result visualization.

## Project Overview

The main objectives of this project are to:

- analyze COVID-19 case patterns across Indonesia
- identify regional risk groups using clustering
- predict high-case risk levels using classification
- provide an interactive dashboard for easier interpretation of results

This project was developed as part of an academic team project at Telkom University, where I contributed as the Team Leader and coordinated the end-to-end workflow from data preparation to modeling and dashboard development.

## Key Highlights

- Led a cross-functional team of 6+ members in delivering the project end-to-end
- Processed and analyzed 31,822 COVID-19 records across 8 variables
- Evaluated 9 clustering scenarios to determine the optimal segmentation structure
- Identified 3 provincial risk clusters using K-Means Clustering
- Built a Logistic Regression model with 82.36% accuracy for high-case classification
- Developed an interactive Streamlit dashboard with 5 symptom inputs:
  - fever
  - cough
  - shortness of breath
  - sore throat
  - fatigue

## Dataset

- Source: Indonesia COVID-19 time series dataset
- Coverage: March 2020 to December 2022
- Records: 31,822
- Variables used:
  - New Cases
  - New Deaths
  - New Recovered
  - Total Cases
  - Total Deaths
  - Total Recovered
  - Location
  - Province Encoded

## Methods Used

### 1. Data Preprocessing
- feature selection
- label encoding
- missing value handling
- standardization using StandardScaler
- exploratory data analysis

### 2. K-Means Clustering
K-Means Clustering was used to group provinces based on COVID-19 characteristics.

- Tested cluster range: k = 2 to 10
- Optimal number of clusters: 3
- Evaluation methods:
  - Elbow Method
  - Silhouette Score

### 3. Logistic Regression
Logistic Regression was used to classify high-case risk levels.

- Train-test split: 70:30
- Accuracy: 82.36%

### 4. Streamlit Dashboard
The project includes an interactive dashboard that allows users to input symptoms and view prediction results visually.

Features:
- symptom-based input form
- prediction results
- probability visualization
- model information
- confusion matrix display

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

Results
This project shows that machine learning can help uncover regional COVID-19 risk patterns and support predictive analysis for public health decision-making. The clustering model highlights differences in provincial case characteristics, while the classification model provides a practical way to identify high-risk conditions.

My Role
As Team Leader, I was responsible for:
coordinating the team workflow
guiding data preprocessing and feature preparation
overseeing clustering and classification modeling
supporting dashboard development and integration
ensuring the project was completed successfully end-to-end
Future Improvements
add more advanced classification models
improve dashboard usability and design
integrate real-time or updated datasets
enhance model evaluation with additional metrics

Author
Muhammad Abrar Rayhan
Telkom University
