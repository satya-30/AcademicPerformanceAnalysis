
# AcademicPerformanceAnalysis

This code focuses on applying data science methods to analyze student performance data. In order to facilitate data-driven decision-making in education, it seeks to identify trends and insights that influence academic results. With 10,000 records, the dataset offers a solid foundation for statistical analysis and predictive modeling.


## Installation

pandas (import pandas as pd) - For data manipulation and analysis.
numpy (import numpy as np) - For numerical computations.
matplotlib (import matplotlib.pyplot as plt) - For data visualization.
seaborn (import seaborn as sns) - For advanced data visualization.
scikit-learn - Used for machine learning tasks, including:
RandomForestClassifier
LinearRegression, LogisticRegression
classification_report, mean_squared_error, accuracy_score
confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
train_test_split
LabelEncoder
DecisionTreeClassifier
tabulate (from tabulate import tabulate) - For printing tabular data in a readable format.
## Road Map 

Data Cleaning and Preprocessing
Exploratory Data Analysis (EDA)
Model Training and Evaluation
Results and Visualizations

## Model Performance 


Linear Regression was effective for predicting performance scores (continuous variable) but struggled with non-linear patterns.
Logistic Regression and Random Forest showed superior accuracy in classifying pass/fail outcomes, with Random Forest achieving the highest accuracy of 92%.
Decision Trees provided interpretable results but were prone to overfitting on training data.
