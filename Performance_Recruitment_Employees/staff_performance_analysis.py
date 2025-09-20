import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import PartialDependenceDisplay

# Load the dataset
data = pd.read_csv('Performance_Recruitment_Employees/clinic_performance.csv')

print("Dataset First Sight:\n")
print(data.head())

# Cleaning process
print("\nMissing Values in the dataset:\n")
print(data.isnull().sum())

# missing values for a column and the type of data
print("\nGeneral Information:\n")
print(data.info())

# No missing values -> no data imputation necessary

# =================== IMPUTATION not necessary =============================
# imputation for numerical: fill missing values with the mean
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in num_cols:
    data.loc[:, col] = data[col].fillna(data[col].mean())

# imputation for categorical data: fill missing values with the mode
cat_cols = data.select_dtypes(include=['object']).columns
for col in cat_cols:
    data.loc[:, col] = data[col].fillna(data[col].mode()[0])

# statistical overview
print("\nDescriptive Statistics:\n")
print(data.describe())


# ======================== Exploratory Data Analyses =======================
# EDA Process

# ======================== Grasp the data visualization ====================
# 1. Boxplot for Age
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Age'], color='lightgreen')
plt.title("Boxplot for Age")
plt.show()

# 2. Histograms
# Readability issues
column_name_mapping = {
    'EmpEnvironmentSatisfaction': 'Environment Satisfaction',
    'EmpJobSatisfaction': 'Job Satisfaction',
    'EmpRelationshipSatisfaction': 'Relationship Satisfaction'
}
columns_to_plot = [
    'EmpEnvironmentSatisfaction',
    'EmpJobSatisfaction',
    'EmpRelationshipSatisfaction',
]
# Print separate histograms
for col in columns_to_plot:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[col], kde=True, bins=20, color='green')
    plt.title(f"Histogram for {column_name_mapping[col]}")
    plt.xlabel('Satisfaction rate')
    plt.show()

# ======================== Deep into the data visualization ==================

#  Bar-plot for checking job satisfaction regarding each job
def avg_job_satisfy(job):
    # used average
    avg_satisfaction = job.groupby('EmpJobRole')['EmpJobSatisfaction'].mean().reset_index()
    avg_satisfaction = avg_satisfaction.sort_values(by='EmpJobSatisfaction', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='EmpJobRole', y='EmpJobSatisfaction', data=avg_satisfaction, palette="Set2")
    plt.title("Average Job Satisfaction")
    plt.xlabel("Position")
    plt.ylabel("Satisfaction")
    plt.xticks(rotation=45, ha='right')
    plt.show()

avg_job_satisfy(data)

#  Bar-plot for checking job performance regarding each job
def avg_job_perform(job):
    # used average
    avg_performance = job.groupby('EmpJobRole')['PerformanceRating'].mean().reset_index()
    avg_performance = avg_performance.sort_values(by='PerformanceRating', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='EmpJobRole', y='PerformanceRating', data=avg_performance, palette="Set2")
    plt.title("Average Job Performance")
    plt.xlabel("Position")
    plt.ylabel("Performance")
    plt.xticks(rotation=45, ha='right')
    plt.show()

avg_job_perform(data)

# ======================== Label Encoding ======================

# process: encoding categorical data
categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    label_encoders[col] = label_encoder #store for later use

# check conversion
print("\nConverted columns:")
print(categorical_columns)

# Correlation Analysis for each variable
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap=plt.get_cmap('Greens'), fmt='.2f')
plt.title('Correlation Matrix')
plt.show()



# The variable most correlated with performance
performance_corr_var = correlation_matrix['PerformanceRating'].sort_values(ascending=False)
print("\nPerformance Correlation:\n")
print(performance_corr_var)

"""from plot: environment satisfaction is highly influencing performance"""


#   ================ MACHINE LEARNING ============================

X = data.drop('PerformanceRating', axis=1) # dependent variable
y = data['PerformanceRating']                     # independent variable
feature_names = X.columns

# features standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""random_state=42: just as common value in engineering. 
    Important is to have same random state for split and model."""

# Training
# Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

# ================== Classification report ====================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Check accuracy score
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

"""from plot: a good and accurate prediction >80. """

# Plot feature importance for target var = performance rating
imp_feature = model.feature_importances_

# Dataframe
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': imp_feature
})

# Sorting
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Plot -> descending
plt.figure(figsize=(10, 6))
plt.barh(fi_df['Feature'], fi_df['Importance'], color='green')
plt.xlabel('Importance level')
plt.ylabel('Features List')
plt.title('Feature impact')
plt.gca().invert_yaxis()
plt.show()

"""again from plot: environment satisfaction is key factor affecting performance"""

target_class = 4  # Check for class 4 -> since it had the lowest metrics from report

# Since bot variables are correlating, check how much is predicted to affect in terms of probability
# & see if there's strong positive cause relationship
PartialDependenceDisplay.from_estimator(
    model, X_train, ['EmpEnvironmentSatisfaction'], kind="average",
    feature_names=feature_names, target=target_class
)

plt.title(f"Highest performance rating <- correlation -> employees' environment satisfaction level")
plt.xlabel("Environment Satisfaction Level")
plt.ylabel("Predicted Probability for performance rating (4)")
plt.show()
