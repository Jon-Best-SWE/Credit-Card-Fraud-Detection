# -*- coding: utf-8 -*-
"""
Created on Sun Jul 2 08:41:52 2023

@author: JonBest
"""

# Jon Best
# 7/2/2023
# CS379 - Machine Learning
# Common Applications of Machine Learning 
# The purpose of this Python code is to use the Random Forest Classifier algorithm 
# to evaluate statistical information about credit card fraud and detect anomolies 
# in spending patterns and behavior activities that could indicate identity theft.
 
#***************************************************************************************
# Title: Credit Card Fraud Detection With Classification Algorithms In Python
# Author: Polamuri, S.
# Date: 2020
# Availability: https://dataaspirant.com/credit-card-fraud-detection-classification-algorithms-python/
#
# Title: Credit Card Fraud Detection With Machine Learning in Python
# Author: Adithyan, N. 
# Date: 2020
# Availability: https://medium.com/codex/credit-card-fraud-detection-with-machine-learning-in-python-ac7281991d87
#
# Title: A Complete Guide to Data Visualization in Python With Libraries & More
# Author: Ravikiran A S
# Date: 2023
# Availability: https://www.simplilearn.com/tutorials/python-tutorial/data-visualization-in-python
#
#***************************************************************************************

# Imported libraries include: pandas to develop dataframes, sklearn for machine learning functions, 
# and matplotlib plus seasborn for graphic representation of data.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

# Reading CSV file to retrieve the required data.
data = pd.read_csv('Credit Card Fraud Modified.csv')

# Explores data analysis of dataset. 
print(data.shape)
print(data.head())
print(data.dtypes)

# Selects features from the dataset.
features = ["over_draft", "credit_usage", "credit_history", "purpose", "current_balance",
            "Average_Credit_Balance", "employment", "location", "personal_status",
            "other_parties", "residence_since", "property_magnitude", "cc_age",
            "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
            "own_telephone", "foreign_worker"]

# Creates histograms for numeric features.
numeric_features = ["credit_usage", "current_balance", "Average_Credit_Balance", "cc_age", "existing_credits", "num_dependents"]
data[numeric_features].hist(bins=10, figsize=(12, 8))
plt.tight_layout()
plt.show()
plt.close()

# Creates count plots for categorical features.
categorical_features = [feat for feat in features if feat not in numeric_features]
for feat in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=feat, hue="class")
    plt.xticks(rotation=90)
    plt.xlabel(feat)
    plt.ylabel("Count")
    plt.title(feat + " Distribution")
    plt.legend(title="Class", loc="upper right")
    plt.tight_layout()
    plt.show()
    plt.close()

# Ensure that dataset is void of null values.
print(data.isna().sum())

# Display unique objects before conversion.
print(data['over_draft'].unique())
print(data['credit_history'].unique())
print(data['purpose'].unique())
print(data['Average_Credit_Balance'].unique())
print(data['employment'].unique())
print(data['personal_status'].unique())
print(data['other_parties'].unique())
print(data['property_magnitude'].unique())
print(data['other_payment_plans'].unique())
print(data['housing'].unique())
print(data['job'].unique())
print(data['own_telephone'].unique())
print(data['foreign_worker'].unique())
print(data['class'].unique())

#Convert object datatypes to integers for all applicable features.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transform the 'over_draft' column to integers.
data.iloc[:,0]= labelencoder.fit_transform(data.iloc[:,0].values)

# Transform the 'credit_history' column to integers.
data.iloc[:,2]= labelencoder.fit_transform(data.iloc[:,2].values)

# Transform the 'purpose' column to integers.
data.iloc[:,3]= labelencoder.fit_transform(data.iloc[:,3].values)

# Transform the 'Average_Credit_Balance' column to integers.
data.iloc[:,5]= labelencoder.fit_transform(data.iloc[:,5].values)

# Transform the 'employment' column to integers.
data.iloc[:,6]= labelencoder.fit_transform(data.iloc[:,6].values)

# Transform the 'personal_status' column to integers.
data.iloc[:,8]= labelencoder.fit_transform(data.iloc[:,8].values)

# Transform the 'other_parties' column to integers.
data.iloc[:,9]= labelencoder.fit_transform(data.iloc[:,9].values)

# Transform the 'property_magnitude' column to integers.
data.iloc[:,11]= labelencoder.fit_transform(data.iloc[:,11].values)

# Transform the 'other_payment_plans' column to integers.
data.iloc[:,13]= labelencoder.fit_transform(data.iloc[:,13].values)

# Transform the 'housing' column to integers.
data.iloc[:,14]= labelencoder.fit_transform(data.iloc[:,14].values)

# Transform the 'job' column to integers.
data.iloc[:,16]= labelencoder.fit_transform(data.iloc[:,16].values)

# Transform the 'own_telephone' column to integers.
data.iloc[:,18]= labelencoder.fit_transform(data.iloc[:,18].values)

# Transform the 'foreign_worker' column to integers.
data.iloc[:,19]= labelencoder.fit_transform(data.iloc[:,19].values)

# Transform the 'class' column to integers.
data.iloc[:,20]= labelencoder.fit_transform(data.iloc[:,20].values)

# Display unique objects after conversion.
print(data['over_draft'].unique())
print(data['credit_history'].unique())
print(data['purpose'].unique())
print(data['Average_Credit_Balance'].unique())
print(data['employment'].unique())
print(data['personal_status'].unique())
print(data['other_parties'].unique())
print(data['property_magnitude'].unique())
print(data['other_payment_plans'].unique())
print(data['housing'].unique())
print(data['job'].unique())
print(data['own_telephone'].unique())
print(data['foreign_worker'].unique())
print(data['class'].unique())

# Ensure that dataset is void of null values.
print(data.isna().sum())

# Count the number of fraudulent cases on entries with a "bad" status in the "class" feature.
fraudulent_cases = data[data["class"] == 0].shape[0]
print("Number of fraudulent cases:", fraudulent_cases)

# Count the number of non-fraudulent cases on entries with a "good" status in the "class" feature.
non_fraudulent_cases = data[data["class"] == 1].shape[0]
print("Number of non-fraudulent cases:", non_fraudulent_cases)

# Separate the features (X) and the target variable (y)
X = data[["over_draft", "credit_usage", "credit_history", "purpose", "current_balance", "Average_Credit_Balance", "employment", "location", "personal_status",
          "other_parties", "residence_since", "property_magnitude", "cc_age", "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
          "own_telephone", "foreign_worker"]]
y = data["class"]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest classifier.
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set.
y_pred = clf.predict(X_test)

# Evaluate the model.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
average_precision = average_precision_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Average Precision:", average_precision)

# Predict the probabilities of each class for the test set.
y_pred_proba = clf.predict_proba(X_test)

# Create a DataFrame with the individual probabilities of being a fraudster.
probabilities_df = pd.DataFrame(y_pred_proba[:, 1], columns=["Fraud Probability"])
probabilities_df.index = X_test.index

# Sort the DataFrame by fraud probability in descending order.
sorted_probabilities_df = probabilities_df.sort_values(by="Fraud Probability", ascending=False)

# Print the most likely individual(s) to commit credit fraud.
most_likely_fraudsters = sorted_probabilities_df.head(1)
print(most_likely_fraudsters)
