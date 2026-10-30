
# IRIS FLOWER CLASSIFICATION USING MACHINE LEARNING

# Step 1: Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Step 2: Load Iris dataset
iris = load_iris()

# Step 3: Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target

# Display dataset
print("IRIS FLOWER DATASET")
print(df.head())

# Step 4: Dataset information
print("\nDataset Information:")
print(df.info())

# Step 5: Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Step 6: Statistical description
print("\nStatistical Description:")
print(df.describe())

# Step 7: Separate input and output
X = iris.data
y = iris.target

# Step 8: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Step 9: Create Gaussian Naive Bayes model
model = GaussianNB()

# Step 10: Train the model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Step 11: Predict test data
y_pred = model.predict(X_test)

print("\nPredicted Values:")
print(y_pred)

print("\nActual Values:")
print(y_test)

# Step 12: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

# Step 13: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 14: Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

# Step 15: Visualize confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.title("Iris Flower Classification")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.show()

# Step 16: Visualize flower measurements
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x='sepal length (cm)',
    y='petal length (cm)',
    hue='species',
    palette='Set1'
)

plt.title("Iris Flower Measurements")
plt.show()

# Step 17: Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nNEW FLOWER PREDICTION:")
print("Sepal Length:", sample[0][0])
print("Sepal Width:", sample[0][1])
print("Petal Length:", sample[0][2])
print("Petal Width:", sample[0][3])

print("Predicted Flower Species:",
      iris.target_names[prediction[0]])

# Step 18: Final Result
print("\nIris Flower Classification Completed Successfully!")
