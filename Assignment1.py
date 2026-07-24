# Step 1: Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Step 2: Load the Heart Disease Dataset

df = pd.read_csv("heart.csv")

# Step 3: Display the Dataset

print(df.head())
print(df.info())
print(df.describe())

# Step 4: Identify Missing Values

print(df.isnull().sum())

# Step 5: Replace Missing Values

# Fill numerical columns with Mean
df = df.fillna(df.mean(numeric_only=True))

# Fill categorical columns with Mode
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nAfter replacing missing values:\n")
print(df.isnull().sum())

# Step 6: Calculate Variance

print("Variance:\n")
print(df.var(numeric_only=True))

# Calculate Standard Deviation

print("\nStandard Deviation:\n")
print(df.std(numeric_only=True))

# Calculate Covariance Matrix

print("\nCovariance Matrix:\n")
print(df.cov(numeric_only=True))

# Calculate Correlation Matrix

print("\nCorrelation Matrix:\n")
print(df.corr(numeric_only=True))

# Step 7: Normalize Numerical Columns

scaler = MinMaxScaler()

num_cols = df.select_dtypes(include="number").columns

df_normalized = df.copy()

df_normalized[num_cols] = scaler.fit_transform(df[num_cols])

print("\nNormalized Data")
print(df_normalized.head())

# Step 8: Standardize Numerical Columns

scaler = StandardScaler()

df_standardized = df.copy()

df_standardized[num_cols] = scaler.fit_transform(df[num_cols])

print("\nStandardized Data")
print(df_standardized.head())

# Step 9: Perform Discretization

df["Age_Category"] = pd.cut(
    df["age"],
    bins=[20,40,50,60,80],
    labels=["Young","Adult","Middle","Senior"]
)

print(df[["age","Age_Category"]].head(10))

# Step 10: Create Histogram

plt.figure(figsize=(8,5))
plt.hist(df["chol"], bins=20, edgecolor="black")
plt.title("Histogram of Cholesterol")
plt.xlabel("Cholesterol")
plt.ylabel("Frequency")
plt.show()

# Step 11: Create Boxplot

plt.figure(figsize=(6,5))
plt.boxplot(df["chol"])
plt.title("Boxplot of Cholesterol")
plt.ylabel("Cholesterol")
plt.show()

# Step 12: Detect Outliers Using IQR

Q1 = df["chol"].quantile(0.25)
Q3 = df["chol"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - (1.5 * IQR)
upper_bound = Q3 + (1.5 * IQR)

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# Step 13: Remove Outliers

df_no_outliers = df[
    (df["chol"] >= lower_bound) &
    (df["chol"] <= upper_bound)
]

print("Original Shape:", df.shape)
print("After Removing Outliers:", df_no_outliers.shape)

# Save Cleaned Dataset

df_no_outliers.to_csv("heart_cleaned.csv", index=False)

print("\nheart_cleaned.csv saved successfully.")

# Step 14: Create Scatter Plot

plt.figure(figsize=(8,5))
plt.scatter(df["age"], df["chol"])
plt.title("Age vs Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.show()

# Step 15: Create Quantile (Q-Q) Plot

stats.probplot(df_no_outliers["chol"], dist="norm", plot=plt)

plt.title("Q-Q Plot of Cholesterol")
plt.show()

# Step 16: Interpretation

print("\nInterpretation")
print("• Missing values were identified and replaced.")
print("• Variance, standard deviation, covariance and correlation were calculated.")
print("• Data was normalized and standardized.")
print("• Age was discretized into different categories.")
print("• Histogram and boxplot of Cholesterol were created.")
print("• Outliers were detected and removed using the IQR method.")
print("• Scatter plot showed the relationship between Age and Cholesterol.")
print("• Q-Q plot checked whether Cholesterol follows a normal distribution.")
print("• Cleaned dataset was saved as heart_cleaned.csv.")