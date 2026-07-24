# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("Housing.csv")

# Display Dataset
print("First Five Records")
print(df.head())
print("\nDataset Information")
print(df.info())
print("\nMissing Values")
print(df.isnull().sum())
print("\nStatistical Summary")
print(df.describe())

# Features and Target
X = df[['area', 'bedrooms', 'bathrooms', 'parking']]
y = df['price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Create Linear Regression Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction on Training Data
y_train_pred = model.predict(X_train)

# Prediction on Testing Data
y_pred = model.predict(X_test)

# Model Evaluation
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)
print("\nModel Evaluation")
print("Training R2 Score :", train_r2)
print("Testing R2 Score :", test_r2)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", test_r2)

# Check Underfitting and Overfitting
print("\nModel Check")

# Underfitting
if train_r2 < 0.70 and test_r2 < 0.70:
    print("UNDERFITTING DETECTED")
    print("The model is too simple and has high error.")

# Overfitting
elif train_r2 > 0.85 and (train_r2 - test_r2) > 0.10:
    print("OVERFITTING DETECTED")
    print("Applying Lasso Regression...")

    # Lasso Regression
    lasso_model = Lasso(alpha=0.1)

    # Train Lasso Model
    lasso_model.fit(X_train, y_train)

    # Lasso Prediction
    y_pred = lasso_model.predict(X_test)

    # Lasso Evaluation
    print("\nLasso Regression Evaluation")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("MSE :", mean_squared_error(y_test, y_pred))
    print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("R2 Score :", r2_score(y_test, y_pred))

    # Use Lasso as Final Model
    model = lasso_model

# Good Fit
else:
    print("GOOD FIT")
    print("The model has a good balance between training and testing performance.")

# Actual vs Predicted Values
result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
print("\nActual vs Predicted")
print(result.head(15))

# Coefficients
print("\nIntercept")
print(model.intercept_)
print("\nCoefficients")
coeff = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print(coeff)

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# Prediction for New House
new_house = pd.DataFrame([[2500, 4, 3, 2]], columns=['area', 'bedrooms', 'bathrooms', 'parking'])
prediction = model.predict(new_house)
print("\nPredicted Price for New House :", prediction[0])