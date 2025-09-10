
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing(as_frame=True)
df = housing.frame



# View first 5 rows
print(df.head())

# Summary statistics
print(df.describe())

# Check missing values
print(df.isnull().sum())

# --- Visualizations ---

# 1. Histogram of target variable
plt.figure(figsize=(8,5))
sns.histplot(df['MedHouseVal'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of House Prices')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.show()

# 2. Pairplot of some features vs target
sns.pairplot(df[['MedInc', 'AveRooms', 'HouseAge', 'MedHouseVal']])
plt.show()

# 3. Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

# 4. Scatter plot: Median Income vs House Price
plt.figure(figsize=(8,5))
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df)
plt.title('Median Income vs House Price')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.show()

# --- Prepare data for modeling ---
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model training ---
# Option 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Option 2: Random Forest Regressor (better performance)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# --- Evaluation ---
def evaluate_model(y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} -> Mean Squared Error: {mse:.3f}, R² Score: {r2:.3f}")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# --- Residual plot for Random Forest ---
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_test - y_pred_rf)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals: Random Forest')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.show()

# --- Feature Importance ---
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importances.sort_values().plot(kind='barh', color='lightgreen')
plt.title('Feature Importance - Random Forest')
plt.show()
