
# BIKE RENTAL DEMAND PREDICTION AND ANALYSIS USING PYTHON

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# 2. LOAD DATASET

df = pd.read_csv("train.csv")

print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()

print("\nSTATISTICAL SUMMARY")
print(df.describe())


# 3. DATA CLEANING

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Convert datetime column
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

# Remove rows with missing datetime or target
df = df.dropna(subset=["datetime", "count"])

# Check invalid values
print("\nNEGATIVE RENTAL COUNTS")
print((df["count"] < 0).sum())

# Remove invalid rental counts
df = df[df["count"] >= 0]

# Keep valid season and weather categories
df = df[df["season"].isin([1, 2, 3, 4])]
df = df[df["weather"].isin([1, 2, 3, 4])]

# Reset index
df = df.reset_index(drop=True)

print("\nDATASET AFTER CLEANING")
print(df.shape)


# 4. FEATURE EXTRACTION

df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["day"] = df["datetime"].dt.day
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.dayofweek

df = df.sort_values("datetime")

print("\nDATASET WITH NEW FEATURES")
print(df.head())



# 5. EXPLORATORY DATA ANALYSIS

# Graph 1: Average Bike Rentals by Hour

hourly_rentals = df.groupby("hour")["count"].mean()

plt.figure(figsize=(10, 5))
plt.plot(hourly_rentals.index, hourly_rentals.values, marker="o",
         color="purple", linewidth=2)
plt.fill_between(hourly_rentals.index, hourly_rentals.values,
                 color="purple", alpha=0.15)
plt.title("Average Bike Rentals by Hour")
plt.xlabel("Hour of the Day")
plt.ylabel("Average Rentals")
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.show()


# Graph 2: Average Bike Rentals by Season

season_rentals = df.groupby("season")["count"].mean()

plt.figure(figsize=(8, 5))
plt.bar(season_rentals.index, season_rentals.values,
        color=["#9370DB", "#20B2AA", "#FFA500", "#FF69B4"])
plt.title("Average Bike Rentals by Season")
plt.xlabel("Season")
plt.ylabel("Average Rentals")
plt.xticks([1, 2, 3, 4], ["Spring", "Summer", "Fall", "Winter"])
plt.show()


# Graph 3: Temperature vs Bike Rentals

plt.figure(figsize=(9, 5))
plt.scatter(df["temp"], df["count"], color="teal", alpha=0.4)
plt.title("Temperature vs Bike Rentals")
plt.xlabel("Temperature")
plt.ylabel("Bike Rentals")
plt.grid(alpha=0.3)
plt.show()


# Graph 4: Bike Rentals by Weather

weather_rentals = df.groupby("weather")["count"].mean()

plt.figure(figsize=(8, 5))
plt.bar(weather_rentals.index, weather_rentals.values,
        color=["#6495ED", "#FFB347", "#90EE90", "#FA8072"])
plt.title("Average Bike Rentals by Weather")
plt.xlabel("Weather Condition")
plt.ylabel("Average Rentals")
plt.xticks([1, 2, 3, 4],
           ["Clear", "Mist/Cloudy", "Light Rain/Snow", "Heavy Rain/Snow"])
plt.show()


# Graph 5: Working Day vs Non-Working Day

working_rentals = df.groupby("workingday")["count"].mean()

plt.figure(figsize=(7, 5))
plt.bar(working_rentals.index, working_rentals.values,
        color=["#9370DB", "#FFD700"])
plt.title("Bike Rentals: Working Day vs Non-Working Day")
plt.xlabel("Day Type")
plt.ylabel("Average Rentals")
plt.xticks([0, 1], ["Non-Working Day", "Working Day"])
plt.show()


# Graph 6: Correlation Heatmap

plt.figure(figsize=(11, 7))
correlation = df.select_dtypes(include=np.number).corr()
sns.heatmap(correlation, annot=True, cmap="Spectral", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 6. FEATURE ENGINEERING

X = df.drop(["datetime", "casual", "registered", "count"], axis=1)
y = df["count"]

# Handle missing values in input features
for column in X.columns:
    if X[column].isnull().sum() > 0:
        if X[column].dtype == "object":
            X[column] = X[column].fillna(X[column].mode()[0])
        else:
            X[column] = X[column].fillna(X[column].median())

# Convert categorical columns into numerical columns
X = pd.get_dummies(X, columns=["season", "weather"], dtype=int)

print("\nFEATURE COLUMNS")
print(X.columns.tolist())

print("\nFEATURE DATA")
print(X.head())

# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 )

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

# 8. FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 9. MODEL 1: LINEAR REGRESSION

lr_model = LinearRegression()

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_mse = mean_squared_error(y_test, lr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2 = r2_score(y_test, lr_pred)

print("\nLINEAR REGRESSION RESULTS")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("RMSE:", lr_rmse)
print("R2 Score:", lr_r2)


# 10. MODEL 2: KNN REGRESSION

knn_model = KNeighborsRegressor(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_mae = mean_absolute_error(y_test, knn_pred)
knn_mse = mean_squared_error(y_test, knn_pred)
knn_rmse = np.sqrt(knn_mse)
knn_r2 = r2_score(y_test, knn_pred)

print("\nKNN REGRESSION RESULTS")
print("MAE:", knn_mae)
print("MSE:", knn_mse)
print("RMSE:", knn_rmse)
print("R2 Score:", knn_r2)


# 11. MODEL 3: DECISION TREE REGRESSION

dt_model = DecisionTreeRegressor(
    random_state=42,
    max_depth=12,
    min_samples_leaf=5
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_mse = mean_squared_error(y_test, dt_pred)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, dt_pred)

print("\nDECISION TREE RESULTS")
print("MAE:", dt_mae)
print("MSE:", dt_mse)
print("RMSE:", dt_rmse)
print("R2 Score:", dt_r2)


# 12. COLOURFUL DECISION TREE VISUALIZATION

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.show()


# 13. MODEL 4: RANDOM FOREST REGRESSION

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRANDOM FOREST RESULTS")
print("MAE:", rf_mae)
print("MSE:", rf_mse)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)


# 14. MODEL COMPARISON

results = pd.DataFrame({

    "Model": ["Linear Regression","KNN Regression","Decision Tree","Random Forest"],

    "MAE": [lr_mae,knn_mae,dt_mae,rf_mae],

    "MSE": [lr_mse,knn_mse,dt_mse,rf_mse],

    "RMSE": [lr_rmse,knn_rmse,dt_rmse,rf_rmse],

    "R2 Score": [lr_r2,knn_r2,dt_r2,rf_r2]})

print("\nMODEL COMPARISON")
print(results)


# 15. COLOURFUL RMSE COMPARISON GRAPH

plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["RMSE"],
        color=["#9370DB", "#20B2AA", "#FFA500", "#FF69B4"])
plt.title("RMSE Comparison of Machine Learning Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("RMSE")
plt.xticks(rotation=15)
plt.show()

# 16. COLOURFUL ACTUAL VS PREDICTED GRAPHS

predictions = [
    ("Linear Regression", lr_pred, "#3498DB"),
    ("KNN Regression", knn_pred, "#E67E22"),
    ("Decision Tree", dt_pred, "#2ECC71"),
    ("Random Forest", rf_pred, "#9B59B6")
]

for model_name, prediction, graph_color in predictions:

    plt.figure(figsize=(7, 5))

    plt.scatter(
        y_test,
        prediction,
        color=graph_color,
        alpha=0.55,
        edgecolors="white"
    )

    minimum = min(y_test.min(), prediction.min())
    maximum = max(y_test.max(), prediction.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="#E74C3C",
        linestyle="--",
        linewidth=2,
        label="Perfect Prediction"
    )

    plt.title("Actual vs Predicted - " + model_name)
    plt.xlabel("Actual Rentals")
    plt.ylabel("Predicted Rentals")
    plt.legend()
    plt.tight_layout()
    plt.show()


# 17. COLOURFUL RANDOM FOREST FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance", ascending=False
)

print("\nRANDOM FOREST FEATURE IMPORTANCE")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=feature_importance.head(10),
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

# 18. AUTOMATIC FINDINGS

print("\n====== PROJECT FINDINGS =======")

print("Busiest Hour:", hourly_rentals.idxmax())

print("Highest Rental Season:", season_rentals.idxmax())

print("Weather Category with Highest Rentals:",
      weather_rentals.idxmax())

print("Highest Average Rentals by Day Type:",
      working_rentals.idxmax())

print("Average Bike Rentals:",
      round(df["count"].mean(), 2))


# 19. BEST MODEL SELECTION

best_model = results.loc[results["RMSE"].idxmin()]

print("\n====== BEST MODEL ======")

print("Best Model:", best_model["Model"])

print("MAE:", best_model["MAE"])

print("MSE:", best_model["MSE"])

print("RMSE:", best_model["RMSE"])

print("R2 Score:", best_model["R2 Score"])

print("\nPROJECT EXECUTION COMPLETED!")
