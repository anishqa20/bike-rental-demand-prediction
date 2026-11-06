
# BIKE RENTAL DEMAND CLASSIFICATION USING PYTHON

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score )


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
plt.plot(hourly_rentals.index, hourly_rentals.values,
         marker="o", color="purple", linewidth=2)
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
plt.xticks([1, 2, 3, 4],
           ["Spring", "Summer", "Fall", "Winter"])
plt.show()


# Graph 3: Temperature vs Bike Rentals

plt.figure(figsize=(9, 5))
plt.scatter(df["temp"], df["count"],
            color="teal", alpha=0.4)
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
           ["Clear", "Mist/Cloudy", "Light Rain/Snow",
            "Heavy Rain/Snow"])
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

# Create binary target variable

median_demand = df["count"].median()

y = (df["count"] > median_demand).astype(int)

print("\nDEMAND CLASSIFICATION")
print("Median Rental Count:", median_demand)
print("0 = Low Demand")
print("1 = High Demand")

print("\nCLASS DISTRIBUTION")
print(y.value_counts())


# Handle missing values

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
    X, y, test_size=0.2,random_state=42,stratify=y )

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)


# 8. FEATURE SCALING

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 9. MODEL 1: LOGISTIC REGRESSION

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]


# 10. MODEL 2: KNN CLASSIFICATION

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_pred = knn_model.predict(X_test_scaled)

knn_prob = knn_model.predict_proba(X_test_scaled)[:, 1]


# 11. MODEL 3: DECISION TREE CLASSIFICATION

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=12,
    min_samples_leaf=5
)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_prob = dt_model.predict_proba(X_test)[:, 1]


# 12. DECISION TREE VISUALIZATION

plt.figure(figsize=(20, 10))

plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=["Low Demand", "High Demand"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=8
)

plt.title("Decision Tree Classification Visualization")
plt.show()


# 13. MODEL 4: RANDOM FOREST CLASSIFICATION

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=2,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(X_test)[:, 1]


# 14. MODEL EVALUATION

models = [
    ("Logistic Regression", lr_pred, lr_prob),
    ("KNN Classifier", knn_pred, knn_prob),
    ("Decision Tree", dt_pred, dt_prob),
    ("Random Forest", rf_pred, rf_prob)
]

results = []

for name, pred, prob in models:

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    results.append([name, acc, pre, rec, f1, auc])

    print("\n", name)
    print(classification_report(y_test, pred,
          target_names=["Low Demand", "High Demand"],
          zero_division=0))


# 15. MODEL COMPARISON

results = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision",
    "Recall", "F1 Score", "ROC AUC"
])

print(results)


# 16. ACCURACY COMPARISON GRAPH

plt.figure(figsize=(10, 5))

plt.bar(
    results["Model"],
    results["Accuracy"],
    color=["#9370DB", "#20B2AA", "#FFA500", "#FF69B4"]
)

plt.title("Accuracy Comparison of Classification Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()


# 17. CONFUSION MATRIX GRAPHS

predictions = [
    ("Logistic Regression", lr_pred),
    ("KNN Classifier", knn_pred),
    ("Decision Tree", dt_pred),
    ("Random Forest", rf_pred)
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (model_name, prediction) in zip(axes.flatten(), predictions):

    cm = confusion_matrix(y_test, prediction)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=["Low Demand", "High Demand"],
        yticklabels=["Low Demand", "High Demand"],
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.show()


# 18. ROC CURVE COMPARISON

plt.figure(figsize=(9, 6))

for model_name, prediction, probability in models:

    fpr, tpr, thresholds = roc_curve(y_test, probability)

    auc = roc_auc_score(y_test, probability)

    plt.plot(
        fpr, tpr,
        linewidth=2,
        label=model_name + " (AUC = " + str(round(auc, 3)) + ")"
    )

plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# 19. RANDOM FOREST FEATURE IMPORTANCE

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


# 20. AUTOMATIC FINDINGS

print("\nPROJECT FINDINGS")

print("Busiest Hour:", hourly_rentals.idxmax())

print("Highest Rental Season:", season_rentals.idxmax())

print("Weather Category with Highest Rentals:",
      weather_rentals.idxmax())

print("Highest Average Rentals by Day Type:",
      working_rentals.idxmax())

print("Average Bike Rentals:",
      round(df["count"].mean(), 2))

print("Median Bike Rentals:", median_demand)


# 21. BEST MODEL SELECTION

best_model = results.loc[results["F1 Score"].idxmax()]

print("\nBEST MODEL")

print("Best Model:", best_model["Model"])

print("Accuracy:", best_model["Accuracy"])

print("Precision:", best_model["Precision"])

print("Recall:", best_model["Recall"])

print("F1 Score:", best_model["F1 Score"])

print("ROC AUC:", best_model["ROC AUC"])


print("\nPROJECT EXECUTION COMPLETED!")
