# Bike Rental Demand Classification Using Python

## Project Overview
This project focuses on analyzing bike rental demand and classifying it into two categories: **Low Demand** and **High Demand** using Python and Machine Learning.

The project uses historical bike rental data to identify patterns and understand how factors such as time, season, temperature, weather, and working days influence rental demand.

## Objectives
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Visualize bike rental patterns.
- Convert rental counts into Low Demand and High Demand categories.
- Build and compare different Machine Learning classification models.
- Evaluate model performance using classification metrics.

## Dataset
- **Source:** Kaggle – Bike Sharing Demand
- **File used:** `train.csv`
- **Rows:** 10,886 (before cleaning)
- **Target variable:** `count`

The target variable is converted into:
- `0` – Low Demand
- `1` – High Demand

The median rental count is used as the threshold for classification.

## Exploratory Data Analysis
The project includes visualizations for:
- Average bike rentals by hour
- Average bike rentals by season
- Temperature vs. bike rentals
- Average bike rentals by weather
- Working day vs. non-working day
- Correlation heatmap

## Machine Learning Models
Four classification algorithms are implemented:

1. Logistic Regression
2. K-Nearest Neighbors (KNN) Classifier
3. Decision Tree Classifier
4. Random Forest Classifier

## Model Evaluation
The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Additional visualizations include:
- Accuracy comparison graph
- Confusion matrices
- ROC curve comparison
- Decision Tree visualization
- Random Forest feature importance

The best-performing model is selected based on the highest F1 Score.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## How to Run the Project

1. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Download the Bike Sharing Demand dataset from Kaggle.

3. Keep `train.csv` in the same folder as the Python file.

4. Run the Python program.

## Project Outcome
This project demonstrates how historical bike rental data can be analyzed and used to classify rental demand into low and high categories.

It provides practical experience with data preprocessing, visualization, classification algorithms, and model evaluation.

## Author
**Anishqa Dwivedi**  
B.Tech CSE  
Lovely Professional University
