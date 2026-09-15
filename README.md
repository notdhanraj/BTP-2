# Soil Nail Wall Stability Prediction Using Machine Learning

## Project Overview

This project develops a **machine learning-based Factor of Safety (FoS) prediction system for soil-nailed excavation walls**.

The workflow combines geotechnical design parameters with supervised regression models to estimate the **global stability Factor of Safety** of a reinforced excavation. A tuned **Random Forest Regressor** is used as the prediction model and is integrated into a **Streamlit web application** for interactive stability assessment.

The system allows users to vary excavation geometry, soil-nail configuration, and geotechnical parameters and obtain an estimated Factor of Safety along with a stability interpretation.

---

## Objectives

- Develop a machine learning model for predicting **Factor of Safety (FoS)**.
- Preprocess and encode geotechnical design data for regression modelling.
- Compare multiple regression algorithms.
- Tune the Random Forest model using **GridSearchCV**.
- Evaluate models using **R² and RMSE**.
- Save trained models using `joblib`.
- Build an interactive **Streamlit-based prediction interface**.
- Provide an engineering-oriented interpretation of predicted FoS values.

---

# Problem Statement

Stability assessment of reinforced excavations involves multiple interacting parameters such as:

- Excavation height
- Slope angle
- Back-slope angle
- Soil nail inclination
- Nail length-to-height ratio
- Vertical nail spacing
- Soil cohesion
- Soil friction angle

Traditional stability analysis can require detailed numerical or analytical calculations. This project explores a **data-driven machine learning approach** in which historical/design data are used to learn the relationship between these parameters and the resulting Factor of Safety.

The trained model can then provide a rapid FoS estimate for a new combination of design parameters.

---

# Dataset

The project uses a geotechnical dataset containing design and soil parameters with **`FactorOfSafety` as the target variable**.

### Input Features

| Feature | Description |
|---|---|
| `BackSlopeAngle` | Back-slope angle |
| `SlopeAngle` | Excavation slope angle |
| `NailInclination` | Soil nail inclination |
| `L/H` | Nail length-to-excavation-height ratio |
| `SpacingVertical` | Vertical spacing between nails |
| `Height` | Excavation height |
| `Cohesion` | Soil cohesion |
| `FrictionAngle` | Soil friction angle |

### Target

```text
FactorOfSafety
```

The preprocessing pipeline removes observations without a Factor of Safety value and fills remaining numerical missing values using the corresponding column means.

Categorical variables, if present in the source data, are converted using one-hot encoding.

---

# Machine Learning Workflow

```text
Geotechnical Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ├── Remove missing FoS
        ├── Fill numerical missing values
        └── One-Hot Encoding
        │
        ▼
Feature / Target Separation
        │
        ▼
Train-Test Split
        │
        ├───────────────┐
        ▼               ▼
Training Data       Test Data
        │
        ▼
Random Forest Hyperparameter Tuning
        │
        ▼
GridSearchCV
        │
        ▼
Best Random Forest
        │
        ├───────────────┐
        ▼               ▼
Model Comparison   Test Prediction
        │               │
        ▼               ▼
R² / RMSE         Actual vs Predicted
        │
        ▼
Saved Model
        │
        ▼
Streamlit Application
        │
        ▼
Predicted Factor of Safety
```

---

# Data Preprocessing

## Missing Values

Rows without a valid `FactorOfSafety` value are removed because the target is required for supervised learning.

Remaining numerical missing values are replaced using the mean of the corresponding numerical feature.

## Categorical Encoding

Categorical input variables are converted into numerical representations using:

```python
pd.get_dummies(X, drop_first=True)
```

This allows categorical information to be incorporated into the regression models.

## Train-Test Split

The dataset is divided into:

- **80% training data**
- **20% test data**
- `random_state = 42`

The test set is kept separate for final model evaluation.

---

# Models Evaluated

The project compares four regression approaches:

### 1. Linear Regression

Provides a simple baseline by modelling the relationship between the input variables and Factor of Safety using a linear function.

### 2. Decision Tree Regressor

Captures nonlinear relationships through recursive feature-based splits.

### 3. Random Forest Regressor

Combines multiple decision trees to improve predictive robustness and capture nonlinear interactions between geotechnical parameters.

### 4. XGBoost Regressor

Uses gradient-boosted decision trees to model complex nonlinear relationships in the data.

---

# Random Forest Hyperparameter Tuning

The Random Forest model is tuned using **GridSearchCV with 3-fold cross-validation**.

The search explores:

```text
n_estimators:
100, 200

max_depth:
None, 10, 20

min_samples_split:
2, 5

min_samples_leaf:
1, 2
```

The best configuration is selected based on **R² scoring**.

This tuned Random Forest model is subsequently used as the primary prediction model in the Streamlit application.

---

# Model Evaluation

Two primary regression metrics are used.

## R² Score

R² measures the proportion of variance in Factor of Safety explained by the model.

```text
Higher R² → Better explanatory performance
```

## Root Mean Squared Error

RMSE measures the typical magnitude of prediction errors while giving greater weight to larger errors.


```text
Lower RMSE → Better prediction accuracy
```

For each model, actual and predicted Factor of Safety values are also visualized using scatter plots.

---

# Model Comparison

The project trains and evaluates all four models on the test set:

| Model | Evaluation |
|---|---|
| Linear Regression | R² + RMSE |
| Decision Tree Regressor | R² + RMSE |
| **Random Forest Tuned** | **R² + RMSE** |
| XGBoost | R² + RMSE |

The **tuned Random Forest** is selected for deployment in the prediction application.

> **Note:** The training notebook does not retain numerical model-evaluation output in the distributed project files, so specific R²/RMSE values are intentionally not claimed here.

---

# Streamlit Prediction Application

The trained Random Forest model is integrated into a **Streamlit web interface** called:

## Soil Nail Wall Stability Predictor

The application organizes the inputs into three sections.

### 1. Excavation Geometry

Users can specify:

- Excavation Height
- Slope Angle
- Back Slope Angle

### 2. Nail Configuration

Users can specify:

- Length/Height Ratio
- Nail Inclination
- Vertical Spacing

### 3. Soil Properties

Users can specify:

- Cohesion
- Friction Angle

After entering the parameters, the application predicts the Factor of Safety.

---

# Stability Interpretation

The application provides an engineering-oriented interpretation of the predicted Factor of Safety:

| Predicted FoS | Interpretation |
|---:|---|
| **FoS < 1.0** | Failure |
| **1.0 ≤ FoS < 1.3** | Marginal / Unsafe |
| **1.3 ≤ FoS < 1.5** | Stable |
| **FoS ≥ 1.5** | Highly Stable |

The application also displays an explanatory message corresponding to each stability category.

> These thresholds are implemented as application interpretation rules and should not be treated as a substitute for project-specific geotechnical design standards or professional engineering assessment.

---

# Model Deployment

The trained models are serialized using **Joblib**:

```text
Linear_Regression_model.joblib
Decision_Tree_model.joblib
Random_Forest_Tuned_model.joblib
XGBoost_model.joblib
```

The Streamlit application loads:

```text
Random_Forest_Tuned_model.joblib
```

and uses it to generate real-time FoS predictions.

---

# Application Workflow

```text
User Inputs
    │
    ├── Excavation Geometry
    ├── Nail Configuration
    └── Soil Properties
          │
          ▼
   Input DataFrame
          │
          ▼
   Trained Random Forest
          │
          ▼
   Predicted Factor of Safety
          │
          ▼
   Stability Classification
          │
     ┌────┼───────────────┐
     ▼    ▼               ▼
 Failure Marginal       Stable
                       / Highly Stable
```

---

# Technologies Used

| Area | Technologies |
|---|---|
| Programming | **Python** |
| Data Processing | **Pandas, NumPy** |
| Visualization | **Matplotlib, Seaborn** |
| Machine Learning | **Scikit-learn** |
| Gradient Boosting | **XGBoost** |
| Hyperparameter Tuning | **GridSearchCV** |
| Model Serialization | **Joblib** |
| Web Application | **Streamlit** |
| Development | **Jupyter Notebook / Google Colab** |

---

# Project Structure

```text
BTP-2/
│
└── BTP_2/
    ├── Model_Training.ipynb
    └── app.py
```

The notebook contains the complete model-training and evaluation workflow, while `app.py` contains the Streamlit prediction interface.

---

# Key Takeaways

- Developed a **machine learning-based Factor of Safety prediction system** for soil-nailed excavation walls.
- Used **8 geotechnical and excavation design parameters** as model inputs.
- Implemented data cleaning, missing-value handling, and categorical encoding.
- Compared **Linear Regression, Decision Tree, Random Forest, and XGBoost** regression models.
- Used **GridSearchCV with 3-fold cross-validation** to tune the Random Forest model.
- Evaluated model performance using **R² and RMSE**.
- Serialized trained models with **Joblib** for deployment.
- Built an interactive **Streamlit Soil Nail Wall Stability Predictor**.
- Added automated FoS-based stability interpretation to make model predictions easier to understand.
- Created an end-to-end workflow connecting **geotechnical data → machine learning → deployed prediction interface**.
