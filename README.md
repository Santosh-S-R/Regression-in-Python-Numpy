# Flight Price Prediction – Regression Analysis

This project develops a regression framework to model and predict airline ticket prices using a structured dataset of 10,683 flight records.

The dataset contains 10 input features, including airline, route, journey date, departure and arrival times, duration, total stops, and additional flight information, with ticket price as the target variable.

---

## Objective

The objective is to transform raw categorical and time-based flight data into a fully numeric feature matrix suitable for regression modelling, while ensuring:

- Removal of duplicate records  
- Proper handling of missing values  
- Structured categorical encoding  
- Clear and reproducible feature engineering  

---

## Data Preparation

The notebook performs systematic preprocessing steps:

- Removal of duplicate rows  
- Handling and inspection of null values  
- Column-wise exploratory inspection  
- Conversion of categorical variables using:
  - One-hot encoding  
  - Ordinal encoding where ordering is meaningful  
- Custom utilities for encoding and transformation tracking  

The encoding process is implemented manually to maintain transparency and full control over feature construction.

---

## Models Implemented

All models were implemented from scratch using NumPy:

- Ordinary Least Squares (OLS)  
- Ridge Regression  
- Lasso-style regularisation  
- Polynomial Regression  
- Kernel Ridge Regression (RBF kernel)  
- A small Neural Network (1 hidden layer, tanh activation)  

Model selection was performed using **10-fold cross-validation**, with final evaluation on a held-out test set.  
Primary metric: **R²** (with MSE also reported).

---

## Results

Nonlinear models significantly outperformed linear baselines.

**Best performing model:**  
Kernel Ridge Regression (RBF kernel)

### Cross-Validation (10-fold)

- **RBF-KRR:** R² = **0.8337 ± 0.0340**  
- Polynomial (degree 4): R² = 0.7950  
- OLS: R² = 0.7504  

### Test Set Performance

- **RBF-KRR:** R² = **0.8433**, MSE = 3,055,854  
- Polynomial (degree 4): R² = 0.7960  
- OLS: R² = 0.7635  

Results show that performance gains were driven primarily by improved feature representation and nonlinear modelling, rather than increasing model complexity alone.

---

## Key Insights

- Number of stops, duration, and major hub indicators show strong correlation with price.
- Linear models provide interpretability but are limited in capturing interactions.
- The RBF kernel captures nonlinear structure across route, timing, and airline features.
- Careful feature engineering had a larger impact than switching between similar linear models.

---

## Implementation Notes

- Implemented entirely in NumPy (models coded from first principles)  
- Structured feature blocks for debugging and interpretability  
- Consistent train/test split across raw and log-transformed targets  
- Uniform hyperparameter tuning framework  

---

## Conclusion

The strongest performance was achieved using RBF Kernel Ridge Regression, confirming that nonlinear interactions play a central role in airline ticket pricing.

The project highlights a core principle in applied machine learning:  
**representation quality often matters more than model sophistication.**
