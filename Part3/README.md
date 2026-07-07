# Part 3 – Advanced Modeling: Ensembles, Hyperparameter Tuning and Machine Learning Pipeline

## Student Information

**Name:** Debojit Lahiri

---

# Objective

The objective of Part 3 is to evaluate multiple advanced machine learning models for binary house price classification using the cleaned Ames Housing dataset. This part focuses on ensemble learning, hyperparameter tuning, model comparison using cross-validation, feature importance analysis, pipeline creation, model serialization, and learning curve analysis.

---

# Dataset

The project uses the cleaned Ames Housing dataset generated in Part 1.

- Dataset: Ames Housing Dataset
- Number of rows: 1460
- Target (Regression): SalePrice
- Target (Classification): HighPrice

HighPrice is created by converting SalePrice into a binary target using the median sale price:

```python
HighPrice = (SalePrice > SalePrice.median()).astype(int)
```

---

# 1. Decision Tree Baseline

A Decision Tree classifier was trained using the default parameters.

### Results

| Metric | Value |
|--------|-------|
| Training Accuracy | **1.0000** |
| Testing Accuracy | **0.8973** |

### Interpretation

The baseline Decision Tree perfectly classified the training data but performed worse on the testing data. This indicates **overfitting**. Decision Trees are considered **high variance models** because they greedily split the data at each node without revisiting previous decisions, making them highly sensitive to the training dataset.

---

# 2. Controlled Decision Tree

A second Decision Tree was trained using:

- max_depth = 5
- min_samples_split = 20

### Results

| Metric | Value |
|--------|-------|
| Training Accuracy | **0.9272** |
| Testing Accuracy | **0.8767** |

### Interpretation

Limiting the tree depth reduced overfitting by preventing the model from learning noise in the training data. Although the testing accuracy decreased slightly, the difference between training and testing accuracy became much smaller.

### Parameter Explanation

**max_depth**

Limits the maximum depth of the tree. Smaller trees reduce variance but may introduce additional bias.

**min_samples_split**

Prevents splitting nodes that contain very few samples, reducing noisy decisions and improving generalization.

---

# 3. Gini vs Entropy

Two Decision Trees were trained using different splitting criteria.

### Results

| Criterion | Test Accuracy |
|-----------|--------------|
| Gini | **0.8938** |
| Entropy | **0.9007** |

### Formula

Gini Impurity

\[
Gini = 1-\sum p_i^2
\]

Entropy

\[
Entropy = -\sum p_i \log_2(p_i)
\]

A Gini value of **0** indicates that every sample in the node belongs to a single class.

---

# 4. Random Forest

Random Forest was trained using:

- n_estimators = 100
- max_depth = 10
- random_state = 42

### Results

| Metric | Value |
|--------|-------|
| Training Accuracy | **0.9957** |
| Testing Accuracy | **0.9452** |
| ROC-AUC | **0.9844** |

## Top Five Important Features

| Feature | Importance |
|---------|-----------|
| GrLivArea | 0.0839 |
| YearBuilt | 0.0605 |
| OverallQual | 0.0593 |
| FullBath | 0.0517 |
| TotalBsmtSF | 0.0377 |

### Feature Importance

Random Forest computes feature importance using the average reduction in **Gini impurity** contributed by each feature across all trees. Unlike linear regression coefficients, feature importance does not indicate whether a relationship is positive or negative; it only measures how useful a feature is for prediction.

### Bagging

Random Forest uses **Bootstrap Aggregating (Bagging)**.

Each tree is trained on a random sample of the training data selected **with replacement**. At every split, only a random subset of features is considered. Averaging predictions from many independent trees reduces variance and improves generalization compared to a single Decision Tree.

---

# 5. Gradient Boosting

Gradient Boosting was trained using:

- n_estimators = 100
- learning_rate = 0.1
- max_depth = 3

### Results

| Metric | Value |
|--------|-------|
| Training Accuracy | **0.9889** |
| Testing Accuracy | **0.9452** |
| ROC-AUC | **0.9857** |

Gradient Boosting slightly outperformed Random Forest in ROC-AUC while maintaining identical testing accuracy.

---

# 6. Feature Ablation Study

The five least important features identified by the Random Forest were removed.

### Removed Features

- Electrical_Mix
- GarageQual_Po
- SaleType_Con
- MiscFeature_Othr
- MiscFeature_TenC

### Results

| Model | ROC-AUC |
|-------|---------|
| Full Model | **0.9845** |
| Reduced Model | **0.9840** |

### Interpretation

The reduced model achieved nearly identical performance, indicating that the removed features contributed very little predictive information. Removing such features simplifies deployment while maintaining almost the same predictive performance.

---

# 7. Cross Validation

Five-fold Stratified Cross Validation was performed using ROC-AUC.

| Model | Mean AUC | Std AUC |
|------|---------|--------|
| Logistic Regression | **0.9658** | 0.0119 |
| Controlled Decision Tree | **0.9210** | 0.0127 |
| Random Forest | **0.9803** | 0.0044 |
| Gradient Boosting | **0.9780** | 0.0045 |

### Interpretation

Cross-validation provides a more reliable estimate of generalization performance because every sample is used for both training and validation across multiple folds. Random Forest achieved the highest average ROC-AUC with the smallest standard deviation, indicating both high accuracy and stable performance.

---

# 8. Hyperparameter Tuning

GridSearchCV was used with a machine learning pipeline consisting of:

- Median Imputer
- StandardScaler
- RandomForestClassifier

### Hyperparameter Grid

- n_estimators = {50, 100, 200}
- max_depth = {5, 10, None}
- min_samples_leaf = {1, 5}

Total configurations evaluated:

3 × 3 × 2 = **18**

Using 5-fold cross-validation:

18 × 5 = **90 model fits**

### Best Parameters

- max_depth = 10
- min_samples_leaf = 1
- n_estimators = 200

### Best Cross Validation Score

**0.9783**

### Grid Search vs Randomized Search

Grid Search evaluates every possible parameter combination and therefore guarantees finding the best configuration within the search space. Randomized Search evaluates only a random subset of combinations, making it faster but less exhaustive.

---

# 9. Learning Curve

| Training Fraction | Training AUC | Testing AUC |
|------------------|-------------|------------|
| 20% | 1.0000 | 0.9798 |
| 40% | 1.0000 | 0.9825 |
| 60% | 0.99999 | 0.9844 |
| 80% | 0.99999 | 0.9842 |
| 100% | 0.99994 | 0.9844 |

### Interpretation

Training AUC decreased very slightly as more training data was used, which is expected because larger datasets reduce overfitting. Test AUC increased initially and then plateaued near 0.984, indicating that the model is approaching its maximum performance. This suggests the current model is more **capacity-limited** than **data-limited**, meaning additional data alone is unlikely to produce substantial improvements.

---

# Model Serialization

The best pipeline was saved using Joblib.

```
joblib.dump(best_pipeline, "best_model.pkl")
```

The model was successfully reloaded using:

```
loaded_model = joblib.load("best_model.pkl")
```

Predictions on two test samples were successfully generated without errors.

---

# Final Model Comparison

| Model | Test AUC |
|------|---------|
| Logistic Regression | 0.9663 |
| Decision Tree | 0.9007 |
| Random Forest | 0.9844 |
| Gradient Boosting | **0.9857** |

---

# Final Recommendation

Among all evaluated models, **Gradient Boosting** achieved the highest test ROC-AUC, while **Random Forest** demonstrated the best cross-validation stability. Both ensemble methods substantially outperformed the Decision Tree and Logistic Regression models. Considering its excellent predictive performance and robustness, Gradient Boosting is recommended for deployment on this dataset. If interpretability and stability across different samples are prioritized, Random Forest is also an excellent choice.
