# Part 2 – Supervised Machine Learning Model

## Student

**Name:** Debojit Lahiri

# Objective

The objective of Part 2 is to build supervised machine learning models using the cleaned Ames Housing dataset prepared in Part 1.

Two predictive tasks were performed:

- Regression to predict the continuous house sale price.
- Binary classification to predict whether a house price is above the dataset median.

Both models were trained using proper preprocessing techniques to avoid data leakage and evaluated using appropriate performance metrics.

# Dataset

**Dataset:** Ames Housing Dataset (Kaggle House Prices Dataset)

**Original File:** `dataset/train.csv`

**Cleaned Dataset:** `dataset/cleaned_data.csv`

The cleaned dataset generated in Part 1 was used for all machine learning experiments.

# Target Variables

Two target variables were defined.

### Regression Target

- **SalePrice**

The regression model predicts the continuous sale price of a house.

### Classification Target

A binary classification target was created by comparing SalePrice with its median value.

```
y_clf = (SalePrice > SalePrice.median()).astype(int)
```

- Class 1 → House price above the median
- Class 0 → House price below or equal to the median

# Feature Encoding

The feature matrix contained both numerical and categorical variables.

## Ordinal Encoding

Ordinal variables having a natural ordering (such as quality ratings) were encoded while preserving their order.

Examples include:

- ExterQual
- ExterCond
- KitchenQual
- HeatingQC

The ordering used was:

Poor < Fair < Typical < Good < Excellent

which corresponds to:

Po < Fa < TA < Gd < Ex

Preserving this order allows the model to understand the relative ranking between categories.

## One-Hot Encoding

Nominal variables such as neighborhood names, roof styles, garage types and zoning categories do not possess any natural ordering.

These variables were converted into dummy variables using One-Hot Encoding with

```
drop_first=True
```

Dropping the first dummy variable reduces multicollinearity while avoiding the false ordinal relationships that Label Encoding would incorrectly introduce.

# Train-Test Split

The dataset was divided into training and testing sets using

- Training set: 80%
- Testing set: 20%

using

```
random_state = 42
```

to ensure reproducibility.

# Feature Scaling

StandardScaler was applied only on the training data.

```
scaler.fit(X_train)
```

The fitted scaler was then used to transform both training and testing data.

This avoids **data leakage** because fitting the scaler on the complete dataset would expose information from the test set during training, producing overly optimistic evaluation results.

# Linear Regression

A Linear Regression model was trained using the scaled training features.

## Results

| Metric | Value |
|---------|-------|
| Mean Squared Error (MSE) | 2,480,296,419 |
| R² Score | 0.6766 |

The model explains approximately **67.7%** of the variance in SalePrice.

# Largest Regression Coefficients

The three features having the largest absolute coefficient values were

| Feature | Coefficient |

| RoofMatl_CompShg | +76395.77 |
| GarageQual_TA | -54662.37 |
| RoofMatl_Tar&Grv | +52786.44 |

A large positive coefficient indicates that increasing the corresponding standardized feature increases the predicted SalePrice.

A large negative coefficient indicates that increasing the standardized feature decreases the predicted SalePrice.

# Ridge Regression

Ridge Regression was trained using

```
alpha = 1.0
```

to reduce overfitting through L2 regularization.

## Comparison

| Model | MSE | R² |

| Linear Regression | 2480296419 | 0.6766 |
| Ridge Regression | 2107902496 | 0.7252 |

Ridge Regression achieved a lower MSE and higher R² score, indicating better generalization performance.

The parameter **alpha** controls the strength of regularization.

Higher alpha values shrink regression coefficients more aggressively, reducing model complexity and helping prevent overfitting.

# Class Distribution

The binary classification dataset contained

| Class | Percentage |
|------|-----------:|
| Above Median | 51.11% |
| Below Median | 48.89% |

Since both classes exceeded 35% of the training samples, the dataset was considered balanced.

Therefore, no class imbalance handling techniques such as SMOTE or class weighting were required.

# Logistic Regression

A Logistic Regression model was trained using

```
max_iter = 1000
```

## Confusion Matrix

```
[[148 13]
 [  5 126]]
```

## Classification Results

| Metric | Value |

| Accuracy | 0.9384 |
| Precision | 0.9065 |
| Recall | 0.9618 |
| F1 Score | 0.9333 |
| AUC | 0.9663 |

The classifier achieved excellent performance across all evaluation metrics.

# Precision and Recall

Precision measures the proportion of predicted positive samples that are actually positive.

\[
Precision = \frac{TP}{TP + FP}
\]

Recall measures the proportion of actual positive samples that are correctly identified.

\[
Recall = \frac{TP}{TP + FN}
\]

For this housing classification task, **Precision and Recall are both important** because incorrectly classifying expensive and inexpensive houses can affect downstream decision making.

The model achieved a high Recall while maintaining strong Precision, resulting in an excellent F1-score.

# ROC Curve and AUC

The ROC curve illustrates the trade-off between the True Positive Rate and False Positive Rate across different classification thresholds.

The obtained

```
AUC = 0.9663
```

indicates outstanding discrimination ability.

An AUC close to 1 demonstrates that the model is highly effective at separating high-priced houses from low-priced houses.

# Decision Threshold Sensitivity

Thresholds from 0.30 to 0.70 were evaluated.

| Threshold | Precision | Recall | F1 |

|0.30|0.8707|0.9771|0.9209|
|0.40|0.8873|0.9618|0.9231|
|0.50|0.9065|0.9618|0.9333|
|0.60|0.9065|0.9618|0.9333|
|0.70|0.9197|0.9618|0.9403|

The highest F1-score was obtained at a threshold of **0.70**.

Increasing the threshold improves Precision while potentially reducing Recall.

For this dataset, a threshold of 0.70 provides the best balance between Precision and Recall.

# Logistic Regression Regularization

A second Logistic Regression model was trained using

```
C = 0.01
```

which corresponds to stronger L2 regularization.

| Model | Precision | Recall | AUC |

| C = 1.0 | 0.9197 | 0.9618 | 0.9663 |
| C = 0.01 | 0.9407 | 0.9695 | 0.9805 |

The parameter **C** controls the inverse strength of regularization.

Smaller values of C produce stronger regularization.

In this dataset, stronger regularization improved overall classification performance.

# Bootstrap Confidence Interval

A bootstrap experiment using **500 resamples** was performed to estimate the confidence interval of the AUC difference between the two Logistic Regression models.

## Results

Mean Difference

```
-0.0148
```

95% Confidence Interval

```
[-0.0301 , -0.0024]
```

The confidence interval does not include zero.

Therefore, the improved AUC achieved by the model with **C = 0.01** is likely to be statistically consistent rather than occurring by random chance.

# Conclusion

Part 2 successfully developed and evaluated supervised machine learning models for both regression and classification tasks.

Linear Regression and Ridge Regression were compared for predicting house prices, with Ridge Regression demonstrating superior performance.

Logistic Regression achieved excellent classification performance with an AUC of 0.9663, and stronger regularization further improved model performance.

These trained models provide the foundation for the advanced ensemble learning methods that will be developed in Part 3.
