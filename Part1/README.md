# Part 1: Data Cleaning and Exploratory Data Analysis (EDA)

## Applied AI & ML Capstone Project

Student: Debojit Lahiri

# Objective

The objective of Part 1 is to inspect, clean, and understand the Ames Housing dataset before applying machine learning techniques. This stage focuses on identifying missing values, detecting duplicates, optimizing data types, analyzing statistical properties, detecting outliers, and visualizing relationships between variables. The cleaned dataset generated in this part will be used in Parts 2 and 3 of the project.

# Dataset Description

Dataset Name: Ames Housing Dataset (House Prices: Advanced Regression Techniques)

Source: Kaggle

The dataset contains residential property information collected from Ames, Iowa. Each record represents one house and includes structural characteristics, neighborhood information, construction details, and the final sale price.

# Dataset Characteristics

- Total observations: **1460**
- Total features: **81**
- Target variable: **SalePrice**
- Numerical features: Multiple continuous and discrete variables
- Categorical features: Multiple nominal and ordinal variables

The target variable, **SalePrice**, is continuous and is therefore suitable for regression. It can also be converted into a binary target (HighPrice / LowPrice) using a threshold such as the median, making the dataset suitable for classification tasks as required by the assignment.

# Dataset Justification

The Ames Housing dataset was selected because it satisfies all assignment requirements.

- It contains more than 500 observations.
- It includes considerably more than five numerical features.
- It contains more than two categorical features.
- The target variable (SalePrice) supports both regression and classification tasks.
- It is a well-known benchmark dataset used extensively in machine learning research and education.
- The dataset contains realistic missing values, skewed distributions, categorical variables, and outliers, making it suitable for demonstrating complete data preprocessing and exploratory analysis.

# Initial Data Exploration

The dataset was loaded into a Pandas DataFrame using `pd.read_csv()`. An initial inspection was performed by displaying the first five rows, examining the data types of all columns, and checking the overall dimensions of the dataset.

The dataset contains **1460 observations** and **81 columns**, consisting of both numerical and categorical variables. The target variable, **SalePrice**, represents the selling price of each house and serves as the prediction target for future machine learning models.

Initial exploration confirmed that the dataset contains several missing values distributed across multiple features. Therefore, data cleaning was required before model development.

# Missing Value Analysis

Missing values were identified using:

- `df.isnull().sum()`
- `(df.isnull().sum() / df.shape[0]) * 100`

Both the number and percentage of missing values were computed for every column.

Several features contained a very high percentage of missing values. The most significant were:

| Feature | Approximate Missing Percentage |

| PoolQC | 99.5% |
| MiscFeature | 96.3% |
| Alley | 93.8% |
| Fence | 80.8% |
| MasVnrType | 59.7% |
| FireplaceQu | 47.3% |

Columns with more than **20%** missing values were identified separately because they require special consideration and were not imputed during this stage.

For numerical columns with **20% or fewer missing values**, missing values were replaced using the **median** of the respective column.

# Why Median was Chosen Instead of Mean

The median was selected for numerical imputation because several numerical features in the Ames Housing dataset exhibit skewed distributions and contain outliers.

Unlike the mean, the median is resistant to extreme values and therefore provides a better estimate of the central tendency for skewed data.

For example, in a positively skewed distribution, a few unusually large observations increase the mean considerably, whereas the median remains relatively stable. Using the median therefore reduces the influence of extreme observations and produces more reliable imputations.

This strategy preserves the overall distribution of the dataset and minimizes bias introduced during data preprocessing.

# Duplicate Detection and Removal

Duplicate records were identified using the `duplicated()` function.

The analysis showed:

- Duplicate rows before removal: **0**
- Duplicate rows removed: **0**
- Final number of observations: **1460**

Since no duplicate records were present, removing duplicates had no effect on the dataset or on the percentage of missing values.

# Data Type Correction

The inferred data types of the dataset were inspected before preprocessing.

The Ames Housing dataset already contained correctly inferred numerical data types. No incorrectly stored numerical columns required conversion.

However, all repetitive string (`object`) columns were converted to the `category` data type to improve memory efficiency.

Memory usage was measured before and after conversion, confirming that categorical encoding reduced the overall memory consumption without affecting the underlying information.

This optimization is beneficial because categorical variables often contain repeated values, making the `category` data type considerably more memory efficient than storing repeated strings.

# Descriptive Statistics

Descriptive statistics were computed for all numerical features using the `describe()` function. The analysis included the count, mean, standard deviation, minimum, maximum, and quartile values for each numerical variable.

These statistics provide an overview of the distribution and variability of each feature before machine learning modeling. Features such as `SalePrice`, `GrLivArea`, and `LotArea` exhibited substantial variation, indicating that the dataset contains houses with a wide range of characteristics.

# Skewness Analysis

The skewness of every numerical feature was calculated to identify variables with asymmetric distributions.

The feature with the highest absolute skewness was:

- **Feature:** `MiscVal`
- **Skewness:** **24.4768**

The distribution of `MiscVal` is **positively skewed (right-skewed)**. Most observations have values close to zero, while a small number of houses contain very large miscellaneous values.

In a positively skewed distribution, the mean is pulled upward by extreme observations, whereas the median remains comparatively stable. Therefore, the median provides a better estimate of the central tendency for this feature and is preferred when imputing missing values.

Understanding skewness is important because highly skewed variables may require transformations or robust preprocessing techniques before training machine learning models.

# Outlier Detection Using the Interquartile Range (IQR)

Outliers were identified using the Interquartile Range (IQR) method.

The following values were calculated for each selected feature:

- First Quartile (Q1)
- Third Quartile (Q3)
- Interquartile Range (IQR)
- Lower Bound
- Upper Bound

Rows outside these bounds were identified as potential outliers.

## SalePrice

The IQR analysis identified **61 outliers** in the `SalePrice` feature.

These observations correspond to houses with exceptionally high selling prices. Since these values are realistic observations rather than data-entry errors, they were retained in the dataset.

Removing these observations could reduce the model's ability to learn the complete range of housing prices.

## GrLivArea

The IQR analysis identified **31 outliers** in the `GrLivArea` feature.

These observations represent houses with unusually large living areas. Similar to the SalePrice outliers, these records appear to be valid and therefore were not removed.

Instead of deleting these observations, robust preprocessing methods or suitable machine learning algorithms will be considered in Part 2 to reduce the influence of extreme values while preserving important information.

# Summary of Data Cleaning

The following preprocessing steps were completed successfully:

- Loaded the raw dataset into a Pandas DataFrame.
- Inspected data types and dataset dimensions.
- Calculated missing-value counts and percentages.
- Identified columns exceeding 20% missing values.
- Imputed eligible numerical columns using the median.
- Verified duplicate records and confirmed that none were present.
- Optimized categorical columns using the `category` data type.
- Computed descriptive statistics.
- Performed skewness analysis.
- Detected outliers using the IQR method.
- Generated a cleaned dataset for subsequent machine learning tasks.

# Data Visualization

Several visualizations were created to better understand the distribution of variables and the relationships between features in the dataset.

## Line Plot

A line plot of **SalePrice** against the house index was generated to observe how sale prices vary across the dataset.

### Interpretation

The plot shows substantial variation in house prices across different observations. Although no time-based trend exists because the dataset is not chronological, the visualization clearly demonstrates the presence of several exceptionally expensive properties that appear as sharp peaks. These observations are consistent with the outliers identified during the IQR analysis.

## Bar Chart

A bar chart was created showing the **average SalePrice for each Neighborhood**.

### Interpretation

The visualization indicates that average house prices vary considerably across neighborhoods. Some neighborhoods consistently exhibit much higher average selling prices than others, suggesting that location has a significant influence on house value.

This observation indicates that **Neighborhood** is likely to be an informative predictor for machine learning models.

## Histogram

A histogram was generated for the most skewed numerical feature, **MiscVal**.

### Interpretation

The histogram confirms that the distribution is **highly positively skewed**. Most observations are concentrated near zero, while only a few houses contain very large miscellaneous values.

This shape explains why the median is preferred over the mean for representing the center of this distribution.

## Scatter Plot

A scatter plot was created using **GrLivArea** and **SalePrice**.

### Interpretation

The scatter plot reveals a **strong positive relationship** between ground living area and sale price.

In general, houses with larger living areas tend to have higher selling prices. Although several outliers are present, the overall trend is clearly increasing.

This suggests that **GrLivArea** is one of the strongest predictive variables for estimating house prices.


## Box Plot

A box plot was constructed to compare **SalePrice** across different values of **OverallQual**.

### Interpretation

The box plot shows a clear increase in the median SalePrice as the overall quality rating increases.

Higher-quality houses not only have higher median prices but also exhibit greater variability in selling prices.

This indicates that **OverallQual** is a highly informative feature for predicting house prices.

The highest Pearson correlation identified in the dataset was between **GarageCars** and **GarageArea**, with a correlation coefficient of **0.8825**.

This strong positive correlation is expected because houses with more garage spaces generally require larger garage areas. Although these variables are strongly correlated, this does not necessarily imply a direct causal relationship. Both variables describe different aspects of the same physical structure (the garage), which naturally results in a high correlation.

This relationship suggests that one of these features may provide information similar to the other during machine learning. Feature importance and multicollinearity will be examined further in Part 2 before deciding whether both variables should be retained.

# Mean vs. Median Imputation Comparison

The two numerical features with the highest absolute skewness were:

- **MiscVal**
- **PoolArea**

The mean and median values were compared before imputation.

| Feature | Mean | Median |

| MiscVal | 43.489 | 0 |
| PoolArea | 2.759 | 0 |

The large difference between the mean and median confirms that these variables are highly positively skewed. Extreme values pull the mean upward, whereas the median remains unaffected.

For highly skewed variables, the median provides a better representation of the central tendency. Therefore, the median was selected as the preferred statistic for numerical imputation wherever applicable.

After applying the selected imputation strategy, all remaining eligible numerical missing values were successfully handled.

# Spearman Rank Correlation

A Spearman rank correlation matrix was computed and compared with the Pearson correlation matrix.

Unlike Pearson correlation, Spearman correlation measures monotonic relationships and is therefore capable of identifying variables that move together consistently even when the relationship is not perfectly linear.

The three variable pairs with the largest absolute difference between Spearman and Pearson correlations were:

| Variable Pair | |Spearman − Pearson| |

| LotFrontage ↔ LotArea | 0.2496 |
| LotArea ↔ BedroomAbvGr | 0.2181 |
| LotArea ↔ TotRmsAbvGrd | 0.2159 |

These differences indicate that the relationships between these variables are more monotonic than strictly linear. Consequently, Spearman correlation provides additional insight that Pearson correlation alone cannot capture.

For feature selection in Part 2, Pearson correlation will primarily be used because several planned machine learning models assume approximately linear relationships. However, Spearman correlation will also be considered when evaluating variables that exhibit clear monotonic but non-linear behaviour.

# Grouped Aggregation

Grouped aggregation was performed by grouping houses according to **Neighborhood** and computing the **mean**, **standard deviation**, and **count** of **SalePrice**.

### Results

- **Neighborhood with the highest mean SalePrice:** **NoRidge**
- **Neighborhood with the highest standard deviation:** **NoRidge**
- **Ratio of highest mean to lowest mean SalePrice:** **3.40**

The results indicate that **NoRidge** contains the most expensive houses on average while also exhibiting the greatest variation in property prices.

A high within-group standard deviation suggests that the neighborhood alone cannot completely explain house prices. Even within the same neighborhood, factors such as living area, construction quality, number of rooms, garage size, and house age contribute significantly to variations in sale price.

The ratio of **3.40** between the highest and lowest neighborhood mean SalePrice indicates a substantial difference in housing prices across neighborhoods. This suggests that **Neighborhood** carries meaningful predictive information and is expected to be an important feature during machine learning model development.

# Conclusion

Part 1 successfully transformed the raw Ames Housing dataset into a clean and well-understood dataset suitable for machine learning.

The preprocessing workflow included missing value analysis, duplicate detection, data type optimization, descriptive statistics, skewness analysis, outlier detection, visualization, correlation analysis, grouped aggregation, and dataset cleaning.

A cleaned dataset (`cleaned_data.csv`) was generated and saved for use in Parts 2 and 3 of the project.

The insights obtained during exploratory data analysis provide a strong foundation for feature engineering, supervised machine learning, ensemble modeling, and the LLM-powered explanation system that will be developed in the subsequent parts of this capstone project.
