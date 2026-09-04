# Credit Risk Scoring: Machine Learning & Explainable Credit Assessment

## Overview

This project develops a machine learning-based credit risk scoring
pipeline for identifying accounts that are more likely to be classified
as **bad credit accounts**.

The project compares three classification approaches:

-   Logistic Regression
-   Random Forest
-   XGBoost

In addition to predictive modelling, the notebook includes:

-   Data cleaning and missing-value analysis
-   Feature engineering
-   Class-imbalance handling
-   Model evaluation using ROC-AUC and PR-AUC
-   Precision, recall and F1-score analysis
-   Risk scoring
-   SHAP-based model explainability
-   KMeans-based risk segmentation
-   Confusion matrices and ROC/Precision-Recall curves
-   Discussion of credit-risk governance and model limitations

The project is designed as a portfolio example of how machine learning
can be applied to a **highly imbalanced credit-risk problem**, while
also considering interpretability and practical decision-making.

------------------------------------------------------------------------

## Business Problem

Credit-risk modelling is a classification problem where the objective is
to distinguish between accounts that are likely to perform well and
accounts that may present a higher risk of default or adverse credit
behaviour.

In this dataset, the target variable is:

-   `bad_flag = 0` → Good account
-   `bad_flag = 1` → Bad account

The main challenge is that bad accounts represent only a small
proportion of the dataset. A model that simply predicts every account as
good could achieve very high accuracy while being practically useless
for identifying risky customers.

The project therefore focuses on metrics such as **recall, precision,
F1-score, ROC-AUC and PR-AUC**, rather than relying on accuracy alone.

------------------------------------------------------------------------

## Dataset

The dataset contains approximately **96,806 accounts** and initially
includes **1,216 columns**.

The available variables are grouped into several feature families:

-   `transaction_attribute`
-   `bureau`
-   `bureau_enquiry`
-   `onus_attribute`

The target variable is `bad_flag`.

### Class Distribution

  Class  Meaning     Accounts   Proportion
 - 0           Good               95,434            98.58%
 - 1           Bad                 1,372             1.42%

This represents a significant class imbalance, making appropriate
evaluation and class-weighting strategies particularly important.

------------------------------------------------------------------------

## Data Preparation

The notebook follows a structured preprocessing workflow.

### 1. Missing-value analysis

Features with more than 50% missing values were removed. This reduced
the feature space by removing variables with limited usable information.

After this step, the working dataset contained approximately **1,197
columns**.

The remaining feature set still contained substantial missingness, with
the median feature missingness being approximately 26%.

### 2. Train/Test Split

The dataset was divided into:

-   **80% training data:** 77,444 accounts
-   **20% testing data:** 19,362 accounts

A stratified split was used so that the proportion of good and bad
accounts remained approximately consistent between the training and
testing sets.

### 3. Missing-value Imputation

Missing numerical values were replaced using **training-set medians**.

Importantly, the imputation statistics were calculated from the training
data rather than the entire dataset. This prevents information from the
test set from influencing preprocessing.

### 4. Feature Engineering

Additional aggregate features were created from the main feature
families.

For each feature family, the notebook calculated:

-   Mean
-   Standard deviation
-   Sum
-   Missing-value count

This produced **16 additional engineered features**.

The resulting modelling dataset contained approximately **1,211
features**.

------------------------------------------------------------------------

## Modelling Approach

Three machine learning models were evaluated.

### Logistic Regression

Logistic Regression provides a relatively simple and interpretable
baseline for binary classification.

Because the target variable is highly imbalanced, class weighting was
used to give greater importance to the minority class.

Feature scaling was also applied for the Logistic Regression model.

### Random Forest

Random Forest was used to capture nonlinear relationships and
interactions between credit-related variables.

The model used balanced class weighting to improve its ability to
identify bad accounts.

Unlike Logistic Regression, feature scaling is not required for Random
Forest.

### XGBoost

XGBoost was included as a more powerful gradient-boosting approach.

The model used `scale_pos_weight` to compensate for the imbalance
between good and bad accounts.

XGBoost achieved the strongest overall predictive performance among the
three models tested.

------------------------------------------------------------------------

## Model Performance

The models were evaluated on the held-out test set.

     
  ROC-AUC    
  - Logistic Regression          **0.771**                                                                
 - Random Forest              **0.818**                                                                        
 - XGBoost         **0.836**  
 
  PR-AUC 
  - Logistic Regression                 **0.049**                                                             
 - Random Forest                   **0.067**                                                                        
 - XGBoost          **0.083**  
  
  Precision   
  - Logistic Regression         **0.045**                                                               
 - Random Forest             **0.052**                                                                        
 - XGBoost           **0.058**     
  
  Recall   
  - Logistic Regression       **0.657**                                                                
 - Random Forest          **0.602**                                                                        
 - XGBoost            **0.613**   
  
  F1  
  - Logistic Regression       **0.085**                                                               
 - Random Forest        **0.095**                                                                      
 - XGBoost      **0.106**    
  
  Accuracy

 - Logistic Regression          **0.799**                                                            
 - Random Forest              **0.838**                                                                    
 - XGBoost          **0.854**


### Key Finding

**XGBoost performed best overall**, achieving:

-   ROC-AUC: **0.836**
-   PR-AUC: **0.083**
-   Precision: **0.058**
-   Recall: **0.613**
-   F1-score: **0.106**
-   Accuracy: **0.854**

The relatively low precision is expected in part because the dataset
contains only about 1.42% bad accounts. The models are intentionally
configured to identify a larger proportion of risky accounts, which
increases recall but also produces more false positives.

For this reason, accuracy should not be interpreted as the primary
measure of credit-risk model quality.

------------------------------------------------------------------------

## Why PR-AUC Matters

With a highly imbalanced target, the Precision-Recall curve provides
important information that ROC-AUC alone may not capture.

The bad-account prevalence is approximately **1.42%**, so a random
classifier would have a PR-AUC close to that prevalence.

The XGBoost PR-AUC of **0.083** is substantially above this baseline,
indicating that the model provides useful ranking information for
identifying the minority class.

However, PR-AUC should not be interpreted as a direct measure of
business "lift" without defining an appropriate baseline and lift
methodology.

------------------------------------------------------------------------

## Risk Scoring

The XGBoost model's output was used to create an illustrative risk
score:

**Risk Score = predicted model score × 100**

Higher scores indicate that the model considers an account more likely
to belong to the bad-account class.

The notebook also demonstrates how a decision threshold could be
explored.

However, the threshold included in the notebook is **illustrative
only**. It should not be treated as a production credit-approval or
decline rule.

A production threshold would need to consider factors such as:

-   Cost of false positives
-   Cost of false negatives
-   Expected credit losses
-   Approval rates
-   Business policy
-   Regulatory requirements
-   Model calibration
-   Fairness and bias
-   Portfolio risk appetite

### Important Calibration Note

Because the models use class weighting and XGBoost uses
`scale_pos_weight`, the raw model scores should **not automatically be
interpreted as calibrated probabilities of default**.

A production implementation would require probability calibration and
validation before using scores as estimated probabilities.

------------------------------------------------------------------------

## Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to understand how
individual features contribute to model predictions.

The notebook includes both:

### Global Explainability

Global SHAP analysis helps identify which features have the greatest
influence across the overall test population.

### Local Explainability

Local SHAP analysis examines an individual account and identifies the
features that contributed most strongly toward its prediction.

This provides a useful way to move beyond simply saying:

> "The model classified this account as high risk."

Instead, the analysis can investigate:

> "Which model inputs contributed most to this account's risk score?"

The feature names in the source dataset are anonymised/technical, so the
notebook deliberately avoids inventing business meanings for individual
variables.

SHAP should also not be considered a complete regulatory adverse-action
explanation system. A production credit model would require additional
governance, documented feature definitions and validated reason-code
processes.

------------------------------------------------------------------------

## Risk Segmentation with KMeans

KMeans clustering was used to explore whether accounts could be
segmented into groups with different levels of observed credit risk.

The clustering analysis uses aggregate behavioural/credit features to
identify groups of accounts with similar profiles.

To reduce evaluation leakage, the revised workflow fits the clustering
preprocessing and KMeans model using the training population and then
assigns clusters to the test population.

Test-set bad rates are then compared across the resulting clusters.

This is intended as **exploratory risk segmentation**, rather than a
replacement for the supervised credit-risk model.

------------------------------------------------------------------------

## Visual Analysis

The notebook includes several visualisations to evaluate model
behaviour:

-   Class distribution
-   Missing-value patterns
-   Feature distributions
-   Confusion matrices
-   ROC curves
-   Precision-Recall curves
-   Precision, recall and F1-score comparisons
-   SHAP feature importance
-   Local SHAP explanations
-   Cluster-level risk comparisons

These visualisations make it easier to evaluate the trade-offs between
identifying risky accounts and generating false positives.

------------------------------------------------------------------------

## Key Challenges

### Severe Class Imbalance

Only approximately 1.42% of accounts are labelled as bad.

This makes accuracy misleading and requires techniques such as:

-   Stratified sampling
-   Class weighting
-   `scale_pos_weight`
-   Precision-Recall evaluation

### High Dimensionality

The dataset contains more than 1,200 variables, creating challenges
around:

-   Computation
-   Missing values
-   Model complexity
-   Overfitting
-   Interpretability

### Missing Data

A large proportion of the feature space contains missing values.

Removing extremely sparse features and applying training-based median
imputation provides a practical baseline, but a production system would
require deeper investigation into whether missingness itself contains
meaningful behavioural information.

### Interpretability

Credit-risk decisions can have significant consequences for customers.

Therefore, predictive performance alone is not sufficient. The model
also needs to be explainable, monitored and governed appropriately.

------------------------------------------------------------------------

## Limitations

This project is intended as a **portfolio-level machine learning
project**, not a production-ready lending system.

Important limitations include:

1.  The dataset and feature definitions are not fully described in
    business terms, limiting domain interpretation.
2.  No external validation dataset was used.
3.  Hyperparameter optimisation was not developed into a full
    production-grade model-selection framework.
4.  The model probabilities were not fully calibrated.
5.  Fairness and demographic bias were not comprehensively evaluated.
6.  The chosen decision threshold is illustrative rather than a
    validated credit policy.
7.  SHAP explanations are model explanations, not automatically valid
    regulatory adverse-action reasons.
8.  KMeans segmentation is exploratory and does not establish causal
    relationships.
9.  Production monitoring, drift detection and champion/challenger
    validation were outside the scope of the project.

------------------------------------------------------------------------

## Recommended Next Steps

A production-oriented version of this project could be extended with:

-   Cross-validation and systematic hyperparameter optimisation
-   Probability calibration
-   Cost-sensitive threshold optimisation
-   Population Stability Index (PSI) and feature drift monitoring
-   Out-of-time validation
-   Fairness and bias testing
-   Feature stability analysis
-   Model monitoring dashboards
-   Champion/challenger modelling
-   Automated model retraining pipelines
-   Formal model governance and documentation
-   Validated credit decision reason codes
-   Expected-loss and portfolio-level impact analysis

------------------------------------------------------------------------

## Technologies Used

-   **Python**
-   **Pandas**
-   **NumPy**
-   **Matplotlib**
-   **Seaborn**
-   **Scikit-learn**
-   **XGBoost**
-   **SHAP**
-   **Jupyter Notebook / Google Colab**

------------------------------------------------------------------------

## Project Structure

``` text
credit-risk-scoring/
│
├── Credit_Risk_Scoring_Portfolio_Ready.ipynb
├── README.md
└── data/
    └── Dev_data_to_be_shared.csv
```

> The dataset is not included in this repository if it is subject to
> distribution or privacy restrictions.

------------------------------------------------------------------------

## How to Run

1.  Clone or download the repository.
2.  Open `Credit_Risk_Scoring_Portfolio_Ready.ipynb` in Jupyter Notebook
    or Google Colab.
3.  Provide the dataset when prompted.
4.  Run the notebook from top to bottom.
5.  Review the preprocessing, model evaluation, explainability and
    segmentation results.

The notebook is structured so that the analysis progresses from data
preparation through modelling, evaluation and interpretation.

------------------------------------------------------------------------

## Portfolio Takeaway

This project demonstrates an end-to-end approach to a realistic and
challenging **imbalanced credit-risk classification problem**.

The strongest model was XGBoost, which achieved a **0.836 ROC-AUC and
0.083 PR-AUC** on the held-out test set. More importantly, the project
goes beyond model training by addressing class imbalance, risk scoring,
model explainability and exploratory risk segmentation.

The project highlights an important principle in applied data science:

**A strong model is not only about predictive performance. It also needs
to be evaluated in the context of the business problem, explained
appropriately, and developed with validation, governance and responsible
decision-making in mind.**

------------------------------------------------------------------------

## Author

**Irfan Moosa**

BSc in Information Technology --- Computer Science & Business Management

This project forms part of a data analytics / machine learning portfolio
focused on applying technical skills to practical business problems.
