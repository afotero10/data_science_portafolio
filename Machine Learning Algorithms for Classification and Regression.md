# Machine Learning Algorithms for Classification and Regression

Machine learning offers a diverse set of algorithms for solving classification and regression problems. This guide provides an overview of some of the most widely used algorithms.

## Classification Algorithms

### 1. **Logistic Regression**
   - **Overview:** Logistic Regression is a linear model for binary classification that predicts the probability of an instance belonging to a particular class.
   - **Use Cases:** Predicting the likelihood of a customer clicking on an online ad, fraud detection in financial transactions.
   - **Hyperparameters:**
     - Regularization strength (`C`): Controls the inverse of regularization strength; smaller values specify stronger regularization.
     - Solver (`solver`): Algorithm to use in optimization (e.g., 'liblinear', 'newton-cg', 'lbfgs').
   - **Evaluation:**
     - Cross-validation during training

### 2. **Decision Trees**
   - **Overview:** Decision Trees recursively split the dataset based on features to make decisions. They are interpretable and suitable for both classification and regression tasks.
   - **Use Cases:** Customer churn prediction in telecom, loan approval systems.
   - **Hyperparameters:**
     - Maximum Depth (`max_depth`): The maximum depth of the tree.
     - Minimum Samples Split (`min_samples_split`): The minimum number of samples required to split an internal node.
     - Criterion (`criterion`): Function to measure the quality of a split (e.g., 'gini', 'entropy').
   - **Evaluation:**
     - Gini Index and Information Gain during training
     - Cross-validation during training

### 3. **Random Forest**
   - **Overview:** Random Forest is an ensemble model that builds multiple decision trees and combines their predictions to improve accuracy and reduce overfitting.
   - **Use Cases:** Predicting diseases based on medical records, stock price forecasting.
   - **Hyperparameters:**
     - Number of Estimators (`n_estimators`): The number of trees in the forest.
     - Maximum Features (`max_features`): The number of features to consider when looking for the best split.
     - Minimum Samples Leaf (`min_samples_leaf`): The minimum number of samples required to be at a leaf node.
   - **Evaluation:**
     - Out-of-Bag Error during training
     - Feature Importance during training
     - Cross-validation during training

### 4. **Support Vector Machines (SVM)**
   - **Overview:** SVM finds a hyperplane that best separates different classes in feature space. It is effective for both linear and non-linear classification.
   - **Use Cases:** Handwriting recognition, image classification.
   - **Hyperparameters:**
     - Kernel (`kernel`): Specifies the kernel type to be used (e.g., 'linear', 'rbf', 'poly').
     - Regularization Parameter (`C`): Trade-off between classifying training points correctly and having a smooth decision boundary.
   - **Evaluation:**
     - Margin Analysis during training
     - F1 Score during training

### 5. **K-Nearest Neighbors (KNN)**
   - **Overview:** KNN makes predictions based on the majority class of its k-nearest neighbors. It is simple and effective.
   - **Use Cases:** Recommender systems, anomaly detection.
   - **Hyperparameters:**
     - Number of Neighbors (`n_neighbors`): The number of neighbors to use for predictions.
     - Distance Metric (`metric`): The distance metric to use when calculating distances between instances.
   - **Evaluation:**
     - Choosing an optimal value for K during training

### 6. **Naive Bayes**
   - **Overview:** Naive Bayes is a probabilistic algorithm based on Bayes' theorem. It is simple and fast, making it suitable for large datasets.
   - **Use Cases:** Spam email detection, sentiment analysis.
   - **Hyperparameters:**
     - None, as it is based on probability and conditional independence assumptions.
   - **Evaluation:**
     - Cross-validation during training

## Regression Algorithms

### 1. **Linear Regression**
   - **Overview:** Linear Regression models the relationship between the dependent variable and one or more independent variables using a linear approach.
   - **Use Cases:** Predicting house prices, GDP growth forecasting.
   - **Hyperparameters:**
     - None, as it is based on solving the normal equation or using optimization techniques.
   - **Evaluation:**
     - Mean Squared Error (MSE)
     - R-squared
     - Adjusted R-squared
     - Mean Absolute Error (MAE)

### 2. **Ridge Regression**
   - **Overview:** Ridge Regression is a regularization technique that adds a penalty term to the linear regression cost function, preventing overfitting.
   - **Use Cases:** Dealing with multicollinearity in regression, preventing overfitting.
   - **Hyperparameters:**
     - Regularization strength (`alpha`): Controls the amount of regularization.
   - **Evaluation:**
     - Mean Squared Error (MSE)
     - R-squared
     - Adjusted R-squared
     - Mean Absolute Error (MAE)

### 3. **Lasso Regression**
   - **Overview:** Lasso Regression, similar to Ridge, adds a penalty term to the cost function, but it uses the absolute values of the coefficients, promoting sparsity.
   - **Use Cases:** Feature selection in high-dimensional datasets.
   - **Hyperparameters:**
     - Regularization strength (`alpha`): Controls the amount of regularization.
   - **Evaluation:**
     - Mean Squared Error (MSE)
     - R-squared
     - Adjusted R-squared
     - Mean Absolute Error (MAE)

### 4. **Decision Trees (for Regression)**
   - **Overview:** Decision Trees can also be used for regression tasks by predicting the target variable based on the average value of instances in leaf nodes.
   - **Use Cases:** Predicting house prices, demand forecasting.
   - **Hyperparameters:**
     - Maximum Depth (`max_depth`): The maximum depth of the tree.
     - Minimum Samples Split (`min_samples_split`): The minimum number of samples required to split an internal node.
     - Criterion (`criterion`): Function to measure the quality of a split (e.g., 'mse', 'mae').
   - **Evaluation:**
     - Mean Squared Error (MSE)
     - Mean Absolute Error (MAE)

### 5. **Random Forest (for Regression)**
   - **Overview:** Random Forest can be adapted for regression tasks by averaging the predictions of multiple decision trees.
   - **Use Cases:** Predicting sales, energy consumption forecasting.
   - **Hyperparameters:**
     - Number of Estimators (`n_estimators`): The number of trees in the forest.
     - Maximum Features (`max_features`): The number of features to consider when looking for the best split.
     - Minimum Samples Leaf (`min_samples_leaf`): The minimum number of samples required to be at a leaf node.
   - **Evaluation:**
     - Out-of-Bag Error during training
     - Feature Importance during training
     - Cross-validation during training

### 6. **Gradient Boosting (for Regression)**
   - **Overview:** Gradient Boosting builds a series of weak learners and combines their predictions, emphasizing the mistakes of the previous models.
   - **Use Cases:** Stock price prediction, medical diagnosis.
   - **Hyperparameters:**
     - Learning Rate (`learning_rate`): Controls the contribution of each weak learner.
     - Number of Estimators (`n_estimators`): The number of weak learners to train.
     - Maximum Depth (`max_depth`): The maximum depth of the weak learners (usually small).
   - **Evaluation:**
     - Mean Squared Error (MSE)
     - R-squared
     - Mean Absolute Error (MAE)
     - Learning Rate Tuning



## Neural Networks (NN)

### Overview
Neural Networks, inspired by the human brain, consist of interconnected nodes organized in layers. They are versatile and excel in complex tasks.

### Use Cases
Image and speech recognition, natural language processing, autonomous vehicles.

### Hyperparameters
Number of Layers, Number of Neurons per Layer, Activation Functions, Learning Rate.

### Evaluation
Cross-validation, Area Under the ROC Curve (AUC-ROC) for classification, Mean Squared Error (MSE) for regression.

# Model Evaluation Descriptions

## ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
Measures the area under the ROC curve, providing a trade-off between sensitivity and specificity.

## Cross-validation
Divides the dataset into multiple subsets, training the model on different combinations, and validating on the remaining data to assess generalization performance.

## Out-of-Bag Error
Random Forest-specific method that uses the data not included in the bootstrap sample to estimate model performance.

## Feature Importance
Assesses the contribution of each feature to the model's predictions.

## Margin Analysis (SVM)
Examines the distance of data points from the decision boundary.

## F1 Score
Harmonic mean of precision and recall, providing a balanced measure of classification performance.

   - **Precision (P):**
     - *Description:* The ratio of correctly predicted positive observations to the total predicted positives. It measures the accuracy of positive predictions.
     - *Formula:* P = True Positives / (True Positives + False Positives)

   - **Recall (R):**
     - *Description:* The ratio of correctly predicted positive observations to the total actual positives. It measures the ability of the model to capture all relevant instances.
     - *Formula:* R = True Positives / (True Positives + False Negatives)

## Mean Squared Error (MSE)
Average of squared differences between predicted and actual values, commonly used for regression tasks.

   - **Formula:**
     - MSE = Σ(Yi - Ŷi)² / n
       - n: Number of data points.
       - Yi: Actual value of the target variable for the i-th data point.
       - Ŷi: Predicted value of the target variable for the i-th data point.

## R-squared
Proportion of the variance in the dependent variable that is predictable from the independent variables.

   - **Formula:**
     - R² = 1 - Σ(Yi - Ŷi)² / Σ(Yi - Ȳ)²
       - Yi: Actual value of the target variable for the i-th data point.
       - Ŷi: Predicted value of the target variable for the i-th data point.
       - Ȳ: Mean of the actual values.

## Adjusted R-squared
R-squared adjusted for the number of predictors, addressing overfitting in regression models.

   - **Formula:**
     - Adjusted R² = 1 - [(1 - R²)(n - 1) / (n - k - 1)]
       - k: Number of predictors.

## Mean Absolute Error (MAE)
Average of absolute differences between predicted and actual values.

   - **Formula:**
     - MAE = Σ|Yi - Ŷi| / n
