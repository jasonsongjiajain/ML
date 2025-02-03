# Binary and Multiclass Classification with Regularization

This repository contains Jupyter notebooks demonstrating the implementation and analysis of logistic regression with different regularization techniques for both binary and multiclass classification problems.In addition, a comparison between logistic regression and classification models is provided.

## Overview

The project is divided into four main sections:

1. Clarifying Concepts 
2. Binary Classification with Regularization (L1/L2)
3. Multiclass Logistic Regression
4. Comparison between Lgoistic Regression and Classification Models

### Part 1: Binary Classification

This section focuses on clarifying the concepts of logistic regression.

#### Features:
- Logistic regression basics
- The Sigmoid function


### Part 2: Binary Classification

This section focuses on email spam classification using different regularization techniques and sampling methods.

#### Features:
- Implementation of L1 (Lasso) and L2 (Ridge) regularization
- Handling imbalanced data using various sampling techniques:
  - SMOTE (Synthetic Minority Over-sampling Technique)
  - Random Undersampling
  - SMOTE + Tomek Links
- Comprehensive model evaluation and comparison
- Feature importance analysis
- Optimal regularization strength selection through cross-validation

### Part 3: Multiclass Classification

This section uses the Iris dataset to demonstrate multiclass classification strategies.

#### Features:
- One-vs-Rest (OvR) classification strategy
- One-vs-One (OvO) classification strategy
- Decision boundary visualization
- Performance comparison between OvR and OvO approaches

### Part 4: Comparison between Logistic Regression and Classification Models

This section uses the Iris dataset to demonstrate comparison between logistic regression and classification models.

#### Features:
- Comparison between logistic regression and classification models

## Requirements

```
python
numpy
pandas
scikit-learn
matplotlib
seaborn
imbalanced-learn

```

## Dataset Information

### Binary Classification
- Dataset: Email spam classification
- Features: Text data converted to TF-IDF features
- Target: Binary (spam/not spam)
- Size: 5,728 samples

### Multiclass Classification
- Dataset: Iris dataset
- Features: 4 numerical features
- Target: 3 classes (setosa, versicolor, virginica)
- Size: 150 samples

### Comparison between Logistic Regression and Classification Models
- Dataset: Iris dataset
- Features: 4 numerical features
- Target: 3 classes (setosa, versicolor, virginica)
- Size: 150 samples

## Key Components

1. **Data Preprocessing**
   - Text vectorization using TF-IDF
   - Feature scaling
   - Handling class imbalance

2. **Model Implementation**
   - Basic Logistic Regression (baseline)
   - L1 Regularized Logistic Regression
   - L2 Regularized Logistic Regression
   - One-vs-Rest and One-vs-One classifiers

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - ROC curves and AUC
   - Confusion matrices
   - Feature importance visualization

4. **Hyperparameter Optimization**
   - Grid search for optimal regularization strength
   - Cross-validation

## Results and Findings

### Binary Classification
- Best sampling method: SMOTE
- Number of non-zero coefficients:
  - Baseline: 1000
  - L1 Regularization: 133
  - L2 Regularization: 1000
- Optimal regularization strength:
  - L1: C=10 (Cross-validation score: 0.9552)
  - L2: C=10 (Cross-validation score: 0.9670)

### Multiclass Classification
- Both OvR and OvO strategies show comparable performance on the Iris dataset
- OvO provides better handling of class imbalance but requires training more classifiers
- Decision boundary visualizations show different partitioning approaches

### Comparison between Logistic Regression and Classification Models
- Logistic Regression and Decision Tree were fastest in training
- Random Forest took significantly longer (0.174 seconds)
- All models had minimal prediction times

## Usage

1. Load and prepare the data:
```python
# For binary classification
df = pd.read_csv('./Dataset/Lab_Enhancement_Lab8/emails.csv')
X = tfidf.fit_transform(df['text'])
y = df['spam'].astype(int)

# For multiclass classification
iris = load_iris()
X = iris.data
y = iris.target
```

2. Train models:
```python
# Binary classification with L1 regularization
l1_model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
l1_model.fit(X_train, y_train)

# Multiclass classification
ovr_clf = OneVsRestClassifier(LogisticRegression())
ovr_clf.fit(X_train_scaled, y_train)
```

3. Evaluate performance:
```python
# Get predictions
y_pred = model.predict(X_test)

# Print metrics
print(classification_report(y_test, y_pred))
```

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is open-source and available under the MIT License.