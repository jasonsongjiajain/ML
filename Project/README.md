# Malaysian Housing Market Analysis Project

## Project Overview
This project focuses on analyzing housing affordability and market dynamics in Malaysia using machine learning approaches for price prediction and market segmentation. The analysis aims to provide valuable insights for various stakeholders including property investors, urban planners, and policymakers.

### Key Problem Areas
- Housing affordability challenges in Malaysia
- Need for data-driven insights for property investors
- Market trend analysis for urban planning
- Evidence-based housing policy development

## Dataset
Source: [Malaysia Housing Market Dataset 2025](https://www.kaggle.com/datasets/lyhatt/house-prices-in-malaysia-2025)
The analysis uses a comprehensive dataset (`malaysia_house_price_data_2025.csv`) containing the following features:
- Township
- Area
- State
- Tenure (Freehold/Leasehold)
- Property Type
- Median Price
- Price per Square Foot (PSF)
- Number of Transactions

## Technical Implementation

### Data Processing and Feature Engineering
1. **Market Value Indicators**
   - Total market value calculation
   - Price segmentation (Very Low to Very High)
   - Transaction volume categorization

2. **Property Categorization**
   - Luxury (Bungalows)
   - High-End (Semi-D, Duplex)
   - Mid-Range (Terrace, Cluster)
   - Affordable (Apartments, Flats)

3. **Location-based Features**
   - State-level price indices
   - Area development indices
   - Price deviation metrics

4. **Market Position Features**
   - Area-specific price percentiles
   - Relative market positioning

### Dimensional Reduction
- Principal Component Analysis (PCA) implementation
- 4 principal components explaining 91.9% of variance:
  - PC1: 35.5% variance explained
  - PC2: 25.2% variance explained
  - PC3: 16.5% variance explained
  - PC4: 14.8% variance explained

### Machine Learning Models
The project includes:
- Classification models for price segment prediction
- Regression models for price prediction
- Market segmentation through clustering

## Key Components
1. Data Preprocessing
   - Missing value handling
   - Feature scaling
   - Categorical encoding

2. Exploratory Data Analysis
   - Price distribution analysis
   - Geographic price variations
   - Transaction volume analysis
   - Property type distribution

3. Feature Engineering
   - Market value indicators
   - Location-based features
   - Price segment analysis
   - Property categorization

4. Model Development
   - PCA for dimension reduction
   - Train-test split (80-20)
   - Multiple model implementations

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Getting Started
1. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Load and prepare the dataset:
   ```python
   import pandas as pd
   df = pd.read_csv('./Dataset/Project/malaysia_house_price_data_2025.csv')
   ```

3. Run data preprocessing:
   ```python
   # Feature engineering
   df_processed = process_data(df)
   
   # PCA transformation
   pca_result = apply_pca(df_processed)
   ```

## Analysis Insights
- Price distribution patterns across different states
- Impact of property type on market value
- Transaction volume trends
- Market segmentation patterns
- Price prediction factors

## Future Improvements
1. Integration of additional data sources
2. Time series analysis for trend prediction
3. Advanced feature engineering
4. Model optimization and tuning
5. Interactive visualization development



# Logistic Regression Analysis

## Introduction

This project involves performing logistic regression to predict categories of median prices using PCA-transformed features. The logistic regression model is trained to classify median prices into five segments: 'Very Low', 'Low', 'Medium', 'High', and 'Very High'.

## Data Preparation

### Principal Component Analysis (PCA)
We performed Principal Component Analysis (PCA) to reduce the dimensionality of the feature space. This step helps in simplifying the model and improving its performance.

### Target Variable Encoding
The target variable, representing the median prices, was categorized into five segments: 'Very Low', 'Low', 'Medium', 'High', and 'Very High'. These categories were then encoded into numerical values for the logistic regression model.

## Model Training

```python
# Define the features and the encoded target variable
X = pca_df  
y = y_encode

# Spliting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)
```

## Model Evaluation
### Accuracy
The accuracy of the model is calculated to provide an overall measure of its performance.

```python
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```

## Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's performance.

```python
# Generate and plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], yticklabels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Classification Report
The classification report provides precision, recall, and F1-score for each class.

```python
# Generate classification report
class_report = classification_report(y_test, y_pred, target_names=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
print('Classification Report:')
print(class_report)
```

## Precision-Recall Curve
The Precision-Recall curve helps in understanding the trade-off between precision and recall.

```python
# Function to plot Precision-Recall curve
def plot_precision_recall_curve_multiclass(X_test, y_test, model, class_names):
    # Binarize the output
    y_test_bin = label_binarize(y_test, classes=[i for i in range(len(class_names))])

    # Compute Precision-Recall curve and area
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(len(class_names)):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    # Plot Precision-Recall curve
    plt.figure(figsize=(12, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
    for i, color in zip(range(len(class_names)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Precision-Recall curve of {class_names[i]} (area = {pr_auc[i]:0.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

class_names = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
plot_precision_recall_curve_multiclass(X_test, y_test, model, class_names)
```

## Conclusion
The logistic regression model provides valuable insights into predicting median price categories. These predictions can be used to make informed decisions regarding marketing strategies, pricing, inventory management, product development, and resource allocation.

## Recommendations
1. Targeted Marketing: Focus marketing efforts on high-value segments predicted by the model.
2. Pricing Strategies: Adjust pricing to attract customers in different segments.
3. Inventory Management: Allocate inventory based on predicted demand.
4. Product Development: Develop products catering to different price segments.
5. Resource Allocation: Invest in high-potential markets and improve services in promising regions.


# Random Forest Analysis for Malaysian Housing Market

## Introduction
In this project, Random Forest classification was used to predict the housing price segments in Malaysia for investors, urban planners and policymakers.

## Data Preparation

### Feature Engineering
```python
# Market value indicators
df_processed['total_market_value'] = df_processed['Median_Price'] * df_processed['Transactions']
df_processed['price_segment'] = pd.qcut(df_processed['Median_Price'], q=5, 
                                      labels=['Very Low','Low', 'Medium', 'High', 'Very High'])

# Property categorization
df_processed['property_category'] = df_processed['Type'].apply(categorize_property)

# Location metrics
state_metrics = df_processed.groupby('State').agg({
    'Median_Price': 'mean',
    'Transactions': 'sum',
    'Median_PSF': 'mean'
}).reset_index()
```

### Principal Component Analysis
```python
pca = PCA(n_components=4)
pca_result = pca.fit_transform(df_processed[numeric_features_for_pca])
explained_variance_ratio = pca.explained_variance_ratio_
```

### Target Variable Encoding
```python
labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
y_multiclass = pd.qcut(df['Median_Price'], q=5, labels=labels)
y_encode = LabelEncoder().fit_transform(y_multiclass)
```

## Model Training
```python
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
```

## Model Evaluation

### Classification Performance
```python
print(classification_report(y_test, y_pred, target_names=labels))
```

### Confusion Matrix
```python
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
```

### ROC Analysis
```python
y_prob = rf_model.predict_proba(X_test)
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_test == i, y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
```

## Key Findings
- Overall accuracy: 65%
- High/Very High segments: AUC = 0.97
- Premium segment precision: 0.88
- PC1 contribution: 41%

## Business Applications
1. Pricing Strategy
   - 0.88 of precision on premium segment enables confident pricing at the high end.
   - There is 0.49 precision, so pricing in medium segment is careful.

2. Risk Management
   - Leverage strong High/Very High segment performance
   - Monitor the indicators of PC1 for changes in the market

3. Market Segmentation
   - Resources should be focused on the high performing segments.
   - Segment-specific accuracy should be used to adjust strategies

4. Resource Allocation
   - Segment with highest prediction confidence should be prioritized.
   - Segments prediction can be used to optimize inventory.

# Project_SupportVectorClassifier
## Filename: Project_SupportVectorCLassifier.ipynb

## Notebook Structure
1. Data Preprocessing with PCA
2. Support Vector Classifier (SVC)
3. Model Performance Evaluation & Justification of Predictions
4. Why the Predictions are Useful and Important:
5. Applications

## Description
1. Data Preprocessing with PCA 
    - data exploration, prepocessing with PCA
2. Support Vector Classifier (SVC) 
    - initialize SVC, train SVC, hyperparameter tuning, results visualisations
3. Model Performance Evaluation & Justification of Predictions 
    - overview of model's performance, justification of predictions, future improvements
4. Why the Predictions are Useful and Important: 
    - model's predictions usage and benefits
5. Applications 
    - real life applications of the model

## Dependencies
- Python 3
- NumPy, Pandas
- Scikit-learn
- Matplotlib

## Usage
1. Load and preprocess the housing dataset.
2. Apply PCA to extract meaningful features.
3. Train an SVC model and analyze its performance.
4. Evaluate model predictions and craete visualisations.

## Key Findings
1. Best Model Accuracy: 64.25% (Improved from 61.25%)
2. Higher Precision & Recall for Classes 3 and 4
3. Moderate Misclassification for Classes 0, 1, and 2

## Business Applications
1. Market Adaptation: 
   - Real estate companies can use the model to adjust property pricing       dynamically based on predictive insights from the model

2. Price Adjustment Strategy: I
   - Implement the model to suggest price adjustments by classifying properties into their corresponding categories, which can be used to optimize profitability and attract buyers based on price trends.

3. Targeted Campaigns: 
   - Target specific customer groups with marketing strategies based on predicted property classifications. This could enhance advertising efficiency, consumer engagement, and conversion rates for property sales.

4. Trend Analysis: 
   - By continuously feeding the model with new data, the predictions can help in monitoring market dynamics. Over time, the model can adjust to shifting demand and supply trends and provide up-to-date insights on the housing market.

5. Price Sensitivity Analysis: 
   - Incorporating economic indicators (e.g., interest rates, inflation) into the model could provide predictive insights on how the housing market will behave in the coming months or years, helping businesses prepare for shifts in the market.