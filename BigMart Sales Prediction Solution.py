# BigMart Sales Prediction Solution
# --------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Data Loading
print("Loading the datasets...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display basic information
print("\nTraining data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

# 2. Exploratory Data Analysis
print("\n== Exploratory Data Analysis ==")

# Check for missing values
print("\nMissing values in training data:")
print(train_data.isnull().sum())

# Descriptive statistics
print("\nDescriptive statistics of numerical features:")
print(train_data.describe())

# Distribution of target variable
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Item_Outlet_Sales'], kde=True)
plt.title('Distribution of Item_Outlet_Sales')
plt.savefig('sales_distribution.png')
plt.close()

# Outlet establishment year distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Outlet_Establishment_Year', data=train_data)
plt.title('Outlet Establishment Year Distribution')
plt.xticks(rotation=90)
plt.savefig('establishment_year_dist.png')
plt.close()

# Item type distribution
plt.figure(figsize=(14, 7))
sns.countplot(y='Item_Type', data=train_data)
plt.title('Item Type Distribution')
plt.savefig('item_type_dist.png')
plt.close()

# Sales by outlet type
plt.figure(figsize=(10, 6))
sns.boxplot(x='Outlet_Type', y='Item_Outlet_Sales', data=train_data)
plt.title('Sales by Outlet Type')
plt.xticks(rotation=90)
plt.savefig('sales_by_outlet_type.png')
plt.close()

# 3. Data Preprocessing
print("\n== Data Preprocessing ==")

# Combine train and test for preprocessing
df = pd.concat([train_data, test_data], ignore_index=True)
df['source'] = ['train']*len(train_data) + ['test']*len(test_data)

# Handle missing values
print("\nHandling missing values...")

# Item_Weight - impute with median
weight_median = df[df['Item_Weight'].notnull()]['Item_Weight'].median()
df['Item_Weight'].fillna(weight_median, inplace=True)

# Outlet_Size - fill with mode per outlet type
df['Outlet_Size'].fillna('Missing', inplace=True)

# Feature Engineering
print("\nPerforming feature engineering...")

# 1. Clean Item_Fat_Content
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 
                                                         'reg': 'Regular'})

# 2. Create year operation feature
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

# 3. Item_Visibility - Replace 0 values with mean per product type
zero_visibility = df['Item_Visibility'] == 0
for item_type in df['Item_Type'].unique():
    item_mean_visibility = df[(df['Item_Type'] == item_type) & (df['Item_Visibility'] > 0)]['Item_Visibility'].mean()
    df.loc[(zero_visibility) & (df['Item_Type'] == item_type), 'Item_Visibility'] = item_mean_visibility

# 4. Item price category
df['Item_MRP_Binned'] = pd.qcut(df['Item_MRP'], 4, labels=['Low', 'Medium', 'High', 'Very High'])

# 5. Item type category - simplify categories
item_type_dict = {
    'Breads': 'Food',
    'Breakfast': 'Food',
    'Canned': 'Food',
    'Dairy': 'Food',
    'Frozen Foods': 'Food',
    'Fruits and Vegetables': 'Food',
    'Hard Drinks': 'Drinks',
    'Health and Hygiene': 'Non-Consumable',
    'Household': 'Non-Consumable',
    'Meat': 'Food',
    'Others': 'Non-Consumable',
    'Seafood': 'Food',
    'Snack Foods': 'Food',
    'Soft Drinks': 'Drinks',
    'Starchy Foods': 'Food'
}

df['Item_Category'] = df['Item_Type'].apply(lambda x: item_type_dict.get(x, 'Other'))

# Label encoding for categorical variables
print("\nEncoding categorical variables...")
categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                    'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 
                    'Item_MRP_Binned', 'Item_Category']

label_encoders = {}
for column in categorical_cols:
    le = LabelEncoder()
    df[column + '_Encoded'] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# One-hot encoding for selected variables
ohe_features = ['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Category']
for feature in ohe_features:
    encoded_features = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
    df = pd.concat([df, encoded_features], axis=1)

# Split data back to train and test
train_processed = df[df['source'] == 'train'].drop('source', axis=1)
test_processed = df[df['source'] == 'test'].drop('source', axis=1)

# 4. Feature Selection
print("\n== Feature Selection ==")

# Exclude identifiers and target from features
exclude_cols = ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales', 'source', 'Item_Type', 'Item_Fat_Content',
               'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_MRP_Binned', 'Item_Category']
numerical_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Years']
categorical_features = [col for col in train_processed.columns 
                        if col.endswith('_Encoded') or '_' in col and not col.startswith('Item_') and not col.startswith('Outlet_')]

# Final feature set
features = numerical_features + categorical_features
features = [f for f in features if f in train_processed.columns and f not in exclude_cols]

print(f"Selected {len(features)} features for modeling.")

# Prepare training data
X = train_processed[features]
y = train_processed['Item_Outlet_Sales']

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Building
print("\n== Model Building ==")

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor.XGBRegressor(random_state=42)
}

# Evaluate models
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {'Validation RMSE': val_rmse, 'CV RMSE': cv_rmse}
    print(f"{name} - Validation RMSE: {val_rmse:.4f}, CV RMSE: {cv_rmse:.4f}")

# Find best model
best_model_name = min(results, key=lambda k: results[k]['CV RMSE'])
print(f"\nBest model: {best_model_name} with CV RMSE: {results[best_model_name]['CV RMSE']:.4f}")

# 6. Hyperparameter Tuning
print("\n== Hyperparameter Tuning for Best Model ==")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestRegressor(random_state=42)
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    best_model = GradientBoostingRegressor(random_state=42)
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    best_model = XGBRegressor.XGBRegressor(random_state=42)
else:
    # LinearRegression doesn't have many hyperparameters to tune
    param_grid = {}
    best_model = LinearRegression()

if param_grid:
    print(f"Tuning hyperparameters for {best_model_name}...")
    random_search = RandomizedSearchCV(
        best_model, param_distributions=param_grid, 
        n_iter=20, scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1
    )
    random_search.fit(X, y)
    
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")
    
    final_model = random_search.best_estimator_
else:
    print("Skipping hyperparameter tuning for Linear Regression.")
    final_model = LinearRegression().fit(X, y)

# 7. Feature Importance Analysis (if applicable)
print("\n== Feature Importance Analysis ==")

if hasattr(final_model, 'feature_importances_'):
    importances = final_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("Top 10 most important features:")
    for i in range(min(10, len(features))):
        print(f"{features[indices[i]]}: {importances[indices[i]]:.4f}")
elif hasattr(final_model, 'coef_'):
    coefficients = final_model.coef_
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
    coef_df = coef_df.reindex(coef_df.Coefficient.abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Coefficients')
    plt.barh(range(len(coef_df)), coef_df['Coefficient'], align='center')
    plt.yticks(range(len(coef_df)), coef_df['Feature'])
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig('feature_coefficients.png')
    plt.close()
    
    print("Top 10 features by coefficient magnitude:")
    print(coef_df.head(10))

# 8. Final Prediction on Test Data
print("\n== Making Predictions on Test Data ==")

# Prepare test features
X_test = test_processed[features]

# Generate predictions
test_predictions = final_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Item_Identifier': test_processed['Item_Identifier'],
    'Outlet_Identifier': test_processed['Outlet_Identifier'],
    'Item_Outlet_Sales': test_predictions
})

submission.to_csv('bigmart_sales_prediction_submission.csv', index=False)
print("Submission file generated successfully.")

print("\n== Complete ==")