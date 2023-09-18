import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler



data = pd.read_csv('data/stock_data_cleaned.csv')

# Remove Inf
rows_with_inf = data[(data['5-day Volume Change'].isin([float('inf'), float('-inf')])) | 
                     (data['Price-to-Volume'].isin([float('inf'), float('-inf')]))].index

data = data.drop(rows_with_inf)

# Test Feature Importance
feature_data = data.dropna()

X = feature_data.drop(columns=['Outperformed_Predicted_Next_Week', 'Date', 'Symbol',
                       'Close_5_Days_Later', '5_Day_%Change', 'SP500_5_Day_%Change', 
                       'Outperformed', 'Day_of_Week'])  

y = feature_data['Outperformed_Predicted_Next_Week']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


feature_importances = model.feature_importances_


features_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

sorted_features_df = features_df.sort_values(by='Importance', ascending=False)

sorted_features_df.reset_index(drop=True, inplace=True)

features = sorted_features_df['Feature'].head(23)


# Model Selection

X_train_top = X_train[features]
X_test_top = X_test[features]

# Define classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=1),
    "Random Forest": RandomForestClassifier(random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "Calibrated SVM (CCV)": CalibratedClassifierCV(estimator=SVC(probability=True), method='isotonic', cv=3),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=1)
}

# Train the classifiers and evaluate ROC AUC score
roc_auc_scores = {}

for name, clf in classifiers.items():
    clf.fit(X_train_top, y_train)
    y_pred_proba = clf.predict_proba(X_test_top)[:,1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    roc_auc_scores[name] = roc_auc

roc_auc_scores

## Hyperparameter Tuning
X = data[features]
y = data['Outperformed_Predicted_Next_Week']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

# Logistic Regression
grid_search_lr = GridSearchCV(
    estimator=LogisticRegression(max_iter=10000, random_state=1),
    param_grid=param_grid_lr,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1
)

grid_search_lr.fit(X_train_scaled, y_train)
print(grid_search_lr.best_params_)

# Gradient Boosting
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.001, 0.01, 0.1, 1],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the RandomizedSearchCV for Gradient Boosting
random_search_gb = RandomizedSearchCV(
    estimator=GradientBoostingClassifier(random_state=1),
    param_distributions=param_grid_gb,
    n_iter=20,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    random_state=42
)

# Fit the randomized search on the training data
random_search_gb.fit(X_train_scaled, y_train)

print(random_search_gb.best_params_)

# Create Voting Model
data.dropna(subset=features, inplace=True)

data['Outperformed_Predicted_Next_Week'] = data['Outperformed_Predicted_Next_Week'].fillna(0)

target = 'Outperformed_Predicted_Next_Week'

train = data.loc[data['Date'] < '2023-06-23']
test = data.loc[data['Date'] >= '2023-06-23']

X_train = train[features]
y_train = train[target].values.ravel()

X_test = test[features]
y_test = test[target].values.ravel()


best_classifiers = {
    "Logistic Regression": classifiers["Logistic Regression"],
    "Calibrated SVM (CCV)": classifiers["Calibrated SVM (CCV)"],
    "Gradient Boosting": classifiers["Gradient Boosting"]
}

voting_clf = VotingClassifier(estimators=list(best_classifiers.items()), voting='soft')

voting_clf.fit(X_train, y_train)

predicted = voting_clf.predict_proba(X_test)[:,1]

roc_auc_voting = roc_auc_score(y_test, predicted)

roc_auc_voting

## Dataframe of results
test['Predicted'] = predicted

test.sort_values('Date', inplace=True)

test_results_df = test[test.groupby('Date')['Predicted'].transform(max) == test['Predicted']][['Date','Symbol','Predicted','Outperformed_Predicted_Next_Week']]

test_results_df.to_csv('Stock Predictions/test-results.csv')