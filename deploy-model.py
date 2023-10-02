import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier

data = pd.read_csv('data/stock_data_cleaned.csv')

# Remove Inf
rows_with_inf = data[(data['5-day Volume Change'].isin([float('inf'), float('-inf')])) | 
                     (data['Price-to-Volume'].isin([float('inf'), float('-inf')]))].index

data = data.drop(rows_with_inf)

# Test Feature Importance
feature_data = data.dropna()

X = feature_data.drop(columns=['Outperformed_Predicted_Next_Week', 'Date', 'Symbol',
                       'Monday Open', '%Change', '%Change_SP500', 
                       'Day_of_Week'])  

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

## Create Voting Model
# Get last Friday date
today = dt.date.today()
days_since_last_friday = (today.weekday() - 4) % 7
last_friday = (today - dt.timedelta(days=days_since_last_friday)).strftime('%Y-%m-%d')

# Resolve NA values
data.dropna(subset=features, inplace=True)

data['Outperformed_Predicted_Next_Week'] = data['Outperformed_Predicted_Next_Week'].fillna(0)

target = 'Outperformed_Predicted_Next_Week'

train = data.loc[data['Date'] < last_friday]
test = data.loc[data['Date'] >= last_friday]

X_train = train[features]
y_train = train[target].values.ravel()

X_test = test[features]
y_test = test[target].values.ravel()

classifiers = {
    "Logistic Regression": LogisticRegression(C=0.1, 
                                              penalty = 'l1', 
                                              solver = 'liblinear',
                                              max_iter=10000, 
                                              random_state=1),
    "Calibrated SVM (CCV)": CalibratedClassifierCV(estimator=SVC(probability=True), method='isotonic', cv=3),
    "Gradient Boosting": GradientBoostingClassifier(subsample = 0.9, 
                                                    n_estimators=200, 
                                                    min_samples_split=2, 
                                                    min_samples_leaf=4,
                                                    max_depth=10,
                                                    learning_rate=0.01,
                                                    random_state=1),
}

voting_clf = VotingClassifier(estimators=list(classifiers.items()), voting='soft')

voting_clf.fit(X_train, y_train)

predicted = voting_clf.predict_proba(X_test)[:,1]

## Dataframe of results
results_df = test.copy()

results_df['Predicted'] = predicted

results_df.sort_values('Predicted', ascending=False, inplace=True)

results_df.to_csv(f'Stock Predictions/{last_friday}results.csv', index=False)