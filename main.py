'''
	Kyle Zeberlein
	Telstra Network Disruptions
	https://www.kaggle.com/c/telstra-recruiting-network
'''

# imports
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.emsemble import RandomForestClassifier

# read data
train         = pd.read_csv('data//train.csv', index_col='id')
severity_type = pd.read_csv('data//severity_type.csv', index_col='id')
event_type    = pd.read_csv('data//event_type.csv')
log_feature   = pd.read_csv('data//log_feature.csv')
resource_type = pd.read_csv('data//resource_type.csv')

# convert event_type from long to wide
event_wide = event_type.pivot(index='id', columns='event_type', values='id')

# convert resource_type from long to wide
resource_wide = resource_type.pivot(index='id', columns='resource_type', values='id')

# convert log_features from long to wide using volume as value
log_feature_wide = log_feature.pivot(index='id', columns='log_feature', values='volume')

# join datasets
train = train.join([severity_type, event_wide, resource_wide, log_feature_wide])

# split into X and y
X = train.drop('fault_severity', axis=1)
y = train[['fault_severity']]

# split into training & validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=3339)

# modeling building/comparison

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', RandomForestClassifier())
])

parameters = {'clf__n_estimators': [10, 100, 500],
			  'clf_n_jobs': [-1],
			  'clf__max_features': ['auto', 'log2', 'None']}

cv = GridSearchCV(pipeline, parameters, scoring='neg_log_loss')

cv.fit(X_train, y_train)

y_preds = cv.predict(X_valid)

print('log loss' + str(log_loss(y_valid, y_preds)))


# create model on full training set
cv.fit(X, y)

# predict on test dataset
y_preds = cv.predict(X_test)

# output results
