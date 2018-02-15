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
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

# read data
train         = pd.read_csv('data//train.csv', index_col='id')
severity_type = pd.read_csv('data//severity_type.csv', index_col='id')
event_type    = pd.read_csv('data//event_type.csv')
log_feature   = pd.read_csv('data//log_feature.csv')
resource_type = pd.read_csv('data//resource_type.csv')
test          = pd.read_csv('data//test.csv', index_col='id')


# convert event_type from long to wide & impute
event_wide = event_type.pivot(index='id', columns='event_type', values='id')
event_wide =  event_wide.fillna(0)

# convert resource_type from long to wide & impute
resource_wide = resource_type.pivot(index='id', columns='resource_type', values='id')
resource_wide = resource_wide.fillna(0)

# convert log_features from long to wide using volume as value & impute
log_feature_wide = log_feature.pivot(index='id', columns='log_feature', values='volume')
log_feature_wide = log_feature_wide.fillna(0)

# join datasets
train = train.join([severity_type, event_wide, resource_wide, log_feature_wide])

# convert categorical variables
# code found at: https://stackoverflow.com/questions/28910851/python-pandas-changing-some-column-types-to-categories
cat_vars = ['location', 'severity_type']
for col in cat_vars:
    train[col] = train[col].astype('category')

train = pd.get_dummies(train, drop_first=True)

# split into X and y
X = train.drop('fault_severity', axis=1)
y = train[['fault_severity']]

y = y.squeeze() # convert 1D array https://stackoverflow.com/questions/34337093/why-am-i-getting-a-data-conversion-warning

# split into training & validation
X_train, X_valid, y_train, y_valid = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.20, random_state=3339)

# modeling building/comparison

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', RandomForestClassifier())
])

pipeline = Pipeline([
    ('clf', RandomForestClassifier())
])

'''
parameters = {'clf__n_estimators': [500],
			  'clf__max_features': ('auto', 'log2'),
			  'clf__max_depth': range(1,11),
			  'clf__max_features': range(1,6),
			  'clf__bootstrap': (True, False)
			  }
'''
parameters = {'clf__n_estimators': [10, 100, 500],
              'clf__max_features': ('auto', 'log2')
			  }


# Used this resource for helping set up Pipeline
# https://www.kaggle.com/giovannibruner/randomforest-with-gridsearchcv
# cv = GridSearchCV(pipeline, parameters, scoring='neg_log_loss', verbose=1)
cv = GridSearchCV(pipeline, parameters, scoring='neg_log_loss', verbose=1)

# fit model on training data
cv.fit(X_train, y_train)

# predict on validation data
y_preds = cv.predict(X_valid)

# score model
print(cv.score(X_valid, y_valid))
print(cv.best_params_)

# save best model
final_model = cv.best_estimator_

# pickle model
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
pickled_file = 'models//random_forest_regressor_3.sav'
joblib.dump(final_model, pickled_file)

# load model
# final_model = joblib.load(pickled_file)

# merge datasets with test data
test = test.join([severity_type, event_wide, resource_wide, log_feature_wide])

for col in cat_vars:
    test[col] = test[col].astype('category')

test = pd.get_dummies(test, drop_first=True)

# ensure matching columns in train and test
# https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data

# Get missing columns in the training test
missing_cols = set( X.columns ) - set( test.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    test[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
test = test[X.columns]

X_test = test.as_matrix()

# create model on full training set
final_model.fit(X.as_matrix(), y.as_matrix())

# predict on test dataset
y_preds_final = final_model.predict(X_test)

# output results
predict_0 = []
predict_1 = []
predict_2 = []

for pred in y_preds_final:
	if pred == 0:
		predict_0.append(1)
		predict_1.append(0)
		predict_2.append(0)
	elif pred == 1:
		predict_0.append(0)
		predict_1.append(1)
		predict_2.append(0)
	else:
		predict_0.append(0)
		predict_1.append(0)
		predict_2.append(1)

test_preds_data = {'id': test.index,
					'predict_0': predict_0,
					'predict_1': predict_1,
					'predict_2': predict_2
					}

submission = pd.DataFrame(data=test_preds_data)

# submit predictions
submission.to_csv('kaggle_submission3.csv', index=False)
