'''
	Kyle Zeberlein
	Telstra Network Disruptions
	https://www.kaggle.com/c/telstra-recruiting-network
'''

# imports
import pandas as pd
import matplotlib.pyplot as plt

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

# explore data
print(train.columns)

# clean data
