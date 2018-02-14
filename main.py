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

# merge train and log_feature
train = train.merge(log_feature, on='id')

# merge train and resource_type
train = train.merge(resource_type, on='id')

# join datasets
train = train.join([severity_type, event_wide])

# explore data
print(train.columns)
print(event_type.columns)


print(train.head())
print(event_type.head())

# clean data
