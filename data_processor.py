'''
:4
Due to row is another different features, split it to exploit features python file to run separately instead.

:3
Updated, There is exploit happened with row_id in gender_age_test and gender_age_train.
row_id can be use to improve overall accuracy. Normalize row_id

:2
Event triggered times (by hour)

:1
First version, Brand, Device, App, Label features
'''


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack


def save_sparse(filename, xmtr):
    np.savez(filename,data = xmtr.data ,indices= xmtr.indices,
             indptr =xmtr.indptr, shape=xmtr.shape )

def load_sparse(filename):
    tmp = np.load(filename)
    return csr_matrix((tmp['data'], tmp['indices'], tmp['indptr']), shape= tmp['shape'])

output = 'WorkSpace/'

train = pd.read_csv("Data/gender_age_train.csv", index_col='device_id')
test = pd.read_csv("Data/gender_age_test.csv", index_col='device_id')
phone_data = pd.read_csv("Data/phone_brand_device_model.csv").drop_duplicates('device_id',keep='first').set_index('device_id')
events = pd.read_csv("Data/events.csv", index_col='event_id')
app_events = pd.read_csv("Data/app_events.csv", usecols=['event_id','app_id','is_active'], dtype={'is_active': bool}, index_col='event_id')
app_labels = pd.read_csv("Data/app_labels.csv")
label_cate = pd.read_csv('Data/label_categories.csv')

print 'total train records: % d' % len(train)
print 'total test records: % d' % len(test)
print 'phone data after removed duplicates: %d ' % len(phone_data)

train['train_index'], test['test_index'] = np.arange(train.shape[0]), np.arange(test.shape[0])
train = train.drop(['gender','age'], axis=1)

y_encoder = LabelEncoder().fit(train.group)
train['group'] = y_encoder.transform(train.group)
yclasses_ = len(y_encoder.classes_)
print 'total number of target classes: ', yclasses_, y_encoder.classes_

""" Join App and label """
app_labels = app_labels.loc[app_labels.app_id.isin(app_events.app_id.unique())]

app_encoder = LabelEncoder().fit(app_events.app_id)
label_encoder = LabelEncoder().fit(app_labels.label_id)
app_events['app'] = app_encoder.transform(app_events.app_id)
app_labels['app'] = app_encoder.transform(app_labels.app_id)
app_labels['label'] = label_encoder.transform(app_labels.label_id)
app_id_to_label = app_labels.join(label_cate, how='left', on='label_id', lsuffix='k')[['app_id','app','label','category']]
joined_app_events = app_events.join(events)
joined_app_events = joined_app_events.groupby(['device_id','app'])['app'].agg(['size'])
joined_app_events = ((joined_app_events.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()
""" Label like 1,2,3,4,5,6, ... """
joined_labels = joined_app_events.merge(app_labels[['app','label']])[['device_id','label','app']]
joined_labels = joined_labels.groupby(['device_id','label'])['app'].agg(['size'])
joined_labels = ((joined_labels.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()


""" 1 - Brand """
brand_encoder = LabelEncoder().fit(phone_data.phone_brand)
phone_data['converted_brand_only'] = brand_encoder.transform(phone_data.phone_brand)

train['brand_only'] = phone_data['converted_brand_only']
test['brand_only'] = phone_data['converted_brand_only']

sparse_train_brand_only = csr_matrix((np.ones(train.shape[0]), (train.train_index, train.brand_only)))
sparse_test_brand_only = csr_matrix((np.ones(test.shape[0]), (test.test_index, test.brand_only)))


""" 2 - Event time """
events['hour'] = pd.to_datetime(events['timestamp']).dt.hour
stampping = events.groupby(['device_id','hour'])['timestamp'].agg(['size'])
stampping = ((stampping.join(train['train_index'], how='left')).join(test['test_index'], how='left')).reset_index()
d = stampping.dropna(subset=['train_index'])
sparse_event_time_train = csr_matrix((np.ones(d.shape[0]), (d.train_index, d.hour)), shape=(train.shape[0], 24))
d = stampping.dropna(subset=['test_index'])
sparse_event_time_test = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.hour)), shape=(test.shape[0], 24))

""" 3 - Device model """
model = phone_data.phone_brand.str.cat(phone_data.device_model)

phone_encoder = LabelEncoder().fit(model)
phone_data['converted_brand'] = phone_encoder.transform(model)

train['brand'] = phone_data['converted_brand']
test['brand'] = phone_data['converted_brand']

sparse_train_brand = csr_matrix((np.ones(train.shape[0]), (train.train_index, train.brand)))
sparse_test_brand = csr_matrix((np.ones(test.shape[0]), (test.test_index, test.brand)))

""" 4 - Bag of Apps """
d = joined_app_events.dropna(subset=['train_index'])
sparse_train_appusage = csr_matrix((np.ones(d.shape[0]), (d.train_index, d.app)), shape=(train.shape[0],len(app_encoder.classes_)))
d = joined_app_events.dropna(subset=['test_index'])
sparse_test_appusage = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.app)), shape=(test.shape[0],len(app_encoder.classes_)))

""" 5 - Bag of Label """
d = joined_labels.dropna(subset=['train_index'])
sparse_train_label = csr_matrix((np.ones(d.shape[0]), (d.train_index,d.label)), shape=(train.shape[0], len(label_encoder.classes_)))
d = joined_labels.dropna(subset=['test_index'])
sparse_test_label = csr_matrix((np.ones(d.shape[0]), (d.test_index,d.label)),shape=(test.shape[0], len(label_encoder.classes_)))


""" Save different features into different files """
save_sparse(output + 'sprm_brand_train',sparse_train_brand)
save_sparse(output + 'sprm_brand_test',sparse_test_brand)
save_sparse(output + 'sprm_label_train', sparse_train_label)
save_sparse(output + 'sprm_label_test', sparse_test_label)
save_sparse(output + 'sparse_train_appusage', sparse_train_appusage)
save_sparse(output + 'sparse_test_appusage', sparse_test_appusage)
save_sparse(output + 'sparse_event_time_train', sparse_event_time_train)
save_sparse(output + 'sparse_event_time_test', sparse_event_time_test)
save_sparse(output + 'sparse_train_brand_only', sparse_train_brand_only)
save_sparse(output + 'sparse_test_brand_only', sparse_test_brand_only)

""" Stack full sparses """
Xtrain = hstack((sparse_train_brand_only, sparse_train_brand,  sparse_train_appusage, sparse_train_label,sparse_event_time_train), format='csr')
Xtest =  hstack((sparse_test_brand_only, sparse_test_brand, sparse_test_appusage, sparse_test_label, sparse_event_time_test), format='csr')

save_sparse('WorkSpace/sprm_train_additional', Xtrain)
save_sparse('WorkSpace/sprm_test_v2', Xtest)
