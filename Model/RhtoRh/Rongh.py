'''2018-5-9资采 不限量迁转 融合转融合模型'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import os
from sqlalchemy import types, create_engine
import random as rd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import time

# train_input = 'F:/01rh_date.txt'
train_input = 'F:/03train.txt'
# train_input = '../data1/zicai/12train.txt'
# train_input = 'F:/03dantrain.txt'
data = pd.read_csv(train_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
data = data.get_chunk(6000000)
data = data.fillna(0)
data.head(5)

# Index(['Prd_Inst_Id', 'Accs_Nbr', 'Latn_Id', 'Strategy_Segment_name',
#        'gender_name', 'Age', 'Term_Type_Id', 'inv_bill_amt1', 'inv_bill_amt2',
#        'inv_bill_amt3', 'inv_bill_amt4', 'O_Roam_Amt', 'total_flux1',
#        'total_flux2', 'total_flux3', 'total_flux4', 'T_Call_Dur', 'Call_Dur',
#        'Pckg_Voice_Dur1', 'Pckg_Voice_Dur2', 'Pckg_Voice_Dur3',
#        'Pckg_Voice_Dur4', 'Sms_Cnt1', 'Extrem_Base_Flux',
#        'Telecom_Ftp_3g_4g_Id', 'Jt_Lte_Flag', 'Prom_Amt', 'zf_flag', 'val1',
#        'val2', 'val3', 'Innet_Dur', 'label_1', 'label_2', 'label_3'],
#       dtype='object')


data['inv_bill_amt1'].quantile([.25, .5, 0.75])
data['total_flux1'].quantile([.25, .5, 0.75])

new_data = data[(data['inv_bill_amt1'] > 40) & (data['total_flux1'] > 1000)]


count1 = new_data[new_data.label_3 == 1].shape[0]
count2 = data[data.label_3 == 1].shape[0]
print(count1,count2)


def RematchingDate(train,bit):
    count = train[train.label_3 == 1].shape[0]
    data = train[train.label_3 == 0].iloc[:count * bit,:]
    new_data_train = pd.concat([train[train.label_3 == 1], data], axis=0)
    return new_data_train


train = RematchingDate(data,1)
train_x = new_data.iloc[:,2:-3]
train_y = new_data.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=5000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
)
clf.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric='auc', early_stopping_rounds=100)
# clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=100)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)
model_path = 'F:/lgb1bitest111.model'

joblib.dump(clf, model_path)


time_start=time.time()
model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=5,
                              min_child_weight=3,
                              gamma=0.3,
                              subsample=0.85,
                              colsample_bytree=0.75,
                              objective='binary:logistic',
                              scale_pos_weight=1,
                              seed=27,
                              nthread=12,
                              reg_alpha=0.0005)
model.fit(x_train, y_train)
model_path = '1bi1.model'

joblib.dump(model, model_path)


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_report = metrics.classification_report(y_train,y_train_pred)
test_report = metrics.classification_report(y_test,y_test_pred)
print(train_report)
print(test_report)
time_end = time.time()
print('总时间',time_end-time_start)

# train_input = 'F:/01rh_date.txt'
# test_input = '../data1/zicai/02test.txt'
# test_input = 'F:/05danTEST.txt'
test_input = 'F:/05TEST.txt'
test = pd.read_csv(test_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
test = test.get_chunk(6000000)
test = test.fillna(0)

new_data = data[(data['inv_bill_amt1'] > 40) & (data['total_flux1'] > 1000)]
train_x = test.iloc[:,2:-3]
train_y = test.iloc[:,-2]

model = clf
y_train_pred = model.predict(train_x)

train_report = metrics.classification_report(train_y,y_train_pred)

print(train_report)


# //------------结果数据-------------
# train_input = 'F:/01rh_date.txt'
# test_input = 'F:/02rh_date.txt'
test_input = 'F:/06danTEST.txt'
test = pd.read_csv(test_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
test = test.get_chunk(4000000)
test = test.fillna(0)

prd_inst_id = test.iloc[:,0:2]

train_x = test.iloc[:,2:-3]
train_y = test.iloc[:,-1]


y_train_pred = model.predict(train_x)

y_train_pred = pd.DataFrame(y_train_pred , columns=['flag'])

result = pd.concat([prd_inst_id, y_train_pred], axis=1)
result = result[result.flag == 1]
print(result)
result2 = result.iloc[:,0]
result2.to_csv('F:/dc_03to06_result.csv',index=False)


# //------------结果数据-2------------
test_input = 'F:/06danTEST10041006.txt'
test = pd.read_csv(test_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
test = test.get_chunk(4000000)
test = test.fillna(0)

prd_inst_id = test.iloc[:,0:2]

train_x = test.iloc[:,2:-3]
train_y = test.iloc[:,-1]

y_train_pred = model.predict(train_x)
y_train_pred = pd.DataFrame(y_train_pred , columns=['flag'])

result = pd.concat([prd_inst_id, y_train_pred], axis=1)
result = result[result.flag == 1]
print(result)
result2 = result.iloc[:,0]
result2.to_csv('F:/dc_03to06_result10041006.csv',index=False)




train_input = 'F:/021bi1.txt'
train = pd.read_csv(train_input, sep='	', header=0,iterator=True, chunksize=200000,encoding='gb2312')
train = train.get_chunk(4000000)
train = train.fillna(0)


train_x = train.iloc[:,2:-3]
train_y = train.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)


model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=5,
                              min_child_weight=3,
                              gamma=0.3,
                              subsample=0.85,
                              colsample_bytree=0.75,
                              objective='binary:logistic',
                              scale_pos_weight=1,
                              seed=27,
                              nthread=12,
                              reg_alpha=0.0005)
model.fit(x_train, y_train)
model_path = 'F:/zirh02.model'
joblib.dump(model, model_path)

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)
train_report = metrics.classification_report(y_train,y_train_pred)
test_report = metrics.classification_report(y_test,y_test_pred)
print(train_report)
print(test_report)

# train_input = 'F:/01rh_date.txt'
test_input = 'F:/04rh_date.txt'
# test_input = 'F:/04rh_date.txt'
test = pd.read_csv(test_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
test = test.get_chunk(4000000)
test = test.fillna(0)

train_x = test.iloc[:,2:-3]
train_y = test.iloc[:,-1]

y_train_pred = model.predict(train_x)

y_train_pred = pd.DataFrame(y_train_pred)

result = pd.concat([prd_inst_id, y_train_pred], axis=1)

result.to_csv('F:/rh_02to04_result.csv')


