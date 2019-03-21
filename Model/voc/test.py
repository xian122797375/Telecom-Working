'''2018-10-18 电渠 语音包模型'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
# import xgboost as xgb
# from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV

#-----------------------------------------读取文件---------------
train = pd.read_csv('F:/telecom_train.csv', sep=',', header=0, low_memory=False)


test = pd.read_csv('F:/telecom_test.csv', sep=',', header=0, low_memory=False)
subscriberID = test.iloc[:,0]
#-----------------------------------------提取x\y---------------
train_x = train.iloc[:,2:]
print(train_x.columns)#检查训练维度是否正确　
train_y = train['churn']

test_x = test.iloc[:,1:]

#----开始训练---
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=400)

#----观察效果--
y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)

#-----开始预测并输出--
test_y_pred = clf.predict(test_x)

result = pd.concat([subscriberID, pd.DataFrame(test_y_pred)], axis=1)
result.to_csv('F:/111.csv',index=False)

#-----输出特征重要--
# feature_importances = pd.DataFrame({'Feature_name': x_train.columns,
#                                         'Importances': clf.feature_importances_})
# feature_importances = feature_importances.set_index('Feature_name')
# feature_importances = feature_importances.sort_values(by='Importances', ascending=False)
# print('Feature importances:', feature_importances.head(20))
#
# a = len(feature_importances) * 7 // 10




