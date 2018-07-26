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

train_input = 'F:/to_xy.csv'
# data = pd.read_csv(train_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
data = pd.read_csv(train_input, sep=',', header=0,iterator=True, chunksize=200000,nrows=100000)
data = data.get_chunk(6000000)
data = data.fillna(0)
data.head(5)

train_y = data.flag
train_x = data.drop(['flag'], axis=1)



clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=500, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
)
clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=100)

y_train_pred = clf.predict(train_x)
y_train_pred = pd.DataFrame(y_train_pred)
train_report = metrics.classification_report(train_y, y_train_pred)
print(train_report)

#   precision    recall  f1-score   support
#           0       0.99      1.00      0.99     97720
#           1       0.87      0.49      0.62      2280
# avg / total       0.99      0.99      0.98    100000
