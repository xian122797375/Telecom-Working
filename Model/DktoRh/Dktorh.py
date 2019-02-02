#!/usr/bin/python
# -*- coding: utf-8 -*-
'''2018-12-10资采 不限量迁转 单k转不限量模型'''
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

chunk_size = 200000
data_cnt = 5000000


#----------------------------------------------功能区------------------------------------------------------#
def chunk_read_data(file_path, chunk_size, data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path, sep='	', header=0, iterator=True, chunksize=chunk_size, low_memory=False,encoding='gb18030')
    train = data.get_chunk(data_cnt)
    print('文件读取完毕,总计{}条'.format(train.shape[0]))
    return train


def Fix_Missing(Enter_Date, Model=False):
    '''MODEL等于Falst表示自动模式，自动缺失补0；等于True表示智能模式补0.1或其他(功能待完成)'''
    if Model == False:
        Enter_Date = Enter_Date.fillna(0)
        Enter_Date.replace('?', 0, inplace=True)
    return Enter_Date


def Data_Var_Convert(input_data):
    '''删除唯一值变量，自动筛选分类变量,80%单一变量值数据进行哑变量处理'''
    categorical_feature = []
    # new_train = input_data.drop([label], axis=1)
    new_train = input_data
    for i in new_train.columns:
        Category_Count = pd.DataFrame(new_train.groupby(i).size().sort_values(ascending=False), columns=['cnt'])  # 列分类统计
        if len(Category_Count) == 1:
            new_train = new_train.drop([i], axis=1)  # 删除唯一值
        elif (len(Category_Count) < 300) & (len(Category_Count) > 1):
            Category_Count_Top1 = Category_Count.iloc[0, 0]
            Category_Count_sum = Category_Count.cnt.sum(axis=0)
            Top1_Bit = Category_Count_Top1 / Category_Count_sum
            if Top1_Bit >= 0.95:
                new_train = new_train.drop([i], axis=1)  # 一个维度值占比较大剔除
            elif (Top1_Bit >= 0.8) & (
                        Top1_Bit < 0.95):
                new_train.loc[new_train[i] != 0, i] = 1  # 大于0.8转化为哑变量
                categorical_feature.append(i)  # 假定分类变量
            else:
                categorical_feature.append(i)  # 假定分类变量
    print('原始维度{}个,剔除后还剩下{}个'.format(input_data.shape[1], new_train.shape[1]))
    print('自动判断分类维度共计：{}个'.format(len(categorical_feature)))
    return categorical_feature, new_train


def Recoding_Cat_Data(data, feature):
    '''对分类变量进行重新编码'''
    feature_new = []
    print('共需要对{}列维度进行重新编码'.format(len(feature)))
    for i in feature:
        Size_count = pd.DataFrame(data.groupby(i).size().sort_values(ascending=False), columns=['cnt'])  # 列分类统计
        for j in range(len(Size_count)):
            data.loc[data[i] == Size_count.index[j], '{}'.format(i + '_New')] = j + 1
        data = data.drop([i], axis=1)
        feature_new.append(i + '_New')
    data = data.fillna(0)
    print('编码完成')
    return data, feature_new

def RematchingDate(train,bit):
    count = train[train['LABEL_3'] == 'T'].shape[0]
    data = train[train['LABEL_3'] == 'F'].iloc[:count * bit,:]
    new_data_train = pd.concat([train[train['LABEL_3'] == 'T'], data], axis=0)
    return new_data_train
#--------------------------------------------主程序区------------------------------------------------------#
train_path = '10test_dktorh.txt'
train = chunk_read_data(train_path, chunk_size, data_cnt)
train = train.iloc[:,3:]


#--------------------------------------------读取测试数据-----------------------
test_path = '11test_dktorh.txt'
test = chunk_read_data(test_path, chunk_size, data_cnt)
test = test.iloc[:,3:]

#--------------------------------------------读取最新数据-----------------------
new_test_path = '12test_dktorh.txt'
new_test = chunk_read_data(new_test_path, chunk_size, data_cnt)
Prd_Inst_Id = new_test.iloc[:,0]
new_test = new_test.iloc[:,3:]

#--------------------------数据合并-----------------------
train = RematchingDate(train,2)

train_test_data = pd.concat([train, test, new_test], axis=0)

train_test_data = Fix_Missing(train_test_data)



# drop_var = 'Exp_Date','LABEL_1','LABEL_2','LABEL_3'
train_test_data = train_test_data.drop(['LABEL_1','LABEL_2','LABEL_3'], axis=1)
# train_test_data['Line_Rate_New'] = train_test_data['Line_Rate'].str.split('M').str[0]
# train_test_data
train_test_data = train_test_data.drop(['Ofr_Name'], axis=1)

# train_x = train_x.drop(['Exp_Date'], axis=1)
# label = ''
cat_feature = []
cat_feature, train_test_data = Data_Var_Convert(train_test_data)
print(cat_feature)

# cat_feature.append('Line_Rate')
train_test_data, cat_feature = Recoding_Cat_Data(train_test_data, cat_feature)

length1 = train.shape[0]
length2 = test.shape[0]
train_x = train_test_data.iloc[:length1]
test_x = train_test_data.iloc[length1:length1 + length2]
new_test_x = train_test_data.iloc[length1 + length2:]

train_x = pd.DataFrame(train_x)
test_x = pd.DataFrame(test_x)
new_test_x = pd.DataFrame(new_test_x)

train_y = train['LABEL_3']
test_y = test['LABEL_2']


# 连续变量转化float
obname = train_x.select_dtypes(include=["object"]).columns
for col in obname:
    train_x[col] = train_x[col].astype(np.float)
    test_x[col] = test_x[col].astype(np.float)
    new_test_x[col] = new_test_x[col].astype(np.float)
# 分类变量转化int
for col in cat_feature:
    train_x[col] = train_x[col].astype(np.int)
    test_x[col] = test_x[col].astype(np.int)
    new_test_x[col] = new_test_x[col].astype(np.int)

train_y = train_y.replace('T','1')
train_y = train_y.replace('F','0')

test_y = test_y.replace('T','1')
test_y = test_y.replace('F','0')

print (train_x.head(5),train_x.shape)
print (train_y.head(5),train_y.shape)

#-------------------------------训练集测试集数据绑定-----------------------#
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)

# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train,free_raw_data=False)


#----------------------------------调参数--------------------------#
### 设置初始参数--不含交叉验证参数
print('设置参数')
params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'binary_logloss', 'auc'},
            'num_threads': 40
          }

### 交叉验证(调参)
print('交叉验证')
min_merror = float('Inf')
best_params = {}


# 准确率
print("调参1：提高准确率")
for num_leaves in range(20,200,5):
    for max_depth in range(3,8,1):
        params['num_leaves'] = num_leaves
        params['max_depth'] = max_depth
        cv_results = lgb.cv(
                            params,
                            lgb_train,
                            seed=2018,
                            nfold=3,
                            metrics=['binary_error'],
                            early_stopping_rounds=10,
                            verbose_eval=True
                            )
        mean_merror = pd.Series(cv_results['binary_error-mean']).min()
        boost_rounds = pd.Series(cv_results['binary_error-mean']).argmin()
        if mean_merror < min_merror:
            min_merror = mean_merror
            best_params['num_leaves'] = num_leaves
            best_params['max_depth'] = max_depth

params['num_leaves'] = best_params['num_leaves']
params['max_depth'] = best_params['max_depth']

print(params)
#
#
#
# parameters = {
#               'max_depth': [6,7,8],
#               'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.15],
#               'feature_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#               'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 0.95],
#               'bagging_freq': [2, 4, 5, 6, 8],
#               'lambda_l1': [0, 0.1, 0.4, 0.5, 0.6],
#               'lambda_l2': [0, 10, 15, 35, 40],
#               'cat_smooth': [1, 10, 15, 20, 35]
# }
# gbm = lgb.LGBMClassifier(boosting_type='gbdt',
#                          objective = 'binary',
#                          metric = 'auc',
#                          verbose = 0,
#                          learning_rate = 0.01,
#                          num_leaves = 113,
#                          feature_fraction=0.8,
#                          bagging_fraction= 0.9,
#                          bagging_freq= 8,
#                          lambda_l1= 0.6,
#                          lambda_l2= 0)
# # 有了gridsearch我们便不需要fit函数
# gsearch = GridSearchCV(gbm, param_grid=parameters, scoring='accuracy', cv=3)
# gsearch.fit(x_train, y_train)
#
# print("Best score: %0.3f" % gsearch.best_score_)
# print("Best parameters set:")
# best_parameters = gsearch.best_estimator_.get_params()


#
# #--------------------------------------------算法模型搭建------------------------------------------------------#
clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=115, reg_alpha=0.0, reg_lambda=1,
    max_depth=7, n_estimators=2500, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1,bagging_fraction = 0.8 ,feature_fraction=0.8,
)
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=400)
# clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=100)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)

test_y_pred = clf.predict(test_x)
test_y_report = metrics.classification_report(test_y, test_y_pred)
print(test_y_report)

model_path = '20180910_DZR_1bi2.model'
joblib.dump(clf, model_path)

y_new_test_pred = clf.predict_proba(new_test_x)
y_new_test_pred = pd.DataFrame(y_new_test_pred ,columns=['zero_prob','one_prob'])

prd_inst_id = Prd_Inst_Id
result = pd.concat([prd_inst_id, y_new_test_pred.one_prob], axis=1)
# result = result[result.one_prob >= 0.5]
print(result)
# result2 = result.iloc[:,0]
result.to_csv('rh_09to11dktorh_result_1bi2.csv',index=False)

#               precision    recall  f1-score   support
#
#           0       0.80      0.68      0.73    227872
#           1       0.72      0.83      0.77    227944
#
# avg / total       0.76      0.75      0.75    455816
#
#              precision    recall  f1-score   support
#
#           0       0.78      0.66      0.71     57013
#           1       0.70      0.81      0.75     56941
#
# avg / total       0.74      0.73      0.73    113954


# precision    recall  f1-score   support
#
#           0       0.97      0.64      0.77   1843464
#           1       0.17      0.81      0.28    164162
#
# avg / total       0.91      0.65      0.73   2007626

# clf = joblib.load('20180910_DZR.model')


feature_importances = pd.DataFrame({'Feature_name': x_train.columns,
                                        'Importances': clf.feature_importances_})
feature_importances = feature_importances.set_index('Feature_name')
feature_importances = feature_importances.sort_values(by='Importances', ascending=False)
print('Feature importances:', feature_importances.head(20))

a = len(feature_importances) * 7 // 10



train_x = train_x[feature_importances.index[0:a]]
test_x = test_x[feature_importances.index[0:a]]
new_test_x = new_test_x[feature_importances.index[0:a]]

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3)

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=115, reg_alpha=0.0, reg_lambda=1,
    max_depth=7, n_estimators=2500, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1,bagging_fraction = 0.8 ,feature_fraction=0.8,
)
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=400)
# clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=100)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)

test_y_pred = clf.predict(test_x)
test_y_report = metrics.classification_report(test_y, test_y_pred)
print(test_y_report)

model_path = '20180910_DZR_1bi23.model'
joblib.dump(clf, model_path)

y_new_test_pred = clf.predict_proba(new_test_x)
y_new_test_pred = pd.DataFrame(y_new_test_pred ,columns=['zero_prob','one_prob'])

prd_inst_id = Prd_Inst_Id
result = pd.concat([prd_inst_id, y_new_test_pred.one_prob], axis=1)
# result = result[result.one_prob >= 0.5]
print(result)
# result2 = result.iloc[:,0]
result.to_csv('rh_09to11dktorh_result_1bi3.csv',index=False)

# telecom_area_id_New              2511
# Acct_Bal_Amt                     2152
# Innet_Dur_New                    2010
# Ofr_Id                           1911
# Age_New                          1645
# Inv_Amt                          1431
# Use_Mons_New                     1179
# Latn_Id_New                      1125
# Net_Kbyte_Trend                   909
# Net_Dur_Trend                     833
# R3A_Net_Dur                       732
# R3A_Wday_Net_Dur                  675
# R3A_Avgday_Net_Kbyte              657
# R3A_Avgday_Net_Dur                626
# R3A_AvgWday_Net_Dur               611
# R3A_Hday_Net_Dur                  585
# R3A_Day_Net_Dur                   564
# R3A_Net_Kbyte                     549
# R3A_Wday_Day_Net_Dur              541
# R3A_Day_Net_Dur_Rate_New          539



params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 40,
    'learning_rate': 0.1,
    'num_leaves': 115,
    'max_depth': 7,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'feature_fraction' : 0.8,
    'n_estimators' : 311
}
params_test1={'max_depth': range(3,11,2),    'num_leaves':range(50, 170, 30)}

cv_results = lgb.cv(params, lgb_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics=['binary_error'],
                    early_stopping_rounds=50, seed=0)
print('best n_estimators:', len(cv_results['binary_error-mean']))
print('best cv score:', pd.Series(cv_results['binary_error-mean']).max())


params_test1 = {'max_depth': range(3, 8, 1), 'num_leaves': range(5, 180, 10)}

gsearch1 = GridSearchCV(estimator=lgb.LGBMClassifier(boosting_type='gbdt', objective='binary', metrics='auc', learning_rate=0.1,
                                 n_estimators=311, max_depth=7, bagging_fraction=0.8, feature_fraction=0.8),param_grid=params_test1, scoring='roc_auc', cv=5, n_jobs=-1)
gsearch1.fit(x_train, y_train)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_



params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',

    'learning_rate': 0.1,
    'num_leaves': 113,
    'max_depth': 7,

    'subsample': 0.8,
    'colsample_bytree': 0.8,
    }


cv_results = lgb.cv(
    params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
    early_stopping_rounds=50, verbose_eval=50, show_stdv=True, seed=0)

print('best n_estimators:', len(cv_results['rmse-mean']))
print('best cv score:', cv_results['rmse-mean'][-1])
