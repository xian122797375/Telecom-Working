'''2019-01-30 信用评价'''
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
from feature_selector import FeatureSelector

chunk_size = 20000
data_cnt = 50000

def chunk_read_data(file_path, chunk_size, data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path, sep=',', header=0, iterator=True, chunksize=chunk_size, low_memory=False)
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


#--------------------------------------------主程序区------------------------------------------------------#
train_path = 'F:/Telecom_Working/Model/Credit/train_dataset.csv'
train = chunk_read_data(train_path, chunk_size, data_cnt)
train = train.iloc[:,1:]

#--------------------------------------------读取测试数据-----------------------
# test_path = 'F:/Telecom_Working/Model/Credit/test_dataset.csv'
# test = chunk_read_data(test_path, chunk_size, data_cnt)
# Prd_inst_id = test.iloc[:,0]
# test = test.iloc[:,1:]

#--------------------------------------------特征工程-----------------------------------------------------#
train_features = train.drop(['信用分'], axis=1)
train_labels = train['信用分']
fs = FeatureSelector(data=train_features, labels=train_labels)

fs.identify_missing(missing_threshold = 0.8)
fs.missing_stats.head()

fs.identify_collinear(correlation_threshold = 0.98)

# fs.identify_zero_importance(task = 'regression',
# eval_metric = 'l2',
# n_iterations = 10,
# early_stopping = True)

fs.identify_all(selection_params =
                {'missing_threshold': 0.8,
                 'correlation_threshold': 0.98,
                 'task': 'regression',
                 'eval_metric': 'l2',
                 'cumulative_importance': 0.99})

train_removed = fs.remove(methods = 'all', keep_one_hot=True)
fs.feature_importances.head()

# #--------------------------数据合并-----------------------
# train_test_data = pd.concat([train, test], axis=0)
#
# train_test_data = Fix_Missing(train_test_data)
#
# train_test_data = train_test_data.drop(['信用分'], axis=1)
#
# cat_feature = []
# cat_feature, train_test_data = Data_Var_Convert(train_test_data)
# print(train_test_data.head(5))
#
# train_test_data, cat_feature = Recoding_Cat_Data(train_test_data, cat_feature)
#
#
# Ex_col = []
# obname = train_test_data.select_dtypes(include=["object"]).columns
# for col in obname:
#     try:
#         train_test_data[col] = train_test_data[col].astype(np.float)
#     except:
#         Ex_col.append(col)
#
# print(Ex_col)
#
# train_test_data = train_test_data.drop(Ex_col, axis=1)
#
# length1 = train.shape[0]
# length2 = test.shape[0]
# train_x = train_test_data.iloc[:length1]
# test_x = train_test_data.iloc[length1:length1 + length2]
#
# train_x = pd.DataFrame(train_x)
# test_x = pd.DataFrame(test_x)
#
# train_y = train['信用分']
#
#
# obname = train_x.select_dtypes(include=["object"]).columns
# for col in obname:
#     train_x[col] = train_x[col].astype(np.float)
#     test_x[col] = test_x[col].astype(np.float)
# # 分类变量转化int
# for col in cat_feature:
#     train_x[col] = train_x[col].astype(np.int)
#     test_x[col] = test_x[col].astype(np.int)
#
#
# print (train_x.head(5),train_x.shape)
# print (train_y.head(5),train_y.shape)
#
# #--------------------------------------------算法模型搭建------------------------------------------------------#
# x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
#
# clf = lgb.LGBMRegressor()
# clf.fit(train_x, train_y,  eval_metric='neg_mean_absolute_error')
# # clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=100)
#
# y_train_pred = clf.predict(x_train)
# y_test_pred = clf.predict(x_test)
# train_report = metrics.mean_squared_error(y_train, y_train_pred)
# test_report = metrics.mean_squared_error(y_test, y_test_pred)
# print(train_report)
# print(test_report)
#
# test_y_pred = clf.predict(test_x)
# test_y_pred = test_y_pred.astype(np.int)
# y_new_test_pred = pd.DataFrame(test_y_pred ,columns=['score']).astype(np.int)
#
# Prd_inst_id = Prd_inst_id.rename('id',inplace = True)
# result = pd.concat([Prd_inst_id, y_new_test_pred.score], axis=1)
# print(result)
#
# result.to_csv('F:/Telecom_Working/Model/Credit/Credit2.csv',index=False ,encoding='utf-8')


