#!/usr/bin/python
# -*- coding: utf-8 -*-
'''2019-2-12资采  单c转融合模型'''
'''更新特征选择'''
import lightgbm as lgb
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.externals import joblib
from feature_selector import FeatureSelector
import warnings
# warnings.FeatureSelector('ignore')

chunk_size = 2000
data_cnt = 10000


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
train_path = 'F:/11dctorh_train.txt'
train = chunk_read_data(train_path, chunk_size, data_cnt)
train = train.iloc[:,2:]

#--------------------------------------------读取测试数据-----------------------
test_path = 'F:/12dctorh_test.txt'
test = chunk_read_data(test_path, chunk_size, data_cnt)
test = test.iloc[:,2:]
# test = test.drop(['Std_Prom_Name','Ofr_Name'], axis=1)
#--------------------------------------------读取最新数据-----------------------
new_test_path = 'F:/01dctorh_test.txt'
new_test = chunk_read_data(new_test_path, chunk_size, data_cnt)
Prd_Inst_Id = new_test.iloc[:,0]
# new_test = new_test.drop(['Std_Prom_Name','Ofr_Name'], axis=1)
new_test = new_test.iloc[:,2:]

#--------------------------数据合并-----------------------
train_labels = train.LABEL_3
test_labels = test.LABEL_2

train_features = train.drop(['LABEL_1','LABEL_2','LABEL_3','Prd_Inst_Id.1','Accs_Nbr'], axis=1)
test_features = test.drop(['LABEL_1','LABEL_2','LABEL_3','Prd_Inst_Id.1','Accs_Nbr'], axis=1)

labels = pd.concat([train_labels, test_labels], axis=0)
features = pd.concat([train_features, test_features], axis=0)

fs = FeatureSelector(data=features, labels=labels)

#--------------------------特征选择-----------------------
fs.identify_all(selection_params = {'missing_threshold': 0.6,
                                    'correlation_threshold': 0.98,
                                    'task': 'classification',
                                    'eval_metric': 'auc',
                                    'cumulative_importance': 0.99})

train_removed = fs.remove(methods = 'all', keep_one_hot=False)