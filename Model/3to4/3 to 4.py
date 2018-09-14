import lightgbm as lgb
import pandas as pd
# import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from sklearn.externals import joblib

chunk_size = 20000
data_cnt = 300000
train_path = 'F:/3UP4_s1_432.csv'


#----------------------------------------------功能区------------------------------------------------------#
def chunk_read_data(file_path, chunk_size, data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path, sep=',', header=0, iterator=True, chunksize=chunk_size, low_memory=False ,encoding='gb2312')
    train = data.get_chunk(data_cnt)
    print('文件读取完毕,总计{}条'.format(train.shape[0]))
    return train


def Fix_Missing(Enter_Date, Model=False):
    '''MODEL等于Falst表示自动模式，自动缺失补0；等于True表示智能模式补0.1或其他(功能待完成)'''
    if Model == False:
        Enter_Date = Enter_Date.fillna(0)
        Enter_Date.replace('?', 0, inplace=True)
    return Enter_Date


def Data_Var_Convert(input_data, label):
    '''删除唯一值变量，自动筛选分类变量,80%单一变量值数据进行哑变量处理'''
    categorical_feature = list()
    new_train = input_data.drop([label], axis=1)
    for i in new_train.columns:
        Category_Count = pd.DataFrame(train.groupby(i).size().sort_values(ascending=False), columns=['cnt'])  # 列分类统计
        if len(Category_Count) == 1:
            new_train = new_train.drop([i], axis=1)  # 删除唯一值
        elif (len(Category_Count) < 30) & (len(Category_Count) > 1):
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
    feature_new = list()
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
train = chunk_read_data(train_path, chunk_size, data_cnt)

obname = list(train.select_dtypes(include=["object"]).columns)
len(obname)

for i in train.columns:
    Category_Count = pd.DataFrame(train.groupby(i).size().sort_values(ascending=False), columns=['cnt'])  # 列分类统计
    if len(Category_Count) == 1:
        new_train = train.drop([i], axis=1)  # 删除唯一值  ,删掉一个

categorical_feature = list()
for i in new_train.columns:
    Category_Count = pd.DataFrame(train.groupby(i).size().sort_values(ascending=False), columns=['cnt'])  # 列分类统计
    if len(Category_Count) == 1:
        new_train = new_train.drop([i], axis=1)  # 删除唯一值
    elif (len(Category_Count) < 30) & (len(Category_Count) > 1):
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
print('原始维度{}个,剔除后还剩下{}个'.format(train.shape[1], new_train.shape[1]))
print('自动判断分类维度共计：{}个'.format(len(categorical_feature)))

a = pd.DataFrame(new_train.columns)
a.to_csv('F:/danc_columns.csv',index=False)
