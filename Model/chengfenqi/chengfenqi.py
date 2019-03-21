'''2019-02-26 橙分期'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
import lightgbm as lgb
import pandas as pd
# import xgboost as xgb
# from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import datetime
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.grid_search import GridSearchCV
# from feature_selector import FeatureSelector

chunk_size = 20000
data_cnt = 700000

def chunk_read_data(file_path, chunk_size, data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path, sep='	', header=0, iterator=True, chunksize=chunk_size, low_memory=False ,encoding='gb18030')
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
    count = train[train['LAB'] == 1].shape[0]
    data = train[train['LAB'] != 1].iloc[:count * bit,:]
    new_data_train = pd.concat([train[train['LAB'] == 1], data], axis=0)
    return new_data_train
#--------------------------------------------主程序区------------------------------------------------------#
train_path = '12train.txt'
train = chunk_read_data(train_path, chunk_size, data_cnt)
prd_inst_id = train.Prd_Inst_Id
train = train.iloc[:,2:]

train.loc[(train.Cert_flag2 == '是') & (train.Serial_flag2 == '是'),'LAB'] = 1
test = train
# train_new = train[train['LAB'] == 1]
train = RematchingDate(train,4)

train_test_data = pd.concat([train, test], axis=0)
train_test_data = train_test_data.fillna(value=0)

train_test_data = Fix_Missing(train_test_data)
# train_test_data = train_test_data.drop(['LAB'], axis=1)


# label = ''
cat_feature = []
cat_feature, train_test_data = Data_Var_Convert(train_test_data)
print(train_test_data.head(5))


train_test_data, cat_feature = Recoding_Cat_Data(train_test_data, cat_feature)



Ex_col = []
obname = train_test_data.select_dtypes(include=["object"]).columns
for col in obname:
    try:
        train_test_data[col] = train_test_data[col].astype(np.float)
    except:
        Ex_col.append(col)

print(Ex_col)

train_test_data = train_test_data.drop(Ex_col, axis=1)




length1 = train.shape[0]
length2 = test.shape[0]
train_x = train_test_data.iloc[:length1]
test_x = train_test_data.iloc[length1:length1 + length2]


train_x = pd.DataFrame(train_x)
test_x = pd.DataFrame(test_x)


train_y = train['LAB']
test_y = test['LAB']



# 连续变量转化float
obname = train_x.select_dtypes(include=["object"]).columns
for col in obname:
    train_x[col] = train_x[col].astype(np.float)
    test_x[col] = test_x[col].astype(np.float)

# 分类变量转化int
for col in cat_feature:
    train_x[col] = train_x[col].astype(np.int)
    test_x[col] = test_x[col].astype(np.int)


# train_y = train_y.replace('T','1')
train_y = train_y.fillna(value=0)
train_x = train_x.fillna(value=0)
test_y = test_y.fillna(value=0)
test_y = test_y.fillna(value=0)
# print (train_x.head(5),train_x.shape)
# print (train_y.head(5),train_y.shape)
# train_x = train_x.drop(['is_quan_nbr_New'], axis=1)
# test_x = test_x.drop(['is_quan_nbr_New'], axis=1)

train_x = train_x.drop(['Cert_flag2_New','Serial_flag2_New','接入号码','接入号码.1','产品实例标识','产品实例标识.1'], axis=1)
test_x = test_x.drop(['Cert_flag2_New','Serial_flag2_New','接入号码','接入号码.1','产品实例标识','产品实例标识.1'], axis=1)

train_x = train_x.drop(['LAB_New'], axis=1)
test_x = test_x.drop(['LAB_New'], axis=1)
train_x = train_x.drop(['Cert_flag3_New','Serial_flag3_New','Serial_flag3_New'], axis=1)
test_x = test_x.drop(['Cert_flag3_New','Serial_flag3_New','Serial_flag3_New'], axis=1)
# train_x = train_x.drop(['Cert_flag1_New','Serial_flag1_New','Serial_flag1_New'], axis=1)
#--------------------------------------------算法模型搭建------------------------------------------------------#
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], eval_metric='auc', early_stopping_rounds=400)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)

test_y_pred = clf.predict(test_x)
test_y_report = metrics.classification_report(test_y, test_y_pred)
print(test_y_report)

# fpr,tpr,threshold = roc_curve(test_y, test_y_pred)
#
# roc_auc = auc(fpr,tpr)


feature_importances = pd.DataFrame({'Feature_name': x_train.columns,
                                        'Importances': clf.feature_importances_})
feature_importances = feature_importances.set_index('Feature_name')
feature_importances = feature_importances.sort_values(by='Importances', ascending=False)
print('Feature importances:', feature_importances.head(20))

y_new_test_pred = clf.predict_proba(test_x)
y_new_test_pred = pd.DataFrame(y_new_test_pred ,columns=['zero_prob','one_prob'])

result = pd.concat([prd_inst_id, y_new_test_pred.one_prob], axis=1)
# result = result[result.one_prob >= 0.5]
result.to_csv('12train_result.csv',index=False,encoding='gb18030',sep=",")


# plt.figure()
# lw = 2
# plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

