#!/usr/bin/python
# coding: utf-8
import random as rd
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import datetime
import os
import time

import matplotlib.pyplot as plt

time_start=time.time()

def binning(col, cut_points, labels=None):
    # Define min and max values:
    minval = col.min()
    maxval = col.max()
    # create list by adding min and max to cut_points
    break_points = [minval] + cut_points + [maxval]
    # if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points) + 1)
    # Binning using cut function of pandas
    colBin = pd.cut(col, bins=break_points, labels=labels, include_lowest=True)
    return colBin

def stratified_sample(data,label):
    train = data
    train["flag"] = train.loc[:, [label]].apply(lambda x: ''.join(str(x.values)), axis=1)
    temp = pd.DataFrame()
    temp['count'] = train.groupby('flag')['flag'].agg("count")
    temp['percent'] = temp['count'] / len(train)
    sample = pd.DataFrame()
    for i in range(len(temp)):
        t = train[train.flag == temp.index[i]]
        t = t.sample(frac=0.1, replace=True)
        sample = pd.concat([sample, t], axis=0)
    df_choice = sample
    df_not_choice = train.iloc[~(train.index.isin(sample.index))]
    print('总用户数{},抽样用户数{}'.format(train.shape[0],df_choice.shape[0]))
    return df_choice,df_not_choice

#--------------------------TD数据库抽取sql---------------------------------
# sel
# distinct prd_inst_id
# ,latn_id
# ,case when kdll is null then 0 else kdll/1024 end as kdll
# ,case when Innet_Billing_Cycle_Id between 201704 and 201803 then 0 else 1 end as  Innet_Billing_Cycle_state
# ,case when Pay_Flag = 'T' then 0 when ofr_name like '%计时%' then 1 else 2 end Pay_state
# ,Prom_amt
# ,Recv_Rate/1024 as Recv_Rate
# ,case when ywqf is not null or cwqf is not null then 1 else 0 end yf_flag
#  from  td_work.tmp_yuanlei_zc201803_xx
# where substr(trim(std_prd_id),1,4)=1015
# and zhrh = 0
# and card_type in(1,3)
#
# and mkt_emp_name is  null
# and charge <= 5000
# and hhmd = '否'
# and dzq is  null
# and (ofr_name not like'%员工%' and ofr_name not like'%演示%' and ofr_name not like'%体验卡%')
# and Billing_Type_Id  not in (100000005,100000001,100000006,100000007)
#
# and (cfq_end_date is null or cfq_end_date <	 CAST('20190501' AS DATE FORMAT'yyyymmdd'))
# and (End_Date is null or End_Date < 20190501)
# and (Exp_Billing_Cycle_Id is null or Exp_Billing_Cycle_Id < 201905)
#--------------------------数据读取、处理---------------------------------
train_input = 'F:\chouyang.txt'
train = pd.read_csv(train_input, sep=',', header=0,iterator=True, chunksize=200000,encoding='gb2312')
train = train.get_chunk(16000000)
train = train.fillna(0)
print('数据读取完毕，共{}条'.format(train.shape[0]))

cfq_data = train[train.is_cfq == 1]
# no_cfq_data = train[train.is_cfq == 0]


# cfq_data(['Prd_Inst_Id', 'Latn_Id', 'Innet_Billing_Cycle_Id', 'Ofr_Name_Flag',
#        'zfk', 'Prom_Amt', 'is_cfq'])


# cfq_data['Innet_Billing_Cycle_Id'].quantile([.25, .5, 0.75])
# 0.25    201210.0
# 0.50    201509.0
# 0.75    201706.0
# cfq_data['Prom_Amt'].quantile([.25, .5, 0.75])



cut_points = [201210,201509,201706,201801,201802,201803]  # 用train['Innet_Billing_Cycle_Id'].quantile([.25, .5, 0.75])
labels = ["0","1","2","3","4","5","6"]
train["Innet_Billing_Cycle_Id_Type"] = binning(train["Innet_Billing_Cycle_Id"], cut_points, labels)
cfq_data["Innet_Billing_Cycle_Id_Type"] = binning(cfq_data["Innet_Billing_Cycle_Id"], cut_points, labels)
print(pd.value_counts(train["Innet_Billing_Cycle_Id_Type"], sort=False))
print(pd.value_counts(cfq_data["Innet_Billing_Cycle_Id_Type"], sort=False))

cut_points = [149,169]  # 用train['Prom_Amt'].quantile([.25, .5, 0.75])
labels = ["0", "1" ,"2"]
train["Prom_Amt_Type"] = binning(train["Prom_Amt"], cut_points, labels)
cfq_data["Prom_Amt_Type"] = binning(cfq_data["Prom_Amt"], cut_points, labels)
print(pd.value_counts(train["Prom_Amt_Type"], sort=False))
print(pd.value_counts(cfq_data["Prom_Amt_Type"], sort=False))







# cut_points = [1002]  # 用train['kdll'].quantile([.25, .5, 0.75])
# labels = ["0", "1"]
# train["Latn_Id_Type"] = binning(train["Latn_Id"], cut_points, labels)
# print(pd.value_counts(train["Latn_Id_Type"], sort=False))
#
# cut_points = [13.4, 26, 58.5] #用train['charge'].quantile([.25, .5, 0.75])
# labels = ["0", "1", "2", "3"]
# train["Charg_Flag"] = binning(train["charge"], cut_points, labels)
# print(pd.value_counts(train["Charg_Flag"], sort=False))
#
# cut_points = [5, 19, 39]
# labels = ["0", "1", "2", "3"]
# train["Prom_Lower_Charge_Flag"] = binning(train["Prom_Lower_Charge"], cut_points, labels)
# print(pd.value_counts(train["Prom_Lower_Charge_Flag"], sort=False))

train["flag"] = train.loc[:, ['Latn_Id', 'Innet_Billing_Cycle_Id_Type' ,'Ofr_Name_Flag' ,'zfk' , 'Prom_Amt_Type']].apply(lambda x: ''.join(str(x.values)), axis=1)
cfq_data["flag"] = cfq_data.loc[:, ['Latn_Id', 'Innet_Billing_Cycle_Id_Type' ,'Ofr_Name_Flag' ,'zfk' , 'Prom_Amt_Type']].apply(lambda x: ''.join(str(x.values)), axis=1)
# train["flag"] = train.loc[:, ['Fav_Inv_Type' ,'Mob_Price_Type']].apply(lambda x: ''.join(str(x.values)), axis=1)


temp = pd.DataFrame()
temp['count'] = cfq_data.groupby('flag')['flag'].agg("count")
# temp['percent'] = temp['count'] / len(train)
# sample = pd.DataFrame()


# a = train1[(train1["flag"] == temp.index[1]) & (train1.is_cfq == 0)]
# count = temp['count'][1] * 1
# b = a.iloc[:count, :]
# sample = pd.concat([b, sample], axis=0)
# print('开始抽样，完成第{}次'.format(1))
# print('找到对照组共计{}条'.format(b.shape[0]))
#
# a = train1[(train1["flag"] == temp.index[0]) & (train1.is_cfq == 0)]
# count = temp['count'][0] * 1
# b = a.iloc[:count, :]
# sample = pd.concat([b, sample], axis=0)
# print('开始抽样，完成第{}次'.format(0))
# print('找到对照组共计{}条'.format(b.shape[0]))

for j in range(1, 5 ,1):
    print('开始第{}次抽样'.format(j))
    train1 = shuffle(train)
    sample = pd.DataFrame()

    for i in range(len(temp)):
        a = train1[(train1["flag"] == temp.index[i]) & (train1.is_cfq == 0)]
        count = temp['count'][i] * j
        b = a.iloc[:count,:]
        sample = pd.concat([b, sample], axis=0)
        print('开始抽样，完成第{}次'.format(i))
        print('找到对照组共计{}条'.format(b.shape[0]))
    sample.to_csv('D:/sample{}.csv'.format(j))
time_end = time.time()
print('抽样完毕')
print('总时间',time_end-time_start)
#--------------------------抽样---------------------------------
# for j in range(1, 11, 1):
#     train1 = shuffle(train)
#     prd_inst_id = train1['prd_inst_id']
#     for i in range(len(temp)):
#         t = train1[train1.flag == temp.index[i]]
#         t = t.sample(frac=0.2, replace=True)
#         sample = pd.concat([prd_inst_id, t], axis=0)
#     df_choice = sample
#     df_not_choice = train1.iloc[~(train1.index.isin(sample.index))]
#     df_choice.to_csv('dc_choice_sample{}.csv'.format(j))
#
#     train1['sample{}'.format(j)] = train1.index.isin(sample.index)
#     time_end = time.time()
#     print(train1.head(2))
#     print('完成第{}次抽样'.format(j))
#     print('totally cost',time_end-time_start)
# time_end = time.time()
# print('抽样完毕')
# print('总时间',time_end-time_start)


