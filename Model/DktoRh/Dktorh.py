'''2018-9-12资采 不限量迁转 单k转不限量模型'''
import lightgbm as lgb
import pandas as pd
# import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from sklearn.externals import joblib

# columns = ['Prd_Inst_Id',
#  'balance',
#  'Bil_1X_Dur',
#  'Bil_3G_Dur',
#  'charge',
#  'cwqf',
#  'F1X_flux',
#  'F3G_flux',
#  'Fav_Inv_Amt',
#  'fkzs',
#  'fz_flag',
#  'hbbt',
#  'Hday_Days',
#  'Hday_Dur',
#  'Hday_Flux',
#  'Home_Dur',
#  'Home_Flux',
#  'Inner_Rgn_Amt',
#  'Int_Sms_Amt',
#  'Inter_Rgn_Amt',
#  'Inv_Amt',
#  'lower_charge',
#  'O_Call_Cnt',
#  'O_Call_Dstn',
#  'O_Call_Dur',
#  'O_Inet_Pp_Sms_Cnt',
#  'O_Inner_Rgn_Cnt',
#  'O_Inner_Rgn_Dur',
#  'O_Inter_Rgn_Cnt',
#  'O_Inter_Rgn_Dur',
#  'O_Onet_Pp_Sms_Cnt',
#  'O_Sp_Sms_Cnt',
#  'O_Tol_Cnt',
#  'O_Tol_Dstn',
#  'O_Tol_Dur',
#  'Off_Dur',
#  'Off_Flux',
#  'Office_Dur',
#  'Office_Flux',
#  'Ofr_Id',
#  'Sp_Sms_Amt',
#  'On_Dur',
#  'On_Flux',
#  'Owe_Amt',
#  'Pp_Sms_Amt',
#  'Prom_Amt',
#  'Recv_Rate',
#  'T_Call_Cnt',
#  'T_Call_Dstn',
#  'T_Call_Dur',
#  'T_Inet_Pp_Sms_Cnt',
#  'T_Onet_Pp_Sms_Cnt',
#  'T_Sp_Sms_Cnt',
#  'Tdd_Bil_Dur',
#  'Tdd_Flux',
#  'Term_Mob_Price',
#  'thsc',
#  'Total_1X_Cnt',
#  'Total_3G_Cnt',
#  'Total_Flux',
#  'Total_Tdd_Cnt',
#  'Wday_Days',
#  'Wday_Dur',
#  'Wday_Flux',
#  'ywqf',
#  'Cer_Send_Rate',
#  'Cer_Recv_Rate',
#  'pro_Inv_Amt',
#  'pro_o_dur',
#  'pro_i_dur',
#  'pro_cdma_nbt',
#  'pro_flux',
#  'yhfl',
#  'Channel_Type_Name',
#  'Channel_Type_Name_Lvl1',
#  'hhmd',
#  'Card_Type',
#  'Std_Merge_Prom_Type_Id',
#  'Term_Type_Id',
#  'Accs_Grade',
#  'black_flag',
#  'Exp_Billing_Cycle_Id',
#  'Gender_Id',
#  'Innet_Billing_Cycle_Id',
#  'Latn_Id',
#  'Std_Prd_Inst_Stat_Id',
#  'Strategy_Segment_Id',
#  'zhrh',
#  'Cde_Merge_Prom_Name_n',
#  'Age_subsection',
#  'balance_last',
#  'Bil_1X_Dur_last',
#  'Bil_3G_Dur_last',
#  'charge_last',
#  'cwqf_last',
#  'F1X_flux_last',
#  'F3G_flux_last',
#  'Fav_Inv_Amt_last',
#  'fkzs_last',
#  'fz_flag_last',
#  'hbbt_last',
#  'Hday_Days_last',
#  'Hday_Dur_last',
#  'Hday_Flux_last',
#  'Home_Dur_last',
#  'Home_Flux_last',
#  'Inner_Rgn_Amt_last',
#  'Int_Sms_Amt_last',
#  'Inter_Rgn_Amt_last',
#  'Inv_Amt_last',
#  'lower_charge_last',
#  'O_Call_Cnt_last',
#  'O_Call_Dstn_last',
#  'O_Call_Dur_last',
#  'O_Inet_Pp_Sms_Cnt_last',
#  'O_Inner_Rgn_Cnt_last',
#  'O_Inner_Rgn_Dur_last',
#  'O_Inter_Rgn_Cnt_last',
#  'O_Inter_Rgn_Dur_last',
#  'O_Onet_Pp_Sms_Cnt_last',
#  'O_Sp_Sms_Cnt_last',
#  'O_Tol_Cnt_last',
#  'O_Tol_Dstn_last',
#  'O_Tol_Dur_last',
#  'Off_Dur_last',
#  'Off_Flux_last',
#  'Office_Dur_last',
#  'Office_Flux_last',
#  'Ofr_Id_last',
#  'Sp_Sms_Amt_last',
#  'On_Dur_last',
#  'On_Flux_last',
#  'Owe_Amt_last',
#  'Pp_Sms_Amt_last',
#  'Prom_Amt_last',
#  'Recv_Rate_last',
#  'T_Call_Cnt_last',
#  'T_Call_Dstn_last',
#  'T_Call_Dur_last',
#  'T_Inet_Pp_Sms_Cnt_last',
#  'T_Onet_Pp_Sms_Cnt_last',
#  'T_Sp_Sms_Cnt_last',
#  'Tdd_Bil_Dur_last',
#  'Tdd_Flux_last',
#  'Term_Mob_Price_last',
#  'thsc_last',
#  'Total_1X_Cnt_last',
#  'Total_3G_Cnt_last',
#  'Total_Flux_last',
#  'Total_Tdd_Cnt_last',
#  'Wday_Days_last',
#  'Wday_Dur_last',
#  'Wday_Flux_last',
#  'ywqf_last',
#  'Cer_Send_Rate_last',
#  'Cer_Recv_Rate_last',
#  'pro_Inv_Amt_last',
#  'pro_o_dur_last',
#  'pro_i_dur_last',
#  'pro_cdma_nbt_last',
#  'pro_flux_last',
#  'yhfl_last',
#  'Channel_Type_Name_last',
#  'Channel_Type_Name_Lvl1_last',
#  'hhmd_last',
#  'Card_Type_last',
#  'Std_Merge_Prom_Type_Id_last',
#  'Term_Type_Id_last',
#  'Accs_Grade_last',
#  'black_flag_last',
#  'Exp_Billing_Cycle_Id_last',
#  'Gender_Id_last',
#  'Innet_Billing_Cycle_Id_last',
#  'Latn_Id_last',
#  'Std_Prd_Inst_Stat_Id_last',
#  'Strategy_Segment_Id_last',
#  'zhrh_last',
#  'Cde_Merge_Prom_Name_n_last',
#  'Age_subsection_last',
#  'balance_last2',
#  'Bil_1X_Dur_last2',
#  'Bil_3G_Dur_last2',
#  'charge_last2',
#  'cwqf_last2',
#  'F1X_flux_last2',
#  'F3G_flux_last2',
#  'Fav_Inv_Amt_last2',
#  'fkzs_last2',
#  'fz_flag_last2',
#  'hbbt_last2',
#  'Hday_Days_last2',
#  'Hday_Dur_last2',
#  'Hday_Flux_last2',
#  'Home_Dur_last2',
#  'Home_Flux_last2',
#  'Inner_Rgn_Amt_last2',
#  'Int_Sms_Amt_last2',
#  'Inter_Rgn_Amt_last2',
#  'Inv_Amt_last2',
#  'lower_charge_last2',
#  'O_Call_Cnt_last2',
#  'O_Call_Dstn_last2',
#  'O_Call_Dur_last2',
#  'O_Inet_Pp_Sms_Cnt_last2',
#  'O_Inner_Rgn_Cnt_last2',
#  'O_Inner_Rgn_Dur_last2',
#  'O_Inter_Rgn_Cnt_last2',
#  'O_Inter_Rgn_Dur_last2',
#  'O_Onet_Pp_Sms_Cnt_last2',
#  'O_Sp_Sms_Cnt_last2',
#  'O_Tol_Cnt_last2',
#  'O_Tol_Dstn_last2',
#  'O_Tol_Dur_last2',
#  'Off_Dur_last2',
#  'Off_Flux_last2',
#  'Office_Dur_last2',
#  'Office_Flux_last2',
#  'Ofr_Id_last2',
#  'Sp_Sms_Amt_last2',
#  'On_Dur_last2',
#  'On_Flux_last2',
#  'Owe_Amt_last2',
#  'Pp_Sms_Amt_last2',
#  'Prom_Amt_last2',
#  'Recv_Rate_last2',
#  'T_Call_Cnt_last2',
#  'T_Call_Dstn_last2',
#  'T_Call_Dur_last2',
#  'T_Inet_Pp_Sms_Cnt_last2',
#  'T_Onet_Pp_Sms_Cnt_last2',
#  'T_Sp_Sms_Cnt_last2',
#  'Tdd_Bil_Dur_last2',
#  'Tdd_Flux_last2',
#  'Term_Mob_Price_last2',
#  'thsc_last2',
#  'Total_1X_Cnt_last2',
#  'Total_3G_Cnt_last2',
#  'Total_Flux_last2',
#  'Total_Tdd_Cnt_last2',
#  'Wday_Days_last2',
#  'Wday_Dur_last2',
#  'Wday_Flux_last2',
#  'ywqf_last2',
#  'Cer_Send_Rate_last2',
#  'Cer_Recv_Rate_last2',
#  'pro_Inv_Amt_last2',
#  'pro_o_dur_last2',
#  'pro_i_dur_last2',
#  'pro_cdma_nbt_last2',
#  'pro_flux_last2',
#  'yhfl_last2',
#  'Channel_Type_Name_last2',
#  'Channel_Type_Name_Lvl1_last2',
#  'hhmd_last2',
#  'Card_Type_last2',
#  'Std_Merge_Prom_Type_Id_last2',
#  'Term_Type_Id_last2',
#  'Accs_Grade_last2',
#  'black_flag_last2',
#  'Exp_Billing_Cycle_Id_last2',
#  'Gender_Id_last2',
#  'Innet_Billing_Cycle_Id_last2',
#  'Latn_Id_last2',
#  'Std_Prd_Inst_Stat_Id_last2',
#  'Strategy_Segment_Id_last2',
#  'zhrh_last2',
#  'Cde_Merge_Prom_Name_n_last2',
#  'Age_subsection_last2',
#  'flag'] #



# train_data = lgb.Dataset(data, label=label, feature_name=['c1', 'c2', 'c3'], categorical_feature=['c3'])
#----------------------------------------------参数区------------------------------------------------------#
# categorical_feature = [] #自定义初始分类变量

chunk_size = 200000
data_cnt = 4000000
train_path = 'F:/06train_dk_torh.txt'

#----------------------------------------------功能区------------------------------------------------------#
def chunk_read_data(file_path, chunk_size, data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path, sep='|', header=0, iterator=True, chunksize=chunk_size, low_memory=False ,encoding='gb2312')
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
    categorical_feature = list()
    # new_train = input_data.drop([label], axis=1)
    new_train = input_data
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

def RematchingDate(train,bit):
    count = train[train['LABEL_3'] == 'T'].shape[0]
    data = train[train['LABEL_3'] == 'F'].iloc[:count * bit,:]
    new_data_train = pd.concat([train[train['LABEL_3'] == 'T'], data], axis=0)
    return new_data_train
#--------------------------------------------主程序区------------------------------------------------------#
train = chunk_read_data(train_path, chunk_size, data_cnt)
# train = RematchingDate(train,1)
# Id = train['Prd_Inst_Id']
train = train.iloc[:,2:]
# train_y = train['LABEL_2']

# train = Fix_Missing(train)
# print(train.head(5))
# Id = train['Prd_Inst_Id']
# train_x = train.iloc[:,2:-3]
# train_y = train['label_3']

#--------------------------------------------读取测试数据-----------------------
test_path = 'F:/07train_dk_torh.txt'
test = chunk_read_data(test_path, chunk_size, data_cnt)
# test = Fix_Missing(test)
# print(test.head(5))
# Id = test['Prd_Inst_Id']
test = test.iloc[:,2:]
# test_y = test['LABEL_2']
#--------------------------------------------读取最新数据-----------------------
new_test_path = 'F:/08train_dk_torh.txt'
new_test = chunk_read_data(new_test_path, chunk_size, data_cnt)
Prd_Inst_Id = new_test.iloc[:,0]
new_test = new_test.iloc[:,2:]

#--------------------------数据合并-----------------------
train = RematchingDate(train,1)

train_test_data = pd.concat([train, test, new_test], axis=0)

train_test_data = Fix_Missing(train_test_data)

# drop_var = 'Exp_Date','LABEL_1','LABEL_2','LABEL_3'
train_test_data = train_test_data.drop(['Line_Rate.1','Accs_Nbr','Ofr_Name','LABEL_1','LABEL_2','LABEL_3'], axis=1)
# train_test_data = train_test_data.drop(['Line_Rate'], axis=1)

# train_x = train_x.drop(['Exp_Date'], axis=1)
# label = ''
cat_feature, train_test_data = Data_Var_Convert(train_test_data)
print(train_test_data.head(5))

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

# #先处理时间变量
# Data_Var = 'Exp_Date'
# New_Data_Var = '{}'.format(Data_Var + '_New')
# month_var = 1
# original_date = datetime.datetime.strptime('2018/01/01', "%Y/%m/%d").replace(month=month_var,day=1)
#
# for i in range(len(train_x)):
#     # print(i)
#     tm = train_x[Data_Var][i]
#     if tm != 0:
#         train_x.loc[i, New_Data_Var] = (
#         datetime.datetime.strptime(tm, "%Y/%m/%d") - original_date).days
# train_x.fillna(0)
# train_x = train_x.drop([Data_Var], axis=1)

# train_x.to_csv('F:/rh_01_result.csv')
# 连续变量转化float
obname = list(train_x.select_dtypes(include=["object"]).columns)
for col in obname:
    train_x[col] = train_x[col].astype(np.float)
# 分类变量转化int
for col in cat_feature:
    train_x[col] = train_x[col].astype(np.int)

for col in obname:
    test_x[col] = test_x[col].astype(np.float)
# 分类变量转化int
for col in cat_feature:
    test_x[col] = test_x[col].astype(np.int)

for col in obname:
    new_test_x[col] = new_test_x[col].astype(np.float)
# 分类变量转化int
for col in cat_feature:
    new_test_x[col] = new_test_x[col].astype(np.int)

train_y = train_y.replace('T','1')
train_y = train_y.replace('F','0')

test_y = test_y.replace('T','1')
test_y = test_y.replace('F','0')

print (train_x.head(5),train_x.shape)
print (train_y.head(5),train_y.shape)


#--------------------------------------------算法模型搭建------------------------------------------------------#
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=3000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
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

model_path = 'F:/20180910_DZR.model'
joblib.dump(clf, model_path)

y_new_test_pred = clf.predict_proba(new_test_x)
y_new_test_pred = pd.DataFrame(y_new_test_pred ,columns=['zero_prob','one_prob'])

prd_inst_id = Prd_Inst_Id
result = pd.concat([prd_inst_id, y_new_test_pred.one_prob], axis=1)
# result = result[result.one_prob >= 0.5]
print(result)
# result2 = result.iloc[:,0]
result.to_csv('F:/rh_05to07dktorh_result.csv',index=False)



print(clf.feature_importances_)
plt.figure(figsize=(12,6))
lgb.plot_importance(clf, max_num_features=30)
plt.title("Featurertances")
plt.show()



#--------------------------------------------测试集数据读取---------------------------------------------#
test_path = 'F:/08test_rh.txt'
test = chunk_read_data(test_path, chunk_size, data_cnt)
test = Fix_Missing(test)
print(test.head(5))
Id = test['Prd_Inst_Id']
test_x = test.iloc[:,2:]
test_y = test['LABEL_2']

drop_var = 'LABEL_3'
test_x = test_x.drop([drop_var], axis=1)
# test_x = test_x.drop(['Exp_Date'], axis=1)
label = 'LABEL_2'
cat_feature, test_x = Data_Var_Convert(test_x, label)
print(test_x.head(5))

test_x, cat_feature = Recoding_Cat_Data(test_x, cat_feature)

#先处理时间变量
Data_Var = 'Exp_Date'
New_Data_Var = '{}'.format(Data_Var + '_New')
month_var = 1
original_date = datetime.datetime.strptime('2018/01/01', "%Y/%m/%d").replace(month=month_var,day=1)

for i in range(len(test_x)):
    # print(i)
    tm = test_x[Data_Var][i]
    if tm != 0:
        test_x.loc[i, New_Data_Var] = (
        datetime.datetime.strptime(tm, "%Y/%m/%d") - original_date).days
test_x = test_x.fillna(0)
test_x = test_x.drop([Data_Var], axis=1)


# 连续变量转化float
obname = list(test_x.select_dtypes(include=["object"]).columns)
for col in obname:
    test_x[col] = test_x[col].astype(np.float)
# 分类变量转化int
for col in cat_feature:
    test_x[col] = test_x[col].astype(np.int)

test_x = test_x.drop(['LABEL_3_New'], axis=1)

test_y = test_y.replace('T','1')
test_y = test_y.replace('F','0')
print (test_x.head(5),test_x.shape)
print (test_y.head(5),test_y.shape)
#--------------------------------------------测试集数据测试---------------------------------------------#
model_path = 'F:/20180808_RZR.model'
clf = joblib.load(model_path)

y_test_pred = clf.predict(test_x)
test_report = metrics.classification_report(test_y, y_test_pred)
print(test_report)


y_test_pred = clf.predict(test_x)
y_test_pred = pd.DataFrame(y_test_pred , columns=['flag'])

prd_inst_id = Id
result = pd.concat([prd_inst_id, y_test_pred], axis=1)
result = result[result.flag == '1']
print(result)
result2 = result.iloc[:,0]
result2.to_csv('F:/rh_03to07rhtorh_result.csv',index=False)

# precision    recall  f1-score   support
# F       0.95      0.98      0.96    257029
# T       0.22      0.11      0.14     14416



#--------------------------------------------调参------------------------------------------------------#
from sklearn.model_selection import GridSearchCV

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=50,
                              learning_rate=0.1, n_estimators=43, max_depth=6,
                              metric='rmse', bagging_fraction=0.8, feature_fraction=0.8)

params_test1 = {
    'max_depth': range(3, 8, 2),
    'num_leaves': range(50, 170, 30)
}
gsearch1 = GridSearchCV(estimator=model_lgb, param_grid=params_test1, scoring='neg_mean_squared_error', cv=5, verbose=1,
                        n_jobs=4)

gsearch1.fit(x_train, y_train)
gsearch1.best_params_, gsearch1.best_score_
#--------------------------------------------其他------------------------------------------------------#
Size_count = pd.DataFrame(train.groupby('Accs_Grade').size().sort_values(ascending = False),columns=['cnt']) #列分类统计
Size_sum = 0 #存储累计列和
Category = list() #存所剩种类
for i in range(len(Size_count)):
    print(i)
    Col_Sum = Size_count.cnt.sum(axis=0)
    if Size_sum / Col_Sum < 0.8:
        Size_sum = Size_count.iloc[i,0] + Size_sum
        Category.append(Size_count.index[i])
    else:
        break
print(Category)

for i in range(len(Category)):
    print (i)
    train.loc[train['Accs_Grade'] == Category[i] , 'New_Accs_Grade']  = i + 1
train['New_Accs_Grade'] = train['New_Accs_Grade'].fillna(0)


# for col in obname:
#     train[col] = train[col].astype(np.float)

# A = pd.DataFrame(train['Accs_Grade'].value_counts())

Size_count = pd.DataFrame(train.groupby('Accs_Grade').size().sort_values(ascending = False),columns=['cnt']) #列分类统计
Size_sum = 0 #存储累计列和
Category = list() #存所剩种类
for i in range(len(Size_count)):
    print(i)
    Col_Sum = Size_count.cnt.sum(axis=0)
    if Size_sum / Col_Sum < 0.8:
        Size_sum = Size_count.iloc[i,0] + Size_sum
        Category.append(Size_count.index[i])
    else:
        break
print(Category)

for i in range(len(Category)):
    print (i)
    train.loc[train['Accs_Grade'] == Category[i] , 'New_Accs_Grade']  = i + 1
train['New_Accs_Grade'] = train['New_Accs_Grade'].fillna(0)







train_y = data.flag
train_x = data.drop(['flag','Ofr_Id','Unnamed: 0','Ofr_Id_last','Ofr_Id_last2'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)

clf = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
    max_depth=-1, n_estimators=500, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
)
clf.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric='auc', early_stopping_rounds=100)

y_train_pred = clf.predict(train_x)
y_train_pred = pd.DataFrame(y_train_pred)
train_report = metrics.classification_report(train_y, y_train_pred)
print(train_report)


y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)
train_report = metrics.classification_report(y_train, y_train_pred)
test_report = metrics.classification_report(y_test, y_test_pred)
print(train_report)
print(test_report)


# importance = clf.feature_importance()


plt.figure(figsize=(12,6))
lgb.plot_importance(clf, max_num_features=30)
plt.title("Featurertances")
plt.show()
#   precision    recall  f1-score   support
#           0       0.99      1.00      0.99     97720
#           1       0.87      0.49      0.62      2280
# avg / total       0.99      0.99      0.98    100000


#7-27训练和验证
#              precision    recall  f1-score   support
#           0       0.99      1.00      0.99    156356
#           1       0.86      0.51      0.64      3644
# avg / total       0.99      0.99      0.99    160000
#              precision    recall  f1-score   support
#           0       0.99      1.00      0.99     39131
#           1       0.69      0.42      0.52       869
# avg / total       0.98      0.98      0.98     40000
