#!/usr/bin/python
# -*- coding: utf-8 -*-
'''孝感、恩施、麻城模型训练'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import numpy as np
import os
import datetime
import cx_Oracle
from sqlalchemy import types, create_engine
import os
from hcode import match
import sys
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'

#--------------------------模型参数区域---------------------------------
'''利用ep库进行训练 '''
# sys.path.append('/data1')
now = datetime.datetime.now() #获取今天日期
now = now - datetime.timedelta(days = 2)
last = now - datetime.timedelta(days = 6)
now = int(now.strftime('%Y%m%d'))
last = int(last.strftime('%Y%m%d'))
latn_name = ('hg','es','xg')
chunk_size = 200000
data_cnt = chunk_size*10  #
split_size = 0.2
engine = create_engine("oracle://anti_fraud:at_87654@133.0.176.69:1521/htxxsvc2")
engine2 = create_engine("oracle://wudelin:cellSMS2016@133.0.186.4:1521/orcl")
DB_path = 'Big_Model_Fxmx_Day'
#--------------------------功能区域---------------------------------
def latn_model(latn_name):
    '''地区编码'''
    if latn_name == 'xg':
        latn_id = 1006
    elif latn_name == 'es':
        latn_id = 1013
    elif latn_name == 'hg':
        latn_id = 1004
    return latn_id


def chunk_read_data(file_path,chunk_size,data_cnt):
    '''文件批量读取'''
    data = pd.read_csv(file_path,header=None,iterator=True,chunksize=chunk_size)
    train = data.get_chunk(data_cnt)
    train = train.fillna(0)
    print('文件读取完毕,总计{}条'.format(train.shape[0]))
    return train

def get_ep_fx_data(path):
    '''前5天不再本地近两日在本地筛选后的用户用家庭圈确认得到'''
    print('提取ep库中确定返乡用户:')
    train = chunk_read_data(path, 200000, 5000000)
    train_new = train[(train.iloc[:, 161] == 0) & (train.iloc[:, 162] == 0) & (train.iloc[:, 163] == 0) &(train.iloc[:,164:168].sum(axis=1) >= 3)]
    nbr = train_new.iloc[:, 0]
    print(train_new.head(5))
    print('确定返乡用户数{}'.format(train_new.shape[0]))
    return nbr

def get_ep_fx_nbr(date):
    '''ep库返省用户读取'''
    sql1 = 'SELECT to_number(self_number) as self_number FROM EP_PHONES'
    nbr1 = pd.read_sql(sql1,engine2)
    sql2 = """SELECT to_number(nbr) as nbr from BS_CODE_DAYS
              where CDATE = {}""".format(date)
    nbr2 = pd.read_sql(sql2, engine)
    result = pd.merge(left=nbr1, right=nbr2, how='inner', left_on='self_number', right_on='nbr')
    return result



def get_label_train_data(train,train_model):
    '''train_model 等于 True 开始训练数据切分
       train_model 等于 Flase 开始实际运行数据切分'''
    if train_model == True:
        print('开启ep库训练／验证模式:')
        train.loc[(train.iloc[:, 161] == 0) & (train.iloc[:, 162] == 0) & (train.iloc[:, 163] == 0) & (train.iloc[:,164:168].sum(axis=1) < 3),'label1'] = 1
        train.loc[(train.iloc[:, 164:168].sum(axis=1) >= 1), 'label2'] = 1
        train = train.fillna(0)
        train_new = train[train['label1'] == 1]
        nbr = train.iloc[:, 0]
        train_x = train_new.iloc[:, 1:161]
        train_y = train_new.iloc[:, -1]
        old_sample = train.shape[0]
        sample = train_new.shape[0]
        n_pos_sample = train_new[train_new.iloc[:, -1] == 1].shape[0]
        n_neg_sample = train_new[train_new.iloc[:, -1] == 0].shape[0]
        print('原样本个数:{},剔除已返乡后剩样本个数:{},4天内返乡用户（正样本）个数:{},4天内未返乡用户（负样本）个数:{}'.format(old_sample, sample, n_pos_sample,n_neg_sample))
        return nbr, train_x, train_y
    if train_model == False:
        print('开启日加载模式:')
        nbr = train.iloc[:, 0]
        train_x = train.iloc[:,1:161]
        return nbr, train_x

def split_data(x,y,split_size):
    '''数据切分'''
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_size)
    return x_train, x_test, y_train, y_test


def load_mode_predict(model_path,x):
    '''模型训练'''
    clf = joblib.load(model_path)
    rusult = clf.predict(x)
    return rusult

#-------------------------主程序----------------------------
def model_train(ep_path):
    '''读取昨日ep库数据观测是否需要修改模型'''
    train = chunk_read_data(ep_path,200000,2000000)
    nbr, train_x, train_y = get_label_train_data(train,train_model = True) #开启训练模式
    x_train, x_test, y_train, y_test = split_data(train_x,train_y,split_size)

    y_pred_train = load_mode_predict(model_path,x_train)  # 训练集测试
    y_pred_test = load_mode_predict(model_path,x_test)  # 验证集测试

    train_precision = metrics.precision_score(y_train, y_pred_train, pos_label=1).astype(float)
    test_precision = metrics.precision_score(y_test, y_pred_test, pos_label=1).astype(float)
    '''命中判断是否需要修正模型'''
    if ((train_precision <= test_precision + 0.05) and (train_precision > test_precision - 0.05)):
        print('验证通过,训练集命中率{},测试集命中率{}'.format(train_precision, test_precision))
        correction_flag = 0  # 不需要修正
    else:
        print('模型需要调整')
        clf = RandomForestClassifier(max_features='sqrt', n_estimators=70, min_samples_leaf=10, max_depth=11,
                                     min_samples_split=50).fit(x_train, y_train)
        y_pred_train = clf.predict(x_train)  # 训练集测试
        y_pred_test = clf.predict(x_test)  # 验证集测试

        # train_111 = metrics.classification_report(y_train,y_pred_train)
        # test_111 = metrics.classification_report(y_test,y_pred_test)

        train_precision = metrics.precision_score(y_train, y_pred_train, pos_label=1).astype(float)
        test_precision = metrics.precision_score(y_test, y_pred_test, pos_label=1).astype(float)

        if ((train_precision <= test_precision + 0.2) and (train_precision > test_precision - 0.2)):
            print('验证通过,训练集命中率{},测试集命中率{}'.format(train_precision, test_precision))
            correction_flag = 0  # 修正完毕
            joblib.dump(clf, model_path)

        else:
            print('模型主体参数需要调整,请手动修正')
            correction_flag = 1  # 需要修正

    return correction_flag

def read_predict_test(path,model_path):
    '''输出用户'''
    test = chunk_read_data(path, 200000, 5000000)
    nbr, train_x = get_label_train_data(test, train_model=False)

    clf = joblib.load(model_path)
    result = clf.predict(train_x)
    result = pd.Series(result)
    out_data = pd.concat([nbr, result], axis=1)
    print('当日预测出用户数量{}户'.format(out_data[out_data.iloc[:, -1] == 1].shape[0]))

    out_data.loc[out_data.iloc[:,-1] == 1 ,'label'] = 1
    user_nbr = out_data[out_data['label'] == 1]
    user_nbr = user_nbr.iloc[:,0]
    user_nbr = pd.DataFrame(user_nbr)
    user_nbr.columns = ['nbr']
    return user_nbr

def read_family_circle(path):
    family_circle_nbr = chunk_read_data(path,100000, 5000000)
    family_circle_nbr.columns = ['family_nbr', 'other_nbr','cnt']
    return family_circle_nbr

def correlate_family_Bigdata(user,family_circle_nbr):
    result = pd.merge(left=family_circle_nbr, right=user, how='inner', left_on='other_nbr', right_on='nbr')
    return result

def out_data_fomat(out_data):
    '''标准化数据,方便入库'''
    out_data['boss_bs_code1'] = 0
    out_data['boss_bs_code2'] = 0
    out_data['remake2'] = 0
    out_data['remake3'] = 0
    out_data['remake4'] = 0
    out_data['flag2'] = 0
    out_data['flag3'] = 0
    out_data['bs_code2'] = 0
    out_data['bs_code1'] = 0
    data_day = out_data.ix[:,['day_id','latn_id','other_nbr','bs_code1','boss_bs_code1','remake1','remake2','family_nbr','bs_code2','boss_bs_code2','remake3','remake4','flag1','flag2','flag3']]
    return data_day

def output_into_DB(engine,data_day,DB_path):
    dtyp = {c: types.VARCHAR(data_day[c].str.len().max())
            for c in data_day.columns[data_day.dtypes == 'object'].tolist()}
    print(dtyp)
    data_day.to_sql(DB_path, engine, index=False, if_exists='append',dtype=dtyp)
    # data_day.to_csv(os.path.join('/data1/fxmx_{}.csv'.format(now)), index=False)
    print('预测结果写入数据库完毕')


def Big_model_run(name,latn_id,date):
    '''路径设置'''
    family_circle_path = '/data1/juan/{}_201709_201712.csv'.format(name)  # 家庭圈
    ep_path = '/data1/output/fxkl_ep_{}_{}_{}/part-00000'.format(name, last, now)
    yw_path = '/data1/output/fxkl_wjjs_{}_{}_{}/part-00000'.format(name, last, now)
    print('处理{}数据,请稍等......'.format(now))

    '''ep库数据,提取明确返乡用户'''
    ep_fx_nbr_day = get_ep_fx_data(ep_path)
    ep_fx_nbr_day = pd.DataFrame(ep_fx_nbr_day)
    ep_fx_nbr_day.columns = ['nbr']

    '''家庭圈数据读取,并关联'''
    family_circle_nbr = read_family_circle(family_circle_path)
    result = correlate_family_Bigdata(ep_fx_nbr_day, family_circle_nbr)
    print('话单分析确定返乡用户{},关联家庭圈后剩下用户{}'.format(len(ep_fx_nbr_day), result.shape[0]))

    '''结果标记'''
    result['day_id'] = date
    result['flag1'] = 0
    result1 = result

    '''利用ep实际位置找到返乡用户进行补充'''
    ep_fx_nbr = get_ep_fx_nbr(now)
    result = correlate_family_Bigdata(ep_fx_nbr, family_circle_nbr)
    print('EP定位确定返乡用户{},关联家庭圈后剩下用户{}'.format(len(ep_fx_nbr), result.shape[0]))

    '''结果标记'''
    result['day_id'] = date
    result['flag1'] = 1
    result2 = result


    '''模型效果检测'''
    correction_flag = model_train(ep_path)
    if correction_flag == 0:

        '''检测通过,开启本网外省返乡预测'''
        user = read_predict_test(ep_path,model_path)
        result = correlate_family_Bigdata(user, family_circle_nbr)
        result['day_id'] = date
        result['flag1'] = 2
        result3 = result
        print(result.shape[0])

        other_nbr = result.iloc[:,0]
        other_nbr = other_nbr.drop_duplicates()
        print(other_nbr.shape[0])
        print('预测本网外省返乡用户数量{},关联家庭圈后号码个数{}'.format(user.shape[0], other_nbr.shape[0]))

        '''检测通过,开启异网返乡用户预测'''
        user = read_predict_test(yw_path, model_path)
        result = correlate_family_Bigdata(user, family_circle_nbr)
        print(result.shape[0])

        result['day_id'] = date
        result['flag1'] = 3
        result4 = result
        other_nbr = result.iloc[:, 0]
        other_nbr = other_nbr.drop_duplicates()
        print(result.shape[0])
        print('预测异网返乡用户数量{},关联家庭圈后号码个数{}'.format(user.shape[0], other_nbr.shape[0]))

        '''合并结果,写入数据库'''
        out_data = pd.concat([result1, result2, result3, result4], axis=0)
        out_data['latn_id'] = latn_id  # ID
        out_data['remake1'] = out_data['other_nbr'].map(f)
        # print(out_data)
        data_day = out_data_fomat(out_data)
        # print(out_data['remake1'].shape[0])
        output_into_DB(engine, data_day, DB_path)
        print('数据已写入数据库')
    else:
        print('模型失效，请调优后再尝试')

def f(nbr):
    d = match(str(nbr))
    if d:
        return d["info"]
    else:
        return

def clear_now_data(connet,date):
    '''清除今日数据,防止数据重复'''
    sql = """delete from "Big_Model_Fxmx_Day"
                WHERE DAY_ID = {}""".format(date)
    connet.execute(sql)
    print('今日数据预测结果数据清除完毕,等待重新写入')
    sql = """delete from fxmx_result1_day
                    WHERE DAY_ID = {}""".format(date)
    connet.execute(sql)
    print('今日汇总结果数据清除完毕，等待重新写入')

def output_data_to_TD(data):
    sql = """INSERT into fxmx_result1_day
            (
                day_id,
                latn_id,
                other_nbr,
                AREA_CODE,
                family_nbr,
                bs_code,
                bs_name1,
                bs_name2,
                flag1,
				flag2,
				flag3)
			select DAY_ID,LATN_ID,OTHER_NBR,REMAKE1,FAMILY_NBR,c.bs_code,BS_NAME2 as bs_name1,shop_addr as BS_NAME2,FLAG1,TO_NUMBER(SUBSTR(cell_distance,0,4)) as flag2,bs_name1 as flag3 from "Big_Model_Fxmx_Day" b
            inner JOIN (SELECT * from BS_CODE_MON)  c
            on b.FAMILY_NBR = c.nbr
			left join (SELECT * from TM_NEAR_CELL_SHOP) s
			on c.bs_code=s.bs_code
            where day_id = {}
            and ((LATN_ID =1013
            and BS_NAME1 = '恩施市') or (LATN_ID =1006
            and BS_NAME1 = '孝感市') or (LATN_ID =1004
            and BS_NAME1 = '黄冈市'))
			GROUP BY DAY_ID,LATN_ID,OTHER_NBR,REMAKE1,FAMILY_NBR,c.bs_code,BS_NAME2,shop_addr,FLAG1,cell_distance,bs_name1
			ORDER BY OTHER_NBR,FAMILY_NBR""".format(data)
    engine.execute(sql)
    print('今日汇总结果数据写入完毕,待传送TD')

if __name__ == '__main__':
    now_date = int((datetime.datetime.now()-datetime.timedelta(days = 1)).strftime('%Y%m%d'))
    clear_now_data(engine,now_date)
    for i in latn_name:
        model_path = '/data1/RF_new_nobalance_{}.model'.format(i)  # 模型path
        print('开始运行{}数据'.format(i))
        latn_id = latn_model(i)
        print(latn_id,i)
        Big_model_run(i,latn_id,now_date) #结果数据
    output_data_to_TD(now_date)
