import pandas as pd
import tensorflow as tf
import numpy as np

chunk_size = 200000
data_cnt = 500000

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
train_path = 'F:/07train_dk_torh.txt'
train = chunk_read_data(train_path, chunk_size, data_cnt)
train = train.iloc[:,2:]


#--------------------------------------------读取测试数据-----------------------
test_path = 'F:/08test_dk_torh.txt'
test = chunk_read_data(test_path, chunk_size, data_cnt)
test = test.iloc[:,2:]

#--------------------------------------------读取最新数据-----------------------
new_test_path = 'F:/09test_dk_torh.txt'
new_test = chunk_read_data(new_test_path, chunk_size, data_cnt)
Prd_Inst_Id = new_test.iloc[:,0]
new_test = new_test.iloc[:,2:]

#--------------------------数据合并-----------------------
train = RematchingDate(train,2)

train_test_data = pd.concat([train, test, new_test], axis=0)

train_test_data = Fix_Missing(train_test_data)



# drop_var = 'Exp_Date','LABEL_1','LABEL_2','LABEL_3'
train_test_data = train_test_data.drop(['LABEL_1','LABEL_2','LABEL_3'], axis=1)
# train_test_data['Line_Rate_New'] = train_test_data['Line_Rate'].str.split('M').str[0]
# train_test_data
# train_test_data = train_test_data.drop(['Line_Rate'], axis=1)

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

#--------------------------------------------算法模型搭建------------------------------------------------------#
def create_columns(continuous_columns):
    deep_columns = []
    for column in continuous_columns:
        column = tf.layers.real_valued_column(column)
        deep_columns.append(column)
    return deep_columns

a = tf.contrib

deep_columns = create_columns(train_x.columns)
