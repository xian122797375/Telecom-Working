'''2018-8-6资采 不限量迁转 融合转融合模型'''
import lightgbm as lgb
import pandas as pd
# import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
from sklearn.externals import joblib

combine_data = pd.DataFrame(data=dataframe3.iloc[-1, :].values, index=[int(x) for x in dataframe3.columns])
target_columns1 = [101, 102, 103]
target_data2 = combine_data.loc[target_columns1, :]
mingdu_columns = target_data2.sort_values(0, ascending=False).index
a = list()
a.append(mingdu_columns)
# mingdu_columns = [mingdu_columns]
mingdulist = [101, 102, 103]
rest_mingdu = list((set(mingdulist).union(set(mingdu_columns))) ^ (set(mingdulist) ^ set(mingdu_columns)))
