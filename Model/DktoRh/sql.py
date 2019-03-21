'''2019-02-21 dktorh sql'''
import pandas as pd
import numpy as np

sql = pd.read_table('F:/Telecom_Working/Model/DktoRh/sql.txt')
sql.replace(201901)