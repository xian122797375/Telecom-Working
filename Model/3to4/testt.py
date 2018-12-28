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
        elif (Top1_Bit >= 0.8) & (Top1_Bit < 0.95):
            new_train.loc[new_train[i] != 0, i] = 1  # 大于0.8转化为哑变量
            categorical_feature.append(i)  # 假定分类变量
        else:
            categorical_feature.append(i)  # 假定分类变量
print('原始维度{}个,剔除后还剩下{}个'.format(input_data.shape[1], new_train.shape[1]))
print('自动判断分类维度共计：{}个'.format(len(categorical_feature)))
    return categorical_feature, new_train