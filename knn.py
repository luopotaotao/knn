# 1.读取数据 得到 每行都是 x1,x2,x3,...,xn,y的dataFrame df
# 2.计算给定点target = (x1,x2,...,xn)与步骤1中df的所有点的距离
# 3.对步骤2所得的所有距离排序 asc,取前n个对应的y
# 4.步骤3中y出现频率最高的为target的y
from collections import namedtuple

import numpy as np
import pandas as pd

MinMaxDict = namedtuple('MinMax', 'min max range')


def load_data()-> pd.DataFrame:
    """
    加载数据
    :return: pd.DataFrame 最后一列为label,其余为输入
    """
    return pd.read_excel('knn.xlsx')


def norm(df):
    min_max_list = []
    for column in df.columns[:-1]:
        col = df[column]
        min_val, max_val = col.min(), col.max()
        length = max_val - min_val
        df[column] = (col - min_val) / length
        min_max_list.append(MinMaxDict(min_val, max_val, length))
    return df, min_max_list


def norm_target(target, min_max_list):
    ret = []
    for i, item in enumerate(min_max_list):
        ret.append((target[i] - item.min) / item.range)
    return tuple(ret)


def classify(n, df, target):
    """
    构建一个DataFrame,包含[labels,distance]两列,
    label就是训练集的label,distance为目标数据targe对训练集df每条数据的欧氏距离
    然后再根据distance排序,取前n个结果
    再对前n个结果统计,个数最多的label就是分类结果
    :param n:
    :param df:
    :param target:
    :return:
    """
    label_column = df.columns[-1]
    result = pd.DataFrame()
    result['labels'] = df[label_column].copy()
    result['distance'] = np.sqrt(np.sum(df.loc[:, df.columns[:-1]].sub(target, axis='columns') ** 2, axis=1))
    result = result.sort_values(by='distance', ascending=True)
    return result[:n]['labels'].value_counts(ascending=False).index[0]


if __name__ == '__main__':
    n = 5
    # 1.读取数据得到DataFrame
    df = load_data()
    # 2.将数据正规化,同时得到每列最大最小及跨度信息,用于目标数据正规化
    x_columns = df.columns[:-1]
    df, min_max_list = norm(df)
    # 3.将目标正规化
    item = (0, 10)
    item = norm_target(item, min_max_list)
    # 4.对目标值和已有数据计算欧氏距离,取前n个最小值得到分类结果
    result = classify(n, df, item)
    print(result)
