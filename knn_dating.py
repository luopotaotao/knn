import pandas as pd

import knn


def load_dating_data(file, ratio):
    df = pd.read_csv(file, sep='\t')
    df.columns = ['fly_miles', 'game_time', 'ice_cream', 'labels']
    df.labels = df.labels.map(lambda x: ['smallDoses', 'didntLike', 'largeDoses'].index(x) + 1)
    size = df.shape[0]
    num = int(size * ratio)
    return df[:num], df[num:]


# class DatingKnn(KNN):
#
#     def __init__(self, n, ratio):
#         super().__init__(n)
#         num = int(self.df.shape[0] * ratio)
#         self.testing_set = self.df[num:]
#         self.df = self.df[:num]
#
#     def load_data(self):
#         return load_dating_data('datingTestSet.txt')
#
#     def test(self):
#         results = pd.Series([self.classify(item[1:-1], False) for item in self.testing_set.itertuples()])
#         labels = self.testing_set.labels
#         labels.index = results.index
#         bingo = (labels - results).value_counts()[0]
#         print(bingo / results.shape[0])
#         # print(labels-results)
#         # print(self.testing_set.labels)


if __name__ == '__main__':
    # df = load_dating_data('datingTestSet.txt')
    # print(df.labels.unique())
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(df.iloc[:, 0], df.iloc[:, 1], s=15 * df.labels, c=15 * df.labels)
    # plt.show()

    n = 5
    ratio = 0.9  # 数据的百分之90为训练数据,10%为测试数据
    # 1.读取数据得到DataFrame
    training_set, testing_set = load_dating_data('datingTestSet.txt', 0.9)
    # 2.将数据正规化,同时得到每列最大最小及跨度信息,用于目标数据正规化
    df, min_max_list = knn.norm(training_set)

    results_list = []
    for item in testing_set.itertuples(index=False):
        # 3.将目标正规化
        item = item[:-1]
        item = knn.norm_target(item, min_max_list)
        # 4.对目标值和已有数据计算欧氏距离,取前n个最小值得到分类结果
        result = knn.classify(n, df, item)
        results_list.append(result)

    # 5.使用测试集计算正确率
    total = len(results_list)
    testing_set.reset_index(inplace=True)
    labels = testing_set[testing_set.columns[-1]] # 两个Series相减时,按照index相同的相减,所以重置测试集的index
    results = pd.Series(results_list)
    bingo = (labels - results_list).value_counts()[0] # 对测试集label和预测结果做差,统计结果为0的个数即为正确的个数
    percent = bingo / total
    print(percent)
