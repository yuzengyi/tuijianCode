import os

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import euclidean
def load_data(data_path):
    '''
    加载数据
    :param data_path: 数据集路径
    :param cache_path: 数据集缓存路径
    :return: 用户-物品评分矩阵
    '''
    # 数据集缓存地址
    cache_path = os.path.join(CACHE_DIR, "ratings_matrix.cache")

    print("开始加载数据集...")
    if os.path.exists(cache_path):  # 判断是否存在缓存文件
        print("加载缓存中...")
        ratings_matrix = pd.read_pickle(cache_path)
        print("从缓存加载数据集完毕")
    else:
        print("加载新数据中...")
        # 设置要加载的数据字段的类型
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        # 透视表，将电影ID转换为列名称，转换成为一个User-Movie的评分矩阵
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        # 存入缓存文件
        ratings_matrix.to_pickle(cache_path)
        print("数据集加载完毕")
    return ratings_matrix
def compute_similarity(ratings_matrix, method="pearson"):
    '''
    根据指定方法计算相似度
    :param ratings_matrix: 用户-物品评分矩阵
    :param method: "pearson", "cosine", "euclidean", "jaccard"
    :return: 相似度矩阵
    '''
    if method == "pearson":
        similarity = ratings_matrix.T.corr()
    elif method == "cosine":
        # 在计算余弦相似度之前用0填充NaN值
        filled_matrix = ratings_matrix.fillna(0)
        similarity = pd.DataFrame(cosine_similarity(filled_matrix), index=ratings_matrix.index,
                                  columns=ratings_matrix.index)
    elif method == "euclidean":
        # 在计算欧式距离之前用0填充NaN值
        filled_matrix = ratings_matrix.fillna(0)
        dist_matrix = pd.DataFrame(
            euclidean_distances(filled_matrix),
            index=ratings_matrix.index,
            columns=ratings_matrix.index
        )
        similarity = 1 / (1 + dist_matrix)
    elif method == "jaccard":
        # # 计算杰卡德相似度（适用于二元数据）
        # bool_matrix = ratings_matrix.notnull()
        # intersection = np.dot(bool_matrix, bool_matrix.T)
        # union = np.dot(bool_matrix.sum(axis=1).values.reshape(-1, 1), np.ones((1, len(bool_matrix)))) + \
        #         np.dot(np.ones((len(bool_matrix), 1)), bool_matrix.sum(axis=1).values.reshape(1, -1)) - \
        #         intersection
        # similarity = pd.DataFrame(intersection / union, index=ratings_matrix.index, columns=ratings_matrix.index)
        # 将评分转换为二元值（1表示评分，0表示未评分）
        bool_matrix = ratings_matrix.notnull()

        # 初始化一个空的相似度矩阵
        similarity_matrix = pd.DataFrame(np.zeros((len(bool_matrix), len(bool_matrix))),
                                         index=bool_matrix.index,
                                         columns=bool_matrix.index)

        # 计算杰卡德相似度
        for i in range(len(bool_matrix)):
            for j in range(len(bool_matrix)):
                intersection = np.sum(bool_matrix.iloc[i] & bool_matrix.iloc[j])
                union = np.sum(bool_matrix.iloc[i] | bool_matrix.iloc[j])
                similarity_matrix.iloc[i, j] = intersection / union if union != 0 else 0

        return similarity_matrix
    else:
        raise ValueError("Unsupported similarity method: %s" % method)
    return similarity
# def compute_similarity(ratings_matrix, method="pearson"):
#     '''
#     根据指定方法计算用户间的相似度
#     :param ratings_matrix: 用户-物品评分矩阵
#     :param method: "pearson", "cosine", "euclidean", "jaccard"
#     :return: 相似度矩阵
#     '''
#     if method == "pearson":
#         similarity = ratings_matrix.corr()  # 计算用户间的皮尔逊相关系数
#     elif method == "cosine":
#         # 在计算余弦相似度之前用0填充NaN值
#         filled_matrix = ratings_matrix.fillna(0)
#         similarity = pd.DataFrame(cosine_similarity(filled_matrix), index=ratings_matrix.index, columns=ratings_matrix.index)
#     elif method == "euclidean":
#         # 同样在计算欧式距离之前用0填充NaN值
#         filled_matrix = ratings_matrix.fillna(0)
#         dist_matrix = pd.DataFrame(
#             euclidean_distances(filled_matrix),
#             index=ratings_matrix.index,
#             columns=ratings_matrix.index
#         )
#         similarity = 1 / (1 + dist_matrix)
#     elif method == "jaccard":
#          # 计算杰卡德相似度
#          bool_matrix = ratings_matrix.T.notnull()
#          intersection = np.dot(bool_matrix, bool_matrix.T)
#          union = bool_matrix.sum(axis=1).values.reshape(-1, 1) + bool_matrix.sum(axis=1) - intersection
#          similarity = pd.DataFrame(intersection / union, index=ratings_matrix.columns, columns=ratings_matrix.columns)
#
#     else:
#         raise ValueError("Unsupported similarity method: %s" % method)
#     return similarity

DATA_PATH = "./ml-latest-small/ratings.csv"  #存储评分数据的文件路径。
CACHE_DIR = "./cache/"  #缓存文件存储的目录。
# 示例：计算并保存所有相似度矩阵
methods = ["pearson", "cosine", "euclidean", "jaccard"]

writer = pd.ExcelWriter("similarity_comparison.xlsx")
for method in methods:
    ratings_matrix = load_data(DATA_PATH)
    print(ratings_matrix)
    print(method)
    similarity_matrix = compute_similarity(ratings_matrix, method=method)
    print(similarity_matrix)
    similarity_matrix.to_excel(writer, sheet_name=method)
writer.close()
