#导入语句: 文件中导入了 MLloading 模块。这可能是用于数据加载和处理的自定义模块。
import MLloading
    #预测指定用户指定评分的方法
def predict(uid, iid, ratings_matrix, user_similar):
    '''
    预测给定用户对给定物品的评分值
    :param uid: 用户ID
    :param iid: 物品ID
    :param ratings_matrix: 用户-物品评分矩阵
    :param user_similar: 用户两两相似度矩阵
    :return: 预测的评分值
    '''
    print("开始预测用户<%d>对电影<%d>的评分..."%(uid, iid))
    # 1. 找出uid用户的相似用户
    similar_users = user_similar[uid].drop([uid]).dropna()
    # print("similar_users")
    # print(similar_users)
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users>0).dropna()
    if similar_users.empty is True:
        raise Exception("用户<%d>没有相似的用户" % uid)

    # 2. 从uid用户的近邻相似用户中筛选出对iid物品有评分记录的近邻用户
    ids = set(ratings_matrix[iid].dropna().index)&set(similar_users.index)
    #集合的交集操作符 & 来找出两个集合中共同的元素，即既对物品 iid 有评分，又与用户 uid 相似的用户集合。
    # print("ids")
    # print(ids)
    finally_similar_users = similar_users.loc[list(ids)]
    print("finally_similar_users")
    print(finally_similar_users)
    # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    sum_up = 0    # 评分预测公式的分子部分的值
    sum_down = 0    # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.items():
        # 近邻用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算分子的值（对相似度与评分数据的乘积进行求和）
        sum_up += similarity * sim_user_rating_for_item
        # 计算分母的值(对相似度进行求和)
        sum_down += similarity

    # 计算预测的评分值并返回
    predict_rating = sum_up/sum_down
    print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (uid, iid, predict_rating))
    return round(predict_rating, 2)
    #预测所有的评分的方法
def predict_all(uid, ratings_matrix, user_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param ratings_matrix: 用户-物品打分矩阵
    :param user_similar: 用户两两间的相似度
    :return: 生成器，逐个返回预测评分
    '''
    # 准备要预测的物品的id列表
    item_ids = ratings_matrix.columns
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating
if __name__ == '__main__':
    ratings_matrix = MLloading.load_data(MLloading.DATA_PATH)
    user_similar = MLloading.compute_pearson_similarity(ratings_matrix, based="user")
    for i in predict_all(1, ratings_matrix, user_similar):
        pass
if __name__ == '__main__':
    ratings_matrix = MLloading.load_data(MLloading.DATA_PATH)
    user_similar = MLloading.compute_pearson_similarity(ratings_matrix, based="user")
    # 预测用户1对物品1的评分
    predict(1, 1, ratings_matrix, user_similar)
    # 预测用户1对物品2的评分
    predict(1, 2, ratings_matrix, user_similar)