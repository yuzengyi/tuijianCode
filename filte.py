from MLloading import load_data, compute_pearson_similarity, DATA_PATH
from predictsc import predict


def _predict_all(uid, item_ids,ratings_matrix, item_similar):
    '''
    预测全部评分
    :param uid: 用户id
    :param item_ids: 要预测物品id列表
    :param ratings_matrix: 用户-物品打分矩阵
    :param item_similar: 物品两两间的相似度
    :return: 生成器，逐个返回预测评分
    '''
    # 逐个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, item_similar)
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating

def predict_all(uid, ratings_matrix, item_similar, filter_rule=None):
    '''
    预测全部评分，并可根据条件进行前置过滤
    :param uid: 用户ID
    :param ratings_matrix: 用户-物品打分矩阵
    :param item_similar: 物品两两间的相似度
    :param filter_rule: 过滤规则，只能是四选一，否则将抛异常："unhot","rated",["unhot","rated"],None
    :return: 生成器，逐个返回预测评分
    '''

    if not filter_rule:
        item_ids = ratings_matrix.columns
    elif isinstance(filter_rule, str) and filter_rule == "unhot":
        '''过滤非热门电影'''
        # 统计每部电影的评分数
        count = ratings_matrix.count()
        # 过滤出评分数高于10的电影，作为热门电影
        item_ids = count.where(count>10).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == "rated":
        '''过滤用户评分过的电影'''
        # 获取用户对所有电影的评分记录
        user_ratings = ratings_matrix.ix[uid]
        # 评分范围是1-5，小于6的都是评分过的，除此以外的都是没有评分的
        _ = user_ratings<6
        item_ids = _.where(_==False).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == set(["unhot", "rated"]):
        '''过滤非热门和用户已经评分过的电影'''
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index

        user_ratings = ratings_matrix.loc[uid]
        _ = user_ratings < 6
        ids2 = _.where(_ == False).dropna().index
        # 取二者交集
        item_ids = set(ids1)&set(ids2)
    else:
        raise Exception("无效的过滤参数")

    yield from _predict_all(uid, item_ids, ratings_matrix, item_similar)

if __name__ == '__main__':
    ratings_matrix = load_data(DATA_PATH)

    item_similar = compute_pearson_similarity(ratings_matrix, based="item")

    for result in predict_all(1, ratings_matrix, item_similar, filter_rule=["unhot", "rated"]):
        print(result)
