# ......
from MLloading import load_data, DATA_PATH, compute_pearson_similarity
from filte import predict_all


def top_k_rs_result(k):
    ratings_matrix = load_data(DATA_PATH)

    item_similar = compute_pearson_similarity(ratings_matrix, based="item")
    results = predict_all(1, ratings_matrix, item_similar, filter_rule=["unhot", "rated"])
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]

if __name__ == '__main__':
    from pprint import pprint
    result = top_k_rs_result(20)
    print("result")
    print(result)
    pprint(result)
