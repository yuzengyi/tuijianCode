import pandas as pd
from sklearn.cluster import KMeans

DATA_PATH = "./ml-latest-small/movies.csv"  # 存储电影数据的文件路径。
all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
              'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']

# 加载数据
data = pd.read_csv(DATA_PATH)

# 创建电影类型向量
def create_genre_vector(row, all_genres):
    genres = row['genres'].split('|')
    return [1 if genre in genres else 0 for genre in all_genres]

genre_vectors = data.apply(lambda row: create_genre_vector(row, all_genres), axis=1)
genre_vectors_df = pd.DataFrame(genre_vectors.tolist(), columns=all_genres)

# 使用 KMeans 聚类
kmeans = KMeans(n_clusters=6, random_state=0).fit(genre_vectors_df)
data['Cluster6'] = kmeans.labels_

# 保存更新后的 DataFrame 到新的 CSV 文件
output_csv_path = 'updated_movies_with_clusters.csv'
data.to_csv(output_csv_path, index=False)

print(f"Updated data with clusters saved to {output_csv_path}")
