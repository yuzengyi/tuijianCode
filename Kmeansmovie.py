import os
import pandas as pd

DATA_PATH = "./ml-latest-small/movies.csv"  # 存储评分数据的文件路径。
all_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
              'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
def load_data(data_path):
    '''
    加载数据
    :param data_path: 数据集路径
    :return: DataFrame对象
    '''
    # 检查文件路径是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"指定的文件路径不存在: {data_path}")

    # 使用 pandas 读取 CSV 文件
    data = pd.read_csv(data_path)

    return data

# 使用示例
data = load_data(DATA_PATH)
rows_genres=[]
# 遍历每一行并分割 genres 列
for index, row in data.iterrows():
    genres = row['genres'].split('|')
    rows_genres.append(genres)

# print(rows_genres)
# Function to create 19-dimensional vector for each row
def create_genre_vector(genres_list, all_genres):
    return [1 if genre in genres_list else 0 for genre in all_genres]

# Creating vectors for each row
genre_vectors = [create_genre_vector(row_genres, all_genres) for row_genres in rows_genres]
# print(genre_vectors)
# Adding the genre vectors as new columns to the DataFrame
for i, genre in enumerate(all_genres):
    data[genre] = [vector[i] for vector in genre_vectors]
print(data)
# Save the updated DataFrame to a new CSV file
output_csv_path = 'updated_movies.csv'
data.to_csv(output_csv_path, index=False)


