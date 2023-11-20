import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# 数据加载
movies_df = pd.read_csv('./ml-latest-small/movies.csv')
ratings_df = pd.read_csv('./ml-latest-small/ratings.csv')
#print(ratings_df)
tags_df = pd.read_csv('./ml-latest-small/tags.csv')
# 定义所有可能的电影类型
genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
              'Sci-Fi', 'Thriller', 'War', 'Western', '(no genres listed)']
# 创建一个空的 NumPy 数组，用于存储 one-hot 编码
genres_encoded1 = np.zeros((len(movies_df), len(genres)))

# 遍历每部电影，填充编码
for i, row in enumerate(movies_df['genres']):
    for genre in row.split('|'):
        if genre in genres:
            genres_encoded1[i][genres.index(genre)] = 1
# 将结果保存为 CSV 文件
genres_encoded1_df=pd.DataFrame(genres_encoded1,columns=genres)
genres_encoded1_df.to_csv('genres_encoded_enhanced.csv', index=False)
# 文本编码 - 将 genres 和 tags 转换为 one-hot 编码
# vectorizer_genres = CountVectorizer()
vectorizer_tags = CountVectorizer()

# genres_encoded = vectorizer_genres.fit_transform(movies_df['genres']).toarray()
tags_encoded = vectorizer_tags.fit_transform(tags_df['tag']).toarray()

# Creating DataFrame from the encoded arrays
# genres_encoded_df = pd.DataFrame(genres_encoded, columns=vectorizer_genres.get_feature_names_out())
tags_encoded_df = pd.DataFrame(tags_encoded, columns=vectorizer_tags.get_feature_names_out())

# Saving the encoded data to CSV files
genres_encoded_path = 'genres_encoded.csv'
tags_encoded_path = 'tags_encoded.csv'
#
# genres_encoded_df.to_csv(genres_encoded_path, index=False)
tags_encoded_df.to_csv(tags_encoded_path, index=False)
#
# print(genres_encoded_path, tags_encoded_path, genres_encoded_df.head(), tags_encoded_df.head())
# 时间特征处理
# 将 timestamp 转换为 datetime，并提取年份、月份等特征
ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
print(ratings_df['datetime'])
ratings_df['year'] = ratings_df['datetime'].dt.year
ratings_df['month'] = ratings_df['datetime'].dt.month
ratings_df['day'] = ratings_df['datetime'].dt.day
#
# # 构建 PyTorch LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)  # 预测一个输出（评分）

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 模型参数
input_size = genres_encoded1.shape[1] + tags_encoded.shape[1] + 3  # genres, tags 和时间特征
hidden_size = 50  # 可以调整
num_layers = 2   # 可以调整
#
# 实例化模型
model = LSTMModel(input_size, hidden_size, num_layers)

# 模型训练和验证
# 注意：需要将数据转换为 PyTorch 张量，并按照时间序列分批输入到模型中。
# 训练代码需要根据具体数据和任务进行编写。
# 模型摘要
print(model)
