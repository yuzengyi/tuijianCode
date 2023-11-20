import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.cuda import device

DATA_PATH = "./ml-latest-small/ratings.csv"  #存储评分数据的文件路径。
from MLloading import load_data, compute_pearson_similarity, DATA_PATH
# Load the data
tags_df = pd.read_csv('./ml-latest-small/tags.csv')

# Basic tokenization by splitting the tags on spaces
tags_df['tokenized'] = tags_df['tag'].apply(lambda x: x.split())
print(tags_df['tokenized'])
# Flatten the list of tokens to fit the Label Encoder
all_tokens = [token for sublist in tags_df['tokenized'].tolist() for token in sublist]

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(all_tokens)

# Convert each list of tokens into a list of integers
tags_df['encoded'] = tags_df['tokenized'].apply(lambda x: label_encoder.transform(x).tolist())

# Normalize the timestamps
scaler = StandardScaler()
tags_df['normalized_timestamp'] = scaler.fit_transform(tags_df[['timestamp']])

# Determine the maximum sequence length
max_seq_length = max(tags_df['encoded'].apply(len))

# Pad the sequences
padded_sequences = np.array([np.pad(encoded, (0, max_seq_length - len(encoded)), 'constant', constant_values=0) for encoded in tags_df['encoded']])
print(padded_sequences)
# Include the normalized timestamp as a feature
padded_sequences_with_timestamp = np.hstack((padded_sequences, tags_df['normalized_timestamp'].values.reshape(-1, 1)))
print(padded_sequences_with_timestamp)
padded_sequences_with_timestamp[:5]  # Show the first 5 sequences with timestamp included
import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们的序列长度和特征数量已经被定义
SEQUENCE_LENGTH = padded_sequences_with_timestamp.shape[1] - 1  # 减去时间戳
FEATURES_NUM = 1  # 只有标签序列，如果您使用了其他特征，需要相应地调整


class LSTMTagPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, tagset_size):
        super(LSTMTagPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequence):
        lstm_out, _ = self.lstm(sequence.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_scores = nn.functional.log_softmax(tag_space, dim=1)
        return tag_scores


# 实例化模型
model = LSTMTagPredictor(input_dim=FEATURES_NUM, hidden_dim=128, tagset_size=len(label_encoder.classes_))

# 定义损失函数和优化器
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 准备训练数据
train_sequences = torch.tensor(padded_sequences_with_timestamp[:, :-1], dtype=torch.float)
train_timestamps = torch.tensor(padded_sequences_with_timestamp[:, -1], dtype=torch.float).view(-1, 1)
train_targets = torch.tensor(padded_sequences, dtype=torch.long)

# 训练模型
for epoch in range(10):  # 这里只是一个示例，实际的 epoch 数需要您根据具体情况来定
    for i in range(len(train_sequences)):
        sequence = train_sequences[i]
        timestamps = train_timestamps[i]
        targets = train_targets[i]

        # 将模型的参数梯度设置为0
        model.zero_grad()

        # 前向传播
        tag_scores = model(sequence)

        # 计算损失，执行反向传播
        loss = loss_function(tag_scores, targets)
        loss.backward()

        # 更新模型参数
        optimizer.step()

    # 打印每个 epoch 的损失
    print(f'Epoch: {epoch + 1}, Loss: {loss.item()}')
# LSTM 模型的定义和训练过程与您提供的代码相同。

# 假设模型已经训练并准备好了
# 此函数将接收一个用户ID和电影ID，并返回该电影的预测标签分数
# def predict_tags(user_id, movie_id, model, sequence_length):
#     # 从训练数据中获取该用户对该电影的标签序列
#     sequence = train_sequences[train_sequences[:, 0] == user_id][train_sequences[:, 1] == movie_id]
#     if len(sequence) == 0:
#         # 如果没有历史标签数据，则返回均匀分布的标签分数
#         return np.full((len(label_encoder.classes_),), 1.0 / len(label_encoder.classes_))
#     # 使用模型进行预测
#     model.eval()  # 将模型设置为评估模式
#     with torch.no_grad():
#         tag_scores = model(sequence)
#         return tag_scores.numpy()
#
# # 生成推荐的函数
# def make_recommendations(user_id, model, user_movie_matrix, num_recommendations=5):
#     # 获取该用户未评价的电影列表
#     user_unrated_movies = user_movie_matrix.columns[~user_movie_matrix.loc[user_id].astype(bool)]
#     # 存储电影和它们的预测得分
#     scores = {}
#     for movie_id in user_unrated_movies:
#         # 预测电影标签的得分
#         predicted_scores = predict_tags(user_id, movie_id, model, SEQUENCE_LENGTH)
#         # 将预测得分加入到分数字典
#         scores[movie_id] = predicted_scores.mean()  # 这里简化处理，使用平均分数
#     # 根据得分排序并选择最高的几个
#     recommended_movie_ids = sorted(scores, key=scores.get, reverse=True)[:num_recommendations]
#     return recommended_movie_ids
# user_movie_matrix = load_data(DATA_PATH)
# print(user_movie_matrix)
# # 假设用户ID为2，我们为该用户生成推荐
# recommended_movies = make_recommendations(2, model, user_movie_matrix)
# print(f"Recommended movies for user 2: {recommended_movies}
def predict_tags(sequence, model):
    # 模型预测
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        tag_scores = model(sequence)
    return tag_scores

# 使用训练好的 LSTM 模型为每个电影生成一个预测分数。
# 这里我们使用简化的逻辑，仅用于演示。
def make_recommendations1(user_id, model, user_movie_matrix, num_recommendations=5):
    # 获取所有电影的列表
    all_movies = user_movie_matrix.columns.tolist()
    # 存储电影和它们的预测得分
    scores = {}
    for movie_id in all_movies:
        # 获取用户对该电影的历史标签序列
        sequence = train_sequences[train_sequences[:, 0] == user_id][train_sequences[:, 1] == movie_id]
        if len(sequence) > 0:  # 如果用户有历史标签数据
            sequence = sequence.to(device)  # 如果你使用了 GPU
            # 预测该序列的标签分数
            predicted_scores = predict_tags(sequence, model)
            # 使用预测分数的某种逻辑（例如最大值、平均值等）作为该电影的得分
            scores[movie_id] = predicted_scores.mean().item()
        else:
            # 如果没有历史数据，我们可以简单地分配一个中性分数，例如平均分数
            scores[movie_id] = 0.5  # 这是一个示例值
    # 根据得分排序并选择最高的几个
    recommended_movie_ids = sorted(scores, key=scores.get, reverse=True)[:num_recommendations]
    return recommended_movie_ids

def make_recommendations(user_id, model, user_movie_matrix, num_recommendations=5):
    # 获取所有电影的列表
    all_movies = user_movie_matrix.columns.tolist()
    # 存储电影和它们的预测得分
    scores = {}
    for movie_id in all_movies:
        # 初始化序列为None
        sequence = None
        # 遍历 train_sequences 查找匹配的用户ID和电影ID
        for seq in train_sequences:
            if seq[0] == user_id and seq[1] == movie_id:
                sequence = seq
                break
        # 如果找到了匹配的序列
        if sequence is not None:
            sequence = sequence.unsqueeze(0)  # 添加批次维度
            # 预测该序列的标签分数
            predicted_scores = predict_tags(sequence, model)
            # 使用预测分数的某种逻辑（例如最大值、平均值等）作为该电影的得分
            scores[movie_id] = predicted_scores.mean().item()
        else:
            # 如果没有历史数据，我们可以简单地分配一个中性分数
            scores[movie_id] = 0.5
    # 根据得分排序并选择最高的几个
    recommended_movie_ids = sorted(scores, key=scores.get, reverse=True)[:num_recommendations]
    return recommended_movie_ids

# # 使用修改后的函数进行推荐
# recommended_movies = make_recommendations(2, model, user_movie_matrix)
# print(f"Recommended movies for user 2: {recommended_movies}")


user_movie_matrix = load_data(DATA_PATH)
print(user_movie_matrix)
# 假设用户ID为2，我们为该用户生成推荐
recommended_movies = make_recommendations(2, model, user_movie_matrix)
print(f"Recommended movies for user 2: {recommended_movies}")

