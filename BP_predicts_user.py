import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from BP_predicts import genre_features_array

# 加载数据
data = pd.read_csv('./ml-latest-small/all_del.csv')

# 用户ID和电影类型特征
user_ids = data['userId'].values
genre_features = data.columns[3:-1]  # 电影类型特征列，跳过 userId, movieId 和 adjusted_rating
movie_features = data[genre_features].values
# # Convert to numpy array and reshape for single prediction
# genre_features_array = genre_features.values.reshape(1, -1)
# 为用户ID编码
user_id_encoder = LabelEncoder()
user_ids_encoded = user_id_encoder.fit_transform(user_ids)

# 标签
ratings = data['adjusted_rating'].values

# 转换为PyTorch张量
user_ids_tensor = torch.tensor(user_ids_encoded, dtype=torch.long)
print(user_ids_tensor)
movie_features_tensor = torch.tensor(movie_features, dtype=torch.float32)
print(movie_features_tensor)
ratings_tensor = torch.tensor(ratings, dtype=torch.float32).view(-1, 1)
print(ratings_tensor)
#这一行定义了一个名为 MovieDataset 的新类，它从 PyTorch 的 Dataset 类继承。这个自定义类旨在处理电影推荐数据。
# 自定义数据集
class MovieDataset(Dataset):
    #方法用于初始化数据集对象。它接受三个参数：user_ids、movie_features 和 ratings，这些数组包含相应的数据。
    def __init__(self, user_ids, movie_features, ratings):
        self.user_ids = user_ids
        self.movie_features = movie_features
        self.ratings = ratings
    #方法返回数据集中的项目数量。在这里，它返回 ratings 数组的长度，用来确定示例的数量。
    def __len__(self):
        return len(self.ratings)
    #方法检索特定索引 idx 处的数据。它返回一个包含给定索引处的用户 ID、电影特征和评分的元组。
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_features[idx], self.ratings[idx]
#使用用户 ID、电影特征和评分的张量创建了 MovieDataset 的一个实例。然后将数据集分割成训练集和测试集，其中20%的数据用于测试。
# 分割数据
dataset = MovieDataset(user_ids_tensor, movie_features_tensor, ratings_tensor)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# 创建数据加载器
#这些行为训练集和测试集创建了 DataLoader 实例。DataLoader 负责批量处理数据，可选地为训练目的打乱数据。批处理大小设置为32。
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 神经网络模型
#这定义了一个名为 Net 的新类，这是一个继承自 PyTorch 的 nn.Module 的神经网络模型。
class Net(nn.Module):
    #构造函数接收总用户数、电影特征数量和用户嵌入大小。
    def __init__(self, num_users, num_movie_features, user_embedding_size):
        super(Net, self).__init__()
        #这行代码调用了基类 (nn.Module) 的构造函数以正确地初始化它。
        self.user_embedding = nn.Embedding(num_users, user_embedding_size)
        #创建了一个嵌入层，以学习用户 ID 的密集表示（嵌入）。每个用户 ID 将被映射到一个8维向量。
        self.fc1 = nn.Linear(num_movie_features + user_embedding_size, 64)
        #这里定义了三个全连接层，第一个层接收电影特征和用户嵌入大小的组合作为输入。
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    #forward 方法定义了网络的前向传播。它接受用户 ID 和电影特征作为输入。
    def forward(self, user_ids, movie_features):
        #用户 ID 通过嵌入层获取用户嵌入。
        user_embedding = self.user_embedding(user_ids)
        #用户嵌入和电影特征沿着维度1（列）连接。
        x = torch.cat([user_embedding, movie_features], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 用户总数和嵌入维度
#计算唯一用户的总数，并将嵌入大小设置为每个用户 ID 的8维。
num_users = len(user_id_encoder.classes_)
user_embedding_size = 8  # 假设我们为每个用户ID分配8维的嵌入
print(num_users)
# 实例化并训练模型
#使用用户数、电影特征数量和嵌入大小创建了 Net 模型的一个实例。
model = Net(num_users, len(genre_features), user_embedding_size)
criterion = nn.MSELoss()
#损失函数设置为均方误差损失（Mean Squared Error Loss），它用于评估回归任务的性能，计算预测值和真实值之间的平均平方差。
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#优化器使用 Adam 算法，它是一种基于自适应估计的方法来调整学习率。这里，它被配置为用 0.001 的学习率来更新模型的参数。
# 训练循环
epochs = 10
for epoch in range(epochs):
    for user_ids_batch, movie_features_batch, ratings_batch in train_loader:
        pred = model(user_ids_batch, movie_features_batch)
        loss = criterion(pred, ratings_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    test_loss = sum(criterion(model(user_ids_batch, movie_features_batch), ratings_batch) for user_ids_batch, movie_features_batch, ratings_batch in test_loader) / len(test_loader)
print(f'Test Loss: {test_loss.item()}')

# # 预测用户1对电影2的评分
# user_id_for_prediction = torch.tensor([user_id_encoder.transform([1])], dtype=torch.long)  # 用户ID为1
# movie_2_features = torch.tensor(genre_features_array, dtype=torch.float32)  # 电影2的特征
#
# model.eval()  # 设置为评估模式
# with torch.no_grad():
#     movie_2_rating_pred = model(user_id_for_prediction, movie_2_features)
# print(f'Predicted rating for user 1 on movie 2: {movie_2_rating_pred.item()}')
# 修正用户ID转换为张量的过程
user_id_for_prediction_np = user_id_encoder.transform([6])  # 用户ID为1的转换
user_id_for_prediction = torch.tensor(user_id_for_prediction_np, dtype=torch.long)

# 获取电影2的特征和用户1的嵌入
movie_2_features = torch.tensor(genre_features_array, dtype=torch.float32).reshape(1, -1)  # 重塑为 (1, 特征数)
print("movie_2_features")
print(movie_2_features)
user_embedding_for_user_1 = model.user_embedding(user_id_for_prediction).reshape(1, -1)  # 重塑为 (1, 嵌入维度)
print("user_embedding_for_user_1")
print(user_embedding_for_user_1)
# 使用模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():
    movie_2_rating_pred = model(user_id_for_prediction, movie_2_features)
    print(f'Predicted rating for user 6 on movie 2: {movie_2_rating_pred.item()}')
