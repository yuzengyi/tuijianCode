import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn as nn
import torch.optim as optim

# 加载数据
data = pd.read_csv('./ml-latest-small/all_del.csv')
#Input
# Extract features for movieId 2
movie_2_features = data[data['movieId'] == 2].iloc[0]

# Extract the genre features which are from 'Action' to the last genre column
genre_features = movie_2_features[2:-2]  # Exclude 'userId', 'movieId', 'adjusted_rating'

# Convert to numpy array and reshape for single prediction
genre_features_array = genre_features.values.reshape(1, -1)

genre_features_array, genre_features.index.tolist()  # Display the array and the feature names
# 特征和标签
feature_cols = data.columns[3:-1]  # 电影类型特征列，跳过 userId, movieId 和 adjusted_rating
X = data[feature_cols].values
y = data['adjusted_rating'].values

# 转换为PyTorch张量
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# 创建数据集和加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# 神经网络模型
class Net(nn.Module):
    def __init__(self, num_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化并训练模型
model = Net(len(feature_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
epochs = 10
for epoch in range(epochs):
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    test_loss = sum(criterion(model(xb), yb) for xb, yb in test_loader) / len(test_loader)
print(f'Test Loss: {test_loss.item()}')

# 电影2的特征
movie_2_features = torch.tensor(genre_features_array, dtype=torch.float32)

# 使用模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():
    movie_2_rating_pred = model(movie_2_features)
print(f'Predicted rating for movie 2: {movie_2_rating_pred.item()}')

# 预测
# 假设 user_1_features_for_movie_2 是用户1对电影2的特征张量
# predicted_rating = model(user_1_features_for_movie_2)

# 注意：请确保将 '/path/to/your/all_del.csv' 替换为您的数据文件的实际路径，
# 并且您已经准备好了用户1对电影2的特征张量 `user_1_features_for_movie_2` 用于预测。
