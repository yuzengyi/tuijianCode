import torch
import torch
import torch.nn as nn

# 原始时间序列
time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# 重新定义LSTM模型
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.linear(output[-1])
        return output

# 创建输入序列和目标序列
input_seq = []
target_seq = []
seq_length = 3  # 使用过去3个时间步来预测下一个

for i in range(len(time_series) - seq_length):
    input_seq.append(time_series[i:i+seq_length])
    target_seq.append(time_series[i+seq_length])

# 转换为PyTorch张量
input_seq = torch.FloatTensor(input_seq).view(-1, 1, seq_length)
target_seq = torch.FloatTensor(target_seq).view(-1, 1)
# 实例化模型
model = SimpleLSTM(input_size=seq_length, hidden_size=20, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 训练模型
for epoch in range(100):
    total_loss = 0
    for i in range(len(input_seq)):
        optimizer.zero_grad()
        # 确保输入保持三维形状：(序列长度, 批次大小, 特征数量)
        input_data = input_seq[i].view(1, 1, -1)
        output = model(input_data)
        loss = criterion(output, target_seq[i])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 9:  # 每10轮输出一次平均损失
        print(f"Epoch {epoch+1} Loss: {total_loss / len(input_seq)}")

# 输出最后一次训练后的预测结果
predictions = []
with torch.no_grad():
    for i in range(len(input_seq)):
        input_data = input_seq[i].view(1, 1, -1)
        pred = model(input_data)
        predictions.append(pred.item())

print(predictions)