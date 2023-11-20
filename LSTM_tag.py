import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('punkt')
import numpy as np
DATA_PATH = "./ml-latest-small/tags.csv"  # 存储评分数据的文件路径。
# Load the data
tags_df = pd.read_csv(DATA_PATH )

# Take a look at the first few rows of the dataframe
print(tags_df)


# Assuming the maximum length of sequences is 10 for this example
MAX_SEQUENCE_LENGTH = 10

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tags_df['tag'])

# Convert the tags into sequences
sequences = tokenizer.texts_to_sequences(tags_df['tag'])

# Pad the sequences to ensure uniform length
tag_data_padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print(tag_data_padded)
# Display the processed features
tag_data_padded[:5]  # Displaying the first 5 for brevity


# Tokenize the 'tag' column
tags_df['tokenized'] = tags_df['tag'].apply(word_tokenize)

# Flatten the list of tokens to fit the Label Encoder
all_tokens = [token for sublist in tags_df['tokenized'].tolist() for token in sublist]

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(all_tokens)

# Convert each list of tokens into a list of integers
tags_df['encoded'] = tags_df['tokenized'].apply(lambda x: label_encoder.transform(x).tolist())

# Determine the maximum sequence length
max_seq_length = max(tags_df['encoded'].apply(len))

# Pad the sequences
padded_sequences = np.array([np.pad(encoded, (0, max_seq_length - len(encoded)), 'constant') for encoded in tags_df['encoded']])

padded_sequences[:5]  # Show the first 5 padded sequences
# Basic tokenization by splitting the tags on spaces
tags_df['tokenized'] = tags_df['tag'].apply(lambda x: x.split())

# Flatten the list of tokens to fit the Label Encoder
all_tokens = [token for sublist in tags_df['tokenized'].tolist() for token in sublist]

# Initialize and fit the label encoder
label_encoder = LabelEncoder()
label_encoder.fit(all_tokens)

# Convert each list of tokens into a list of integers
tags_df['encoded'] = tags_df['tokenized'].apply(lambda x: label_encoder.transform(x).tolist())

# Determine the maximum sequence length
max_seq_length = max(tags_df['encoded'].apply(len))

# Pad the sequences
padded_sequences = np.array([np.pad(encoded, (0, max_seq_length - len(encoded)), 'constant', constant_values=0) for encoded in tags_df['encoded']])
print(padded_sequences)
padded_sequences[:5]  # Show the first 5 padded sequences

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
for epoch in range(300):  # 这里只是一个示例，实际的 epoch 数需要您根据具体情况来定
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


