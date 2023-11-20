import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 生成正弦波数据
timesteps = 100
x = np.linspace(0, 2 * np.pi, timesteps)
y = np.sin(x)

# 数据预处理
X, Y = [], []
for i in range(len(y)-1):
    X.append(y[i])
    Y.append(y[i + 1])
X, Y = np.array(X).reshape(-1, 1, 1), np.array(Y).reshape(-1, 1)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 训练模型
model.fit(X, Y, epochs=200, verbose=0)

# 预测下一个值
next_value = model.predict(np.array([y[-2]]).reshape(1, 1, 1))
print("Predicted next value:", next_value[0][0])
