import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# 生成随机数据
np.random.seed(5)
X = np.random.rand(100, 3)

# 应用 K-means 算法
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(X)
print(labels)
# 绘制结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b']

for i in range(3):
    ax.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], c=colors[i])

# 绘制聚类中心
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='x', c='black', s=100)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
plt.title('3D K-means Clustering')
plt.show()
