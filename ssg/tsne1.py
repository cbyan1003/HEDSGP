import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np

# 加载手写数字数据集
digits = load_digits()
data, target = digits.data, digits.target

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# 假设你已经进行了聚类
kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(tsne_result)

# 计算每个聚类的中心
cluster_centers = np.array([np.mean(tsne_result[cluster_labels == i], axis=0) for i in range(10)])

# 绘制 t-SNE 散点图
plt.figure(figsize=(8, 8), dpi=300)
for i in range(10):  # 0到9的数字
    indices = cluster_labels == i
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], alpha=0.8, linewidths=0.5)

    # 将标签放在每个聚类中心上
    plt.text(cluster_centers[i, 0], cluster_centers[i, 1], str(i), fontsize=8, ha='center', va='center')

plt.title('t-SNE Visualization of Handwritten Digits with Cluster Centers')
plt.show()