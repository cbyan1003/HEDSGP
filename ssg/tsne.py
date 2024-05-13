import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
# from sklearn.manifold import TSNE
from openTSNE import TSNE

# 加载手写数字数据集
digits = load_digits()
data, target = digits.data, digits.target

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit(data)



# 绘制 t-SNE 散点图
plt.figure(figsize=(8, 8))
for i in range(10):  # 0到9的数字
    indices = target == i
    plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], label=str(i),  cmap='tab20', alpha=0.8, linewidths=0.5, marker='+')

plt.title('t-SNE Visualization of Handwritten Digits')
plt.legend()
plt.show()