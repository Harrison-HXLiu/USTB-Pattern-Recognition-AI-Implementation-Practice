# =========================
# 导入库
# ========================
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =========================
# 1. 加载 MNIST 数据集
# =========================

print("Loading MNIST...")

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data
y = mnist.target

# 只取部分样本（否则 t-SNE 很慢）
X = X[:2000]
y = y[:2000]

print("Data shape:", X.shape)

# =========================
# 2. PCA 降维
# =========================

print("Running PCA...")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# =========================
# 3. t-SNE 降维
# =========================

print("Running t-SNE...")

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

X_tsne = tsne.fit_transform(X)

# =========================
# 4. 可视化
# =========================

plt.figure(figsize=(12,5))

# PCA图
plt.subplot(1,2,1)

scatter = plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=y.astype(int),
    cmap='tab10',
    s=10
)

plt.title("PCA Visualization")

# t-SNE图
plt.subplot(1,2,2)

scatter = plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y.astype(int),
    cmap='tab10',
    s=10
)

plt.title("t-SNE Visualization")

plt.show()