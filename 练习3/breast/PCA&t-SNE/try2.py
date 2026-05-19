import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def load_mnist_subset(n_samples=2000, random_state=42):
    """加载 MNIST 数据集并随机抽取部分样本用于降维与可视化。"""
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    rng = np.random.RandomState(random_state)
    indices = rng.choice(len(X), size=n_samples, replace=False)
    return X[indices], y[indices]


def compute_local_neighbor_overlap(X, Y, n_neighbors=10):
    """计算原始空间与嵌入空间的局部邻居重合比例。"""
    nn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X)
    nn_emb = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(Y)

    orig_neighbors = nn_orig.kneighbors(X, return_distance=False)[:, 1:]
    emb_neighbors = nn_emb.kneighbors(Y, return_distance=False)[:, 1:]

    overlap_ratios = []
    for orig, emb in zip(orig_neighbors, emb_neighbors):
        overlap = len(set(orig).intersection(emb))
        overlap_ratios.append(overlap / n_neighbors)
    return np.mean(overlap_ratios)


def plot_embeddings(X_pca, X_tsne, y, save_path=None, show=False):
    cmap = plt.get_cmap('tab10')
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    titles = ['PCA 降维到二维', 't-SNE 降维到二维']
    embeddings = [X_pca, X_tsne]

    for ax, emb, title in zip(axes, embeddings, titles):
        for digit in range(10):
            mask = y == digit
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=12, alpha=0.7,
                color=cmap(digit), label=str(digit)
            )
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend(markerscale=1.5, fontsize=9, ncols=2, loc='best')
        ax.grid(True, linestyle='--', alpha=0.3)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
        print(f'已保存可视化图像：{save_path}')
    if show:
        plt.show()


def main():
    print('正在加载 MNIST 子集数据...')
    X, y = load_mnist_subset(n_samples=2000)

    print('正在执行 PCA...')
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    print('正在执行 t-SNE...（这一步可能需要一些时间）')
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    print('正在计算类别可分性指标与局部结构保留指标...')
    silhouette_pca = silhouette_score(X_pca, y)
    silhouette_tsne = silhouette_score(X_tsne, y)
    overlap_pca = compute_local_neighbor_overlap(X, X_pca, n_neighbors=10)
    overlap_tsne = compute_local_neighbor_overlap(X, X_tsne, n_neighbors=10)

    print('\n==== 比较结果 ====')
    print(f'PCA Silhouette Score: {silhouette_pca:.4f}')
    print(f't-SNE Silhouette Score: {silhouette_tsne:.4f}')
    print(f'PCA 局部邻居重合比例: {overlap_pca:.4f}')
    print(f't-SNE 局部邻居重合比例: {overlap_tsne:.4f}')
    print('\n结论：')
    print('  - PCA 通常更关注全局结构，类别边界较为线性，但对于复杂手写数字形状可能存在类间混叠。')
    print('  - t-SNE 更擅长保留局部结构，使相同类别样本更紧密聚集，常表现出更清晰的类簇。')
    print('  - 如果希望保持原始数据的局部邻域关系，t-SNE 通常优于 PCA。')

    plot_embeddings(X_pca, X_tsne, y, save_path='mnist_pca_tsne.png', show=False)


if __name__ == '__main__':
    main()
