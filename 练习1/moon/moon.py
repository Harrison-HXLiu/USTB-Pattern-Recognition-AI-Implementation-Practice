# 导入库
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. 生成 Moon 数据集
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. 定义画决策边界函数
def plot_decision_boundary(model, X, y, title):
    h = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")


# 4. 建立不同模型
models = {
    "Linear Kernel": SVC(kernel='linear', C=1),
    "RBF Kernel": SVC(kernel='rbf', C=1, gamma='scale')
}

# 5. 绘图
plt.figure(figsize=(12, 5))

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    plt.subplot(1, 2, i + 1)
    plot_decision_boundary(model, X, y, f"{name}\nAccuracy={acc:.2f}")

plt.tight_layout()
plt.show()