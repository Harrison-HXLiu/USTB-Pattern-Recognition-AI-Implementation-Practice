# 导入库
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# 1. 加载鸢尾花数据集
iris = load_iris()

X = iris.data
y = iris.target

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# ==============================
# 不进行标准化
# ==============================

k_values = [1, 3, 5, 15]
acc_without_scaler = []

print("【未标准化】")

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    acc_without_scaler.append(acc)

    print(f"k={k}, 准确率={acc:.4f}")

# ==============================
# 进行 MinMax 标准化
# ==============================

# 创建标准化对象
scaler = MinMaxScaler()

# 训练集拟合并转换
X_train_scaled = scaler.fit_transform(X_train)

# 测试集转换
X_test_scaled = scaler.transform(X_test)

acc_with_scaler = []

print("\n【标准化后】")

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)

    acc_with_scaler.append(acc)

    print(f"k={k}, 准确率={acc:.4f}")

# ==============================
# 绘图
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(k_values, acc_without_scaler,
         marker='o',
         label='Without Scaling')

plt.plot(k_values, acc_with_scaler,
         marker='s',
         label='With MinMaxScaler')

plt.xlabel("k value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy under Different k Values")

plt.legend()

plt.grid(True)

plt.show()