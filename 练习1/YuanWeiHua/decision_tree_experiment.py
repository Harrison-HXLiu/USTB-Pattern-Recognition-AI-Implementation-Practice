"""
决策树构建与剪枝实验
- 使用鸢尾花数据集（Iris）
- 比较信息增益（entropy）与基尼指数（gini）划分准则
- 测试预剪枝（max_depth, min_samples_leaf）与后剪枝（cost-complexity pruning）

运行：
    python decision_tree_experiment.py

生成：控制台输出与保存在 `figures/` 的图像文件
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score


def baseline_compare(X_train, X_test, y_train, y_test):
    print("== 基线：比较 gini vs entropy ==")
    for criterion in ("gini", "entropy"):
        clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"criterion={criterion:7s} depth={clf.get_depth():2d} nodes={clf.tree_.node_count:3d}  accuracy={acc:.4f}")


def pre_pruning_grid(X_train, X_test, y_train, y_test, out_dir):
    print('\n== 预剪枝：max_depth 与 min_samples_leaf 网格搜索 ==')
    max_depths = [1, 2, 3, 4, 5, None]
    min_samples = [1, 2, 5, 10]

    results = []
    for max_d in max_depths:
        for m in min_samples:
            clf = DecisionTreeClassifier(criterion='gini', max_depth=max_d, min_samples_leaf=m, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append((max_d, m, clf.get_depth(), clf.tree_.node_count, acc))
            print(f"max_depth={str(max_d):4s} min_samples_leaf={m:2d} -> depth={clf.get_depth():2d} nodes={clf.tree_.node_count:3d} acc={acc:.4f}")

    # 可视化：根据 max_depth 汇总最佳 min_samples
    depths = []
    accs = []
    labels = []
    for max_d in max_depths:
        filtered = [r for r in results if r[0] == max_d]
        best = max(filtered, key=lambda x: x[4])
        depths.append(best[2])
        accs.append(best[4])
        labels.append(str(max_d))

    plt.figure()
    plt.plot(labels, accs, marker='o')
    plt.xlabel('max_depth')
    plt.ylabel('best accuracy (varying min_samples_leaf)')
    plt.title('Pre-pruning: best accuracy vs max_depth')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'prepruning_best_acc_vs_maxdepth.png'))
    plt.close()


def post_pruning_ccp(X_train, X_test, y_train, y_test, out_dir):
    print('\n== 后剪枝：Cost-Complexity Pruning (基于测试集选择 ccp_alpha) ==')
    base = DecisionTreeClassifier(random_state=42)
    path = base.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = path.ccp_alphas
    ccp_alphas = np.unique(ccp_alphas)
    alphas = []
    test_accs = []
    node_counts = []

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        alphas.append(alpha)
        test_accs.append(acc)
        node_counts.append(clf.tree_.node_count)

    best_idx = int(np.argmax(test_accs))
    best_alpha = alphas[best_idx]
    best_acc = test_accs[best_idx]
    best_nodes = node_counts[best_idx]

    print(f"最佳 ccp_alpha={best_alpha:.6f} -> test_acc={best_acc:.4f} nodes={best_nodes}")

    # 绘图：accuracy 与 nodes 随 alpha 的变化
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(alphas, test_accs, marker='o')
    plt.xscale('log')
    plt.xlabel('ccp_alpha (log scale)')
    plt.ylabel('test accuracy')
    plt.title('Post-pruning: accuracy vs ccp_alpha')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(alphas, node_counts, marker='s')
    plt.xscale('log')
    plt.xlabel('ccp_alpha (log scale)')
    plt.ylabel('number of nodes')
    plt.title('Post-pruning: nodes vs ccp_alpha')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'postpruning_acc_nodes_vs_ccp.png'))
    plt.close()


def save_example_tree(X_train, y_train, out_dir, criterion='gini'):
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_tree(clf, filled=True, ax=ax, feature_names=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid'])
    plt.title(f'Decision Tree ({criterion})')
    plt.tight_layout()
    fname = os.path.join(out_dir, f'tree_{criterion}.png')
    plt.savefig(fname)
    plt.close()


def main():
    out_dir = 'figures'
    os.makedirs(out_dir, exist_ok=True)

    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    baseline_compare(X_train, X_test, y_train, y_test)

    pre_pruning_grid(X_train, X_test, y_train, y_test, out_dir)

    post_pruning_ccp(X_train, X_test, y_train, y_test, out_dir)

    save_example_tree(X_train, y_train, out_dir, criterion='gini')
    save_example_tree(X_train, y_train, out_dir, criterion='entropy')

    print('\n图像已保存到', out_dir)


if __name__ == '__main__':
    main()
