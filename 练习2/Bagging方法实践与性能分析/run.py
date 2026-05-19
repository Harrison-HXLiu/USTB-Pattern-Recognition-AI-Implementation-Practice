import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier


RANDOM_STATE = 42
CV_SPLITS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_bagging_classifier(max_samples=1.0, n_estimators=100):
    """Create a BaggingClassifier that works across sklearn versions."""
    base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "bootstrap": True,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    try:
        return BaggingClassifier(estimator=base_tree, **params)
    except TypeError:
        return BaggingClassifier(base_estimator=base_tree, **params)


def print_cv_result(name, scores):
    print(f"{name:<18} mean accuracy = {scores.mean():.4f}, variance = {scores.var():.6f}, std = {scores.std():.4f}")


def plot_learning_curve(model, x, y, cv):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=model,
        X=x,
        y=y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(9, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training accuracy")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    plt.plot(train_sizes, val_mean, marker="s", label="Cross-validation accuracy")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve of Decision Tree Bagging on Wine")
    plt.ylim(0.70, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "bagging_learning_curve.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def plot_max_samples_curve(x, y, cv):
    max_samples_values = np.linspace(0.1, 1.0, 10)
    means = []
    stds = []

    for max_samples in max_samples_values:
        model = make_bagging_classifier(max_samples=max_samples, n_estimators=100)
        scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
        means.append(scores.mean())
        stds.append(scores.std())

    means = np.array(means)
    stds = np.array(stds)

    plt.figure(figsize=(9, 5))
    plt.plot(max_samples_values, means, marker="o", color="#c0392b", label="Mean CV accuracy")
    plt.fill_between(max_samples_values, means - stds, means + stds, color="#c0392b", alpha=0.15, label="±1 std")
    plt.xlabel("max_samples")
    plt.ylabel("Cross-validation accuracy")
    plt.title("Effect of max_samples on Bagging Performance")
    plt.ylim(0.70, 1.02)
    plt.xticks(max_samples_values)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "bagging_max_samples_curve.png"
    plt.savefig(output_path, dpi=200)
    return output_path, max_samples_values, means, stds


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUTPUT_DIR.mkdir(exist_ok=True)

    wine = load_wine()
    x, y = wine.data, wine.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    bagging = make_bagging_classifier(max_samples=1.0, n_estimators=100)

    tree.fit(x_train, y_train)
    bagging.fit(x_train, y_train)

    tree_pred = tree.predict(x_test)
    bagging_pred = bagging.predict(x_test)

    tree_acc = accuracy_score(y_test, tree_pred)
    bagging_acc = accuracy_score(y_test, bagging_pred)

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    tree_cv_scores = cross_val_score(tree, x, y, cv=cv, scoring="accuracy", n_jobs=-1)
    bagging_cv_scores = cross_val_score(bagging, x, y, cv=cv, scoring="accuracy", n_jobs=-1)

    print("Wine dataset:")
    print(f"  samples = {x.shape[0]}, features = {x.shape[1]}, classes = {len(np.unique(y))}")
    print(f"  train/test split = {x_train.shape[0]}/{x_test.shape[0]}")

    print("\nTest-set accuracy:")
    print(f"  Single decision tree: {tree_acc:.4f}")
    print(f"  Bagging classifier  : {bagging_acc:.4f}")

    print(f"\n{CV_SPLITS}-fold cross-validation stability:")
    print_cv_result("Decision tree", tree_cv_scores)
    print_cv_result("Bagging", bagging_cv_scores)

    learning_curve_path = plot_learning_curve(bagging, x, y, cv)
    max_samples_path, sample_values, sample_means, sample_stds = plot_max_samples_curve(x, y, cv)

    best_index = int(np.argmax(sample_means))
    print("\nmax_samples sensitivity:")
    for value, mean, std in zip(sample_values, sample_means, sample_stds):
        print(f"  max_samples={value:.1f}: mean accuracy={mean:.4f}, std={std:.4f}")

    print(
        f"\nBest max_samples in this experiment: {sample_values[best_index]:.1f} "
        f"(mean accuracy={sample_means[best_index]:.4f})"
    )
    print("\nFigures saved to:")
    print(f"  {learning_curve_path}")
    print(f"  {max_samples_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
