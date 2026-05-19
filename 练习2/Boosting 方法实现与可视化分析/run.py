from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def make_gbdt(learning_rate=0.1, max_depth=3, n_estimators=150):
    return GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
    )


def staged_accuracy(model, x, y):
    return np.array([accuracy_score(y, pred) for pred in model.staged_predict(x)])


def staged_loss(model, x, y, labels):
    return np.array([log_loss(y, proba, labels=labels) for proba in model.staged_predict_proba(x)])


def diagnose_fit(train_acc, test_acc):
    gap = train_acc - test_acc
    if train_acc < 0.9 and test_acc < 0.9:
        return "possible underfitting"
    if gap > 0.08:
        return "possible overfitting"
    return "stable"


def plot_loss_curve(model, x_test, y_test, labels):
    train_loss = model.train_score_
    test_loss = staged_loss(model, x_test, y_test, labels)
    rounds = np.arange(1, len(train_loss) + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, train_loss, label="Training loss")
    plt.plot(rounds, test_loss, label="Test loss")
    plt.xlabel("n_estimators")
    plt.ylabel("Log loss")
    plt.title("GBDT Loss Curve on Wine")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "gbdt_loss_curve.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def plot_error_curve(model, x_train, y_train, x_test, y_test):
    train_error = 1.0 - staged_accuracy(model, x_train, y_train)
    test_error = 1.0 - staged_accuracy(model, x_test, y_test)
    rounds = np.arange(1, len(train_error) + 1)

    best_index = int(np.argmin(test_error))

    plt.figure(figsize=(9, 5))
    plt.plot(rounds, train_error, label="Training error")
    plt.plot(rounds, test_error, label="Test error")
    plt.scatter(rounds[best_index], test_error[best_index], color="#c0392b", zorder=3, label="Best test error")
    plt.xlabel("n_estimators")
    plt.ylabel("Error rate")
    plt.title("GBDT Error Rate vs Iterations")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "gbdt_error_curve.png"
    plt.savefig(output_path, dpi=200)
    return output_path, rounds[best_index], test_error[best_index]


def evaluate_learning_rate_and_depth(x_train, y_train, x_test, y_test):
    learning_rates = [0.01, 0.1, 0.3]
    max_depths = [1, 3, 5]
    results = []
    test_matrix = np.zeros((len(max_depths), len(learning_rates)))

    for row, max_depth in enumerate(max_depths):
        for col, learning_rate in enumerate(learning_rates):
            model = make_gbdt(learning_rate=learning_rate, max_depth=max_depth, n_estimators=150)
            model.fit(x_train, y_train)

            train_acc = model.score(x_train, y_train)
            test_acc = model.score(x_test, y_test)
            best_test_error = 1.0 - staged_accuracy(model, x_test, y_test).max()
            status = diagnose_fit(train_acc, test_acc)

            results.append(
                {
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "best_test_error": best_test_error,
                    "status": status,
                }
            )
            test_matrix[row, col] = test_acc

    return learning_rates, max_depths, results, test_matrix


def plot_hyperparameter_heatmap(learning_rates, max_depths, test_matrix):
    plt.figure(figsize=(7, 5))
    image = plt.imshow(test_matrix, cmap="YlGnBu", vmin=0.80, vmax=1.00)
    plt.colorbar(image, label="Test accuracy")
    plt.xticks(np.arange(len(learning_rates)), [str(v) for v in learning_rates])
    plt.yticks(np.arange(len(max_depths)), [str(v) for v in max_depths])
    plt.xlabel("learning_rate")
    plt.ylabel("max_depth")
    plt.title("GBDT Test Accuracy under Different Settings")

    for row in range(len(max_depths)):
        for col in range(len(learning_rates)):
            plt.text(col, row, f"{test_matrix[row, col]:.3f}", ha="center", va="center", color="black")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "gbdt_hyperparameter_heatmap.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    order = np.argsort(importances)

    plt.figure(figsize=(9, 6))
    plt.barh(np.array(feature_names)[order], importances[order], color="#2c7fb8")
    plt.xlabel("Importance")
    plt.title("GBDT Feature Importance on Wine")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "gbdt_feature_importance.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    wine = load_wine()
    x, y = wine.data, wine.target
    labels = np.unique(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_model = make_gbdt(learning_rate=0.1, max_depth=3, n_estimators=150)
    base_model.fit(x_train, y_train)

    train_acc = base_model.score(x_train, y_train)
    test_acc = base_model.score(x_test, y_test)
    test_loss = log_loss(y_test, base_model.predict_proba(x_test), labels=labels)

    print("Wine dataset:")
    print(f"  samples = {x.shape[0]}, features = {x.shape[1]}, classes = {len(labels)}")
    print(f"  train/test split = {x_train.shape[0]}/{x_test.shape[0]}")

    print("\nBase GBDT model:")
    print("  learning_rate = 0.1, max_depth = 3, n_estimators = 150")
    print(f"  training accuracy = {train_acc:.4f}")
    print(f"  test accuracy     = {test_acc:.4f}")
    print(f"  final test loss   = {test_loss:.4f}")

    learning_rates, max_depths, results, test_matrix = evaluate_learning_rate_and_depth(x_train, y_train, x_test, y_test)

    print("\nLearning-rate and max-depth comparison:")
    print("  lr    depth  train_acc  test_acc  best_test_error  diagnosis")
    for item in results:
        print(
            f"  {item['learning_rate']:<5} {item['max_depth']:<5} "
            f"{item['train_acc']:.4f}     {item['test_acc']:.4f}    "
            f"{item['best_test_error']:.4f}           {item['status']}"
        )

    best_result = max(results, key=lambda item: item["test_acc"])
    print(
        "\nBest setting by held-out test accuracy: "
        f"learning_rate={best_result['learning_rate']}, max_depth={best_result['max_depth']}, "
        f"test_acc={best_result['test_acc']:.4f}"
    )

    loss_path = plot_loss_curve(base_model, x_test, y_test, labels)
    error_path, best_round, best_error = plot_error_curve(base_model, x_train, y_train, x_test, y_test)
    heatmap_path = plot_hyperparameter_heatmap(learning_rates, max_depths, test_matrix)
    feature_path = plot_feature_importance(base_model, wine.feature_names)

    print(f"\nBest iteration in base model: n_estimators={best_round}, test_error={best_error:.4f}")
    print("\nFigures saved to:")
    print(f"  {loss_path}")
    print(f"  {error_path}")
    print(f"  {heatmap_path}")
    print(f"  {feature_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
