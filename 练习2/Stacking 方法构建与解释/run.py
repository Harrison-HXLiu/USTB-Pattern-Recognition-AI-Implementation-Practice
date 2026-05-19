from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
CV_SPLITS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def build_base_models():
    return [
        (
            "svm",
            make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=2.0, gamma="scale", probability=True, random_state=RANDOM_STATE),
            ),
        ),
        (
            "random_forest",
            RandomForestClassifier(n_estimators=200, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        ),
        (
            "gbdt",
            GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
        ),
    ]


def make_meta_model():
    return LogisticRegression(max_iter=2000, multi_class="auto", random_state=RANDOM_STATE)


def make_stacking_model(base_models):
    return StackingClassifier(
        estimators=base_models,
        final_estimator=make_meta_model(),
        stack_method="predict_proba",
        cv=StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )


def make_voting_model(base_models):
    return VotingClassifier(estimators=base_models, voting="soft", n_jobs=-1)


def probability_feature_names(model_names, class_names):
    return [f"{model}_P({class_name})" for model in model_names for class_name in class_names]


def build_oof_meta_features(base_models, x_train, y_train, cv):
    meta_parts = []
    fitted_models = []

    for name, model in base_models:
        oof_proba = cross_val_predict(
            clone(model),
            x_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )
        meta_parts.append(oof_proba)

        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train)
        fitted_models.append((name, fitted_model))

    return np.hstack(meta_parts), fitted_models


def build_test_meta_features(fitted_models, x_test):
    return np.hstack([model.predict_proba(x_test) for _, model in fitted_models])


def evaluate_models(models, x_train, y_train, x_test, y_test, cv):
    rows = []

    for name, model in models:
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        pred = fitted.predict(x_test)
        proba = fitted.predict_proba(x_test)
        cv_scores = cross_val_score(clone(model), np.vstack([x_train, x_test]), np.hstack([y_train, y_test]), cv=cv, scoring="accuracy", n_jobs=-1)

        rows.append(
            {
                "name": name,
                "test_acc": accuracy_score(y_test, pred),
                "test_loss": log_loss(y_test, proba, labels=np.unique(y_train)),
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }
        )

    return rows


def plot_base_probability_correlation(fitted_models, x_test, class_names):
    model_names = [name for name, _ in fitted_models]
    feature_names = probability_feature_names(model_names, class_names)
    probability_matrix = build_test_meta_features(fitted_models, x_test)
    correlation = np.corrcoef(probability_matrix, rowvar=False)

    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    plt.colorbar(image, label="Correlation")
    plt.xticks(np.arange(len(feature_names)), feature_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(feature_names)), feature_names)
    plt.title("Correlation of Base Model Probability Outputs")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "stacking_base_probability_correlation.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def plot_meta_coefficients(meta_model, feature_names, class_names):
    coefficients = meta_model.coef_

    plt.figure(figsize=(11, 6))
    x_positions = np.arange(len(feature_names))
    width = 0.24

    for class_index, class_name in enumerate(class_names):
        offset = (class_index - (len(class_names) - 1) / 2) * width
        plt.bar(x_positions + offset, coefficients[class_index], width=width, label=str(class_name))

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x_positions, feature_names, rotation=45, ha="right")
    plt.ylabel("Logistic regression coefficient")
    plt.title("Meta Learner Coefficients for Stacking")
    plt.legend(title="Class")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "stacking_meta_coefficients.png"
    plt.savefig(output_path, dpi=200)
    return output_path


def print_top_meta_coefficients(meta_model, feature_names, class_names, top_k=5):
    print("\nMeta-model coefficient interpretation:")
    for class_index, class_name in enumerate(class_names):
        coefs = meta_model.coef_[class_index]
        top_indices = np.argsort(np.abs(coefs))[::-1][:top_k]
        print(f"  Class {class_name}:")
        for index in top_indices:
            direction = "supports" if coefs[index] > 0 else "suppresses"
            print(f"    {feature_names[index]:<28} {direction:<9} class {class_name}  coef={coefs[index]: .4f}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    wine = load_wine()
    x, y = wine.data, wine.target
    class_names = wine.target_names

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.3,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    base_models = build_base_models()
    stacking_model = make_stacking_model(base_models)
    voting_model = make_voting_model(base_models)

    comparison_models = base_models + [
        ("soft_voting", voting_model),
        ("stacking", stacking_model),
    ]
    comparison_rows = evaluate_models(comparison_models, x_train, y_train, x_test, y_test, cv)

    oof_meta_features, fitted_base_models = build_oof_meta_features(base_models, x_train, y_train, cv)
    test_meta_features = build_test_meta_features(fitted_base_models, x_test)
    meta_model = make_meta_model()
    meta_model.fit(oof_meta_features, y_train)
    manual_meta_pred = meta_model.predict(test_meta_features)
    manual_meta_proba = meta_model.predict_proba(test_meta_features)

    stacking_model.fit(x_train, y_train)
    stacking_pred = stacking_model.predict(x_test)

    model_names = [name for name, _ in base_models]
    feature_names = probability_feature_names(model_names, class_names)

    print("Wine dataset:")
    print(f"  samples = {x.shape[0]}, features = {x.shape[1]}, classes = {len(class_names)}")
    print(f"  train/test split = {x_train.shape[0]}/{x_test.shape[0]}")

    print("\nBase model probability outputs on the first 3 test samples:")
    for name, model in fitted_base_models:
        probabilities = model.predict_proba(x_test[:3])
        print(f"  {name}:")
        for row in probabilities:
            print("    " + "  ".join(f"{class_name}={value:.3f}" for class_name, value in zip(class_names, row)))

    print("\nModel comparison:")
    print("  model          test_acc  test_loss  cv_mean  cv_std")
    for row in comparison_rows:
        print(
            f"  {row['name']:<14} {row['test_acc']:.4f}    {row['test_loss']:.4f}    "
            f"{row['cv_mean']:.4f}   {row['cv_std']:.4f}"
        )

    print("\nManual 5-fold out-of-fold meta learner:")
    print(f"  meta feature shape = {oof_meta_features.shape}")
    print(f"  test accuracy      = {accuracy_score(y_test, manual_meta_pred):.4f}")
    print(f"  test log loss      = {log_loss(y_test, manual_meta_proba, labels=np.unique(y)):.4f}")

    print("\nStackingClassifier check:")
    print(f"  test accuracy = {accuracy_score(y_test, stacking_pred):.4f}")

    corr_path = plot_base_probability_correlation(fitted_base_models, x_test, class_names)
    coef_path = plot_meta_coefficients(meta_model, feature_names, class_names)
    print_top_meta_coefficients(meta_model, feature_names, class_names)

    print("\nFigures saved to:")
    print(f"  {corr_path}")
    print(f"  {coef_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
