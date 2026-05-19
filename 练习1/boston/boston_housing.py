from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_data(csv_filename: str) -> pd.DataFrame:
    data_path = Path(__file__).resolve().parent / csv_filename
    return pd.read_csv(data_path)


def build_and_evaluate_model(data: pd.DataFrame) -> None:
    X = data.drop(columns=["MEDV"])
    y = data["MEDV"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Linear Regression Model Evaluation")
    print("--------------------------------")
    print(f"Test set mean squared error (MSE): {mse:.4f}")
    print(f"Test set coefficient of determination (R^2): {r2:.4f}")
    print()

    print("Model coefficients:")
    for feature, coef in zip(X.columns, model.coef_):
        print(f"  {feature}: {coef:.4f}")
    print(f"Intercept: {model.intercept_:.4f}")

    plot_true_vs_predicted(y_test.to_numpy(), y_pred)


def plot_true_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", label="Predicted")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect fit y=x")
    plt.xlabel("Actual MEDV")
    plt.ylabel("Predicted MEDV")
    plt.title("Actual vs Predicted: Boston Housing Linear Regression")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file = "boston(1).csv"
    df = load_data(csv_file)
    build_and_evaluate_model(df)
