
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sqlalchemy import create_engine, text

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier


ENGINE = create_engine("postgresql+psycopg:///retailiq")

OBSERVATION_DAYS = 180
PREDICTION_DAYS = 90
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "RetailIQ - Customer Churn"


def load_transactions():
    query = """
        SELECT invoice_date, customer_id, invoice,
               stock_code, quantity, price, country
        FROM retail.transaction_classified
        WHERE included_in_merchandise_sales = TRUE
          AND customer_id IS NOT NULL
          AND quantity > 0
          AND price > 0
    """

    df = pd.read_sql(text(query), ENGINE)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = df["customer_id"].astype(str)
    df["revenue"] = df["quantity"] * df["price"]

    return df


def build_dataset(df, cutoff):
    history_start = cutoff - pd.Timedelta(days=OBSERVATION_DAYS)
    prediction_end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)

    history = df[
        (df["invoice_date"] >= history_start)
        & (df["invoice_date"] < cutoff)
    ]

    future = df[
        (df["invoice_date"] >= cutoff)
        & (df["invoice_date"] < prediction_end)
    ]

    features = history.groupby("customer_id").agg(
        recency=("invoice_date", lambda x: (cutoff - x.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("revenue", "sum"),
        quantity=("quantity", "sum"),
        product_diversity=("stock_code", "nunique"),
        country=("country", "last"),
    )

    features["average_order_value"] = (
        features["monetary"] / features["frequency"]
    )

    future_customers = set(future["customer_id"].unique())

    features["target"] = (
        ~features.index.isin(future_customers)
    ).astype(int)

    return features


def evaluate(train, test):
    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "quantity",
        "product_diversity",
        "average_order_value",
        "country",
    ]

    X_train = train[feature_columns]
    y_train = train["target"]

    X_test = test[feature_columns]
    y_test = test["target"]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError("Both periods must contain both outcome classes.")

    if y_train.value_counts().min() < 5:
        raise ValueError("Insufficient training examples for 5-fold CV.")

    numeric_features = [
        "recency",
        "frequency",
        "monetary",
        "quantity",
        "product_diversity",
        "average_order_value",
    ]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), ["country"]),
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    # --------------------------------------------------------
    # MODEL COMPARISON AND MLFLOW LOGGING
    # --------------------------------------------------------

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
        }

        results.append({
            "model": name,
            **metrics,
        })

        model_key = name.lower().replace(" ", "_")

        mlflow.log_metrics({
            f"{model_key}_{metric}": value
            for metric, value in metrics.items()
        })

        mlflow.log_param(
            f"{model_key}_classifier",
            name,
        )

        if name == "Logistic Regression":
            trusted_types = ["numpy.dtype"]

        elif name == "Random Forest":
            trusted_types = [
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ]

        else:
            trusted_types = [
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ]

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=f"{model_key}_model",
            skops_trusted_types=trusted_types,
        )

    results_df = pd.DataFrame(results)

    print("\nCUSTOMER INACTIVITY MODEL COMPARISON")
    print(results_df.to_string(index=False))

    mlflow.log_table(
        results_df,
        artifact_file="model_comparison.json",
    )

    # --------------------------------------------------------
    # PROBABILITY CALIBRATION
    # --------------------------------------------------------

    print("\nPROBABILITY CALIBRATION — LOGISTIC REGRESSION")

    base_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(models["Logistic Regression"])),
    ])

    base_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", clone(models["Logistic Regression"])),
        ]),
        method="sigmoid",
        cv=3,
    )

    calibrated_model.fit(X_train, y_train)

    uncalibrated_probabilities = base_model.predict_proba(X_test)[:, 1]
    calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]

    uncalibrated_brier = np.mean(
        (uncalibrated_probabilities - y_test) ** 2
    )

    calibrated_brier = np.mean(
        (calibrated_probabilities - y_test) ** 2
    )

    print(
        "Uncalibrated Brier score:",
        round(uncalibrated_brier, 5),
    )

    print(
        "Calibrated Brier score:",
        round(calibrated_brier, 5),
    )

    mlflow.log_metrics({
        "uncalibrated_brier_score": float(uncalibrated_brier),
        "calibrated_brier_score": float(calibrated_brier),
    })

    mlflow.sklearn.log_model(
        sk_model=calibrated_model,
        name="calibrated_churn_model",
        skops_trusted_types=[
            "numpy.dtype",
            "sklearn.calibration._CalibratedClassifier",
            "sklearn.calibration._SigmoidCalibration",
        ],
    )

    # --------------------------------------------------------
    # BUSINESS THRESHOLD SELECTION
    # --------------------------------------------------------

    print("\nBUSINESS THRESHOLD SELECTION — LOGISTIC REGRESSION")

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    training_probabilities = cross_val_predict(
        base_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    threshold_results = []

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (
            training_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
            "threshold": threshold,
            "f1": f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
        })

    best_result = max(
        threshold_results,
        key=lambda result: result["f1"],
    )

    best_threshold = best_result["threshold"]

    print(f"Selected training threshold: {best_threshold:.2f}")

    final_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", clone(models["Logistic Regression"])),
    ])

    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    final_metrics = {
        "selected_threshold": float(best_threshold),
        "threshold_training_f1": float(best_result["f1"]),
        "threshold_test_precision": float(
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "threshold_test_recall": float(
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
        "threshold_test_f1": float(
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            )
        ),
    }

    print(
        "Test precision:",
        round(final_metrics["threshold_test_precision"], 4),
    )

    print(
        "Test recall:",
        round(final_metrics["threshold_test_recall"], 4),
    )

    print(
        "Test F1:",
        round(final_metrics["threshold_test_f1"], 4),
    )

    mlflow.log_metrics(final_metrics)

    mlflow.sklearn.log_model(
        sk_model=final_model,
        name="selected_threshold_churn_model",
        skops_trusted_types=["numpy.dtype"],
    )

    # --------------------------------------------------------
    # PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nINACTIVITY MODEL — PERMUTATION FEATURE IMPORTANCE")

    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values(
        "importance_mean",
        ascending=False,
    )

    print(importance_df.to_string(index=False))

    mlflow.log_table(
        importance_df,
        artifact_file="permutation_feature_importance.json",
    )

    # --------------------------------------------------------
    # DATASET AND EXPERIMENT METADATA
    # --------------------------------------------------------

    mlflow.log_params({
        "observation_days": OBSERVATION_DAYS,
        "prediction_days": PREDICTION_DAYS,
        "random_state": RANDOM_STATE,
        "training_customers": len(X_train),
        "testing_customers": len(X_test),
        "training_inactivity_rate": float(y_train.mean()),
        "testing_inactivity_rate": float(y_test.mean()),
        "feature_count": len(feature_columns),
        "threshold_selection_method": "5-fold stratified cross-validation",
        "calibration_method": "sigmoid",
    })

    print(f"\nTraining customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")
    print(f"Training inactivity rate: {y_train.mean():.2%}")
    print(f"Testing inactivity rate: {y_test.mean():.2%}")


def main():
    transactions = load_transactions()

    max_date = transactions["invoice_date"].max().normalize()

    cutoffs = [
        max_date - pd.Timedelta(days=180),
        max_date - pd.Timedelta(days=90),
    ]

    train = build_dataset(transactions, cutoffs[0])
    test = build_dataset(transactions, cutoffs[1])

    print(f"Training cutoff: {cutoffs[0].date()}")
    print(f"Testing cutoff: {cutoffs[1].date()}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="Customer Churn Model Evaluation"):
        mlflow.log_params({
            "observation_days": OBSERVATION_DAYS,
            "prediction_days": PREDICTION_DAYS,
            "training_cutoff": str(cutoffs[0].date()),
            "testing_cutoff": str(cutoffs[1].date()),
        })

        evaluate(train, test)


if __name__ == "__main__":
    main()
