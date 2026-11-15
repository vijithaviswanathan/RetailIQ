
import pandas as pd
import numpy as np
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
from xgboost import XGBClassifier
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

ENGINE = create_engine("postgresql+psycopg:///retailiq")

OBSERVATION_DAYS = 180
PREDICTION_DAYS = 30
RANDOM_STATE = 42

MLFLOW_EXPERIMENT = "RetailIQ - Purchase Propensity"


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


def build_customer_dataset(df, cutoff):
    history_start = cutoff - pd.Timedelta(days=OBSERVATION_DAYS)
    prediction_end = cutoff + pd.Timedelta(days=PREDICTION_DAYS)

    history = df[
        (df["invoice_date"] >= history_start)
        & (df["invoice_date"] < cutoff)
    ].copy()

    future = df[
        (df["invoice_date"] >= cutoff)
        & (df["invoice_date"] < prediction_end)
    ].copy()

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

    labels = (
        future.groupby("customer_id")["invoice"]
        .nunique()
        .gt(0)
        .astype(int)
        .rename("target")
    )

    dataset = features.join(labels, how="left")
    dataset["target"] = dataset["target"].fillna(0).astype(int)

    return dataset


def train_and_evaluate(dataset):
    X = dataset.drop(columns="target")
    y = dataset["target"]

    split = int(len(dataset) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Insufficient class variation. Use a different cutoff date."
        )

    numeric_features = [
        "recency", "frequency", "monetary",
        "quantity", "product_diversity", "average_order_value",
    ]

    categorical_features = ["country"]

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
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
    }

    results = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ])

        pipeline.fit(X_train, y_train)

        probabilities = pipeline.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        results.append({
            "model": name,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(y_test, predictions, zero_division=0),
        })

    print("\nPURCHASE PROPENSITY RESULTS")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nTraining customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")
    print(f"Training purchase rate: {y_train.mean():.2%}")
    print(f"Testing purchase rate: {y_test.mean():.2%}")


def main():
    transactions = load_transactions()

    max_date = transactions["invoice_date"].max().normalize()

    # Two historical snapshots, each with 180 days of
    # observation and 30 days of future purchase outcomes.
    cutoffs = [
        max_date - pd.Timedelta(days=90),
        max_date - pd.Timedelta(days=30),
    ]

    datasets = []

    for cutoff in cutoffs:
        snapshot = build_customer_dataset(transactions, cutoff)
        snapshot["snapshot_date"] = cutoff
        datasets.append(snapshot.reset_index())

    combined = pd.concat(datasets, ignore_index=True)
    combined = combined.sort_values("snapshot_date")

    train = combined[
        combined["snapshot_date"] == cutoffs[0]
    ].copy()

    test = combined[
        combined["snapshot_date"] == cutoffs[1]
    ].copy()

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
        raise ValueError(
            "Both periods must contain positive and negative outcomes."
        )

    dataset = pd.concat([
        X_train.assign(target=y_train),
        X_test.assign(target=y_test),
    ], ignore_index=True)

    dataset.attrs["train_size"] = len(train)

    train_and_evaluate_temporal(dataset)


def train_and_evaluate_temporal(dataset):

    X = dataset.drop(columns="target")
    y = dataset["target"]

    split = dataset.attrs["train_size"]

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    numeric_features = [
        "recency", "frequency", "monetary",
        "quantity", "product_diversity", "average_order_value",
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

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = []

    # --------------------------------------------------------
    # MODEL COMPARISON WITH MLFLOW
    # --------------------------------------------------------

    for name, classifier in models.items():

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", classifier),
        ])

        with mlflow.start_run(run_name=name):

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

            mlflow.log_param("model", name)
            mlflow.log_param("split_strategy", "chronological")
            mlflow.log_param("training_customers", len(X_train))
            mlflow.log_param("testing_customers", len(X_test))
            mlflow.log_param("classification_threshold", 0.5)

            mlflow.log_params({
                key: str(value)
                for key, value in classifier.get_params(
                    deep=False
                ).items()
            })

            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="propensity_model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

            results.append({
                "model": name,
                **metrics,
            })

    # --------------------------------------------------------
    # BUSINESS THRESHOLD SELECTION
    # --------------------------------------------------------

    print("\nBUSINESS THRESHOLD SELECTION — LOGISTIC REGRESSION")

    threshold_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    training_probabilities = cross_val_predict(
        threshold_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    thresholds = np.arange(0.10, 0.91, 0.01)

    threshold_results = []

    for threshold in thresholds:
        predictions = (training_probabilities >= threshold).astype(int)

        threshold_results.append({
            "threshold": threshold,
            "precision": precision_score(
                y_train, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_train, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_train, predictions, zero_division=0
            ),
        })

    best_threshold = max(
        threshold_results,
        key=lambda result: result["f1"],
    )["threshold"]

    print(f"Selected training threshold: {best_threshold:.2f}")

    # --------------------------------------------------------
    # FINAL SELECTED-THRESHOLD MODEL
    # --------------------------------------------------------

    final_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    final_model.fit(X_train, y_train)

    test_probabilities = final_model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    final_metrics = {
        "roc_auc": roc_auc_score(y_test, test_probabilities),
        "pr_auc": average_precision_score(y_test, test_probabilities),
        "precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, test_predictions, zero_division=0
        ),
    }

    print("Test precision:", round(final_metrics["precision"], 4))
    print("Test recall:", round(final_metrics["recall"], 4))
    print("Test F1:", round(final_metrics["f1"], 4))

    with mlflow.start_run(
        run_name="Logistic Regression - Selected Threshold"
    ):

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("selected_threshold", float(best_threshold))
        mlflow.log_param("threshold_selection_metric", "training_f1")
        mlflow.log_param("training_customers", len(X_train))
        mlflow.log_param("testing_customers", len(X_test))
        mlflow.log_param("split_strategy", "chronological")

        mlflow.log_metrics(final_metrics)

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="selected_threshold_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )

    # --------------------------------------------------------
    # PERMUTATION FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\nPROPENSITY MODEL — PERMUTATION FEATURE IMPORTANCE")

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
    }).sort_values("importance_mean", ascending=False)

    print(importance_df.to_string(index=False))

    # --------------------------------------------------------
    # PROBABILITY CALIBRATION
    # --------------------------------------------------------

    base_model = Pipeline([
        ("preprocessor", clone(preprocessor)),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    base_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
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

    with mlflow.start_run(
        run_name="Logistic Regression - Probability Calibration"
    ):

        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("calibration_method", "sigmoid")
        mlflow.log_param("calibration_cv", 3)
        mlflow.log_param("training_customers", len(X_train))
        mlflow.log_param("testing_customers", len(X_test))

        mlflow.log_metric(
            "uncalibrated_brier_score",
            float(uncalibrated_brier),
        )

        mlflow.log_metric(
            "calibrated_brier_score",
            float(calibrated_brier),
        )

        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            name="calibrated_propensity_model",
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration",
            ],
        )

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\nCHRONOLOGICAL PROPENSITY EVALUATION")
    print(pd.DataFrame(results).to_string(index=False))

    print(f"\nTraining customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")
    print(f"Training purchase rate: {y_train.mean():.2%}")
    print(f"Testing purchase rate: {y_test.mean():.2%}")


if __name__ == "__main__":
    main()
