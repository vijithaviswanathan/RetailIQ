
import numpy as np
import pandas as pd
import mlflow

from sqlalchemy import create_engine, text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from retailiq.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

MLFLOW_EXPERIMENT = "RetailIQ - Uplift Modeling"

RANDOM_STATE = 42
TOP_K_FRACTION = 0.30

NUMERIC_FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
]

CATEGORICAL_FEATURES = [
    "history_segment",
    "zip_code",
    "channel",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TREATMENT_COMPARISONS = [
    ("Mens E-Mail", "No E-Mail"),
    ("Womens E-Mail", "No E-Mail"),
]


def load_data():
    query = """
        SELECT
            recency,
            history_segment,
            history,
            mens,
            womens,
            zip_code,
            newbie,
            channel,
            segment,
            conversion
        FROM hillstrom.campaign_data
    """

    with engine.connect() as connection:
        return pd.read_sql(text(query), connection)


def build_model(include_treatment=False):
    numeric_features = NUMERIC_FEATURES.copy()

    if include_treatment:
        numeric_features.append("treatment")

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier),
    ])


def build_effect_model():
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=50,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def qini_area(uplift, treatment, outcome):
    results = pd.DataFrame({
        "uplift": uplift,
        "treatment": treatment,
        "outcome": outcome,
    }).sort_values("uplift", ascending=False)

    cumulative_treatment = (
        results["treatment"] == 1
    ).cumsum().to_numpy()

    cumulative_control = (
        results["treatment"] == 0
    ).cumsum().to_numpy()

    treatment_responders = (
        results["outcome"] * results["treatment"]
    ).cumsum().to_numpy()

    control_responders = (
        results["outcome"] * (1 - results["treatment"])
    ).cumsum().to_numpy()

    curve = np.zeros(len(results))

    valid = cumulative_control > 0

    curve[valid] = (
        treatment_responders[valid]
        - control_responders[valid]
        * cumulative_treatment[valid]
        / cumulative_control[valid]
    )

    population_fraction = (
        np.arange(1, len(results) + 1) / len(results)
    )

    return float(np.trapezoid(
        curve,
        population_fraction,
    ))


def uplift_at_k(
    uplift,
    treatment,
    outcome,
    fraction=TOP_K_FRACTION,
):
    results = pd.DataFrame({
        "uplift": uplift,
        "treatment": treatment,
        "outcome": outcome,
    }).sort_values("uplift", ascending=False)

    selected = results.head(
        max(1, int(len(results) * fraction))
    )

    treated = selected[selected["treatment"] == 1]
    control = selected[selected["treatment"] == 0]

    if treated.empty or control.empty:
        return np.nan

    return float(
        treated["outcome"].mean()
        - control["outcome"].mean()
    )


def evaluate_model(name, uplift, treatment, outcome):
    return {
        "model": name,
        "qini_area": qini_area(
            uplift,
            treatment,
            outcome,
        ),
        "uplift_top_30_percent": uplift_at_k(
            uplift,
            treatment,
            outcome,
        ),
        "average_predicted_uplift": float(
            np.mean(uplift)
        ),
    }


def log_model_run(
    model_name,
    treatment_name,
    control_name,
    metrics,
    uplift_predictions,
    treatment_test,
    y_test,
):
    run_name = (
        f"{model_name} - "
        f"{treatment_name} vs {control_name}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "project": "RetailIQ",
            "task": "causal_uplift_modeling",
            "model": model_name,
            "treatment": treatment_name,
            "control": control_name,
            "dataset": "Hillstrom Email Marketing",
            "validation": "stratified_70_30_holdout",
        })

        mlflow.log_params({
            "n_estimators": 300,
            "min_samples_leaf": 50,
            "random_state": RANDOM_STATE,
            "test_size": 0.30,
            "top_k_fraction": TOP_K_FRACTION,
        })

        for metric_name, metric_value in metrics.items():
            if metric_name == "model":
                continue

            if pd.notna(metric_value):
                mlflow.log_metric(
                    metric_name,
                    float(metric_value),
                )

        predictions = pd.DataFrame({
            "predicted_uplift": uplift_predictions,
            "treatment": treatment_test,
            "observed_conversion": y_test,
        })

        mlflow.log_table(
            predictions,
            "heldout_uplift_predictions.json",
        )


def run_experiment(df, treatment_name, control_name):
    experiment = df[
        df["segment"].isin([
            treatment_name,
            control_name,
        ])
    ].copy()

    experiment["treatment"] = (
        experiment["segment"] == treatment_name
    ).astype(int)

    X = experiment[FEATURES].copy()
    y = experiment["conversion"].astype(int).to_numpy()
    treatment = experiment["treatment"].to_numpy()

    strata = (
        treatment.astype(str) + "_" + y.astype(str)
    )

    train_idx, test_idx = train_test_split(
        np.arange(len(experiment)),
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y[train_idx]
    y_test = y[test_idx]

    treatment_train = treatment[train_idx]
    treatment_test = treatment[test_idx]

    # --------------------------------------------------------
    # S-LEARNER
    # --------------------------------------------------------

    X_train_s = X_train.copy()
    X_train_s["treatment"] = treatment_train

    s_model = build_model(include_treatment=True)
    s_model.fit(X_train_s, y_train)

    X_test_control = X_test.copy()
    X_test_control["treatment"] = 0

    X_test_treatment = X_test.copy()
    X_test_treatment["treatment"] = 1

    s_control = s_model.predict_proba(
        X_test_control
    )[:, 1]

    s_treatment = s_model.predict_proba(
        X_test_treatment
    )[:, 1]

    s_uplift = s_treatment - s_control

    # --------------------------------------------------------
    # T-LEARNER
    # --------------------------------------------------------

    control_mask = treatment_train == 0
    treatment_mask = treatment_train == 1

    control_positions = np.where(control_mask)[0]
    treatment_positions = np.where(treatment_mask)[0]

    control_model = build_model()
    treatment_model = build_model()

    control_model.fit(
        X_train.iloc[control_positions],
        y_train[control_mask],
    )

    treatment_model.fit(
        X_train.iloc[treatment_positions],
        y_train[treatment_mask],
    )

    t_control = control_model.predict_proba(
        X_test
    )[:, 1]

    t_treatment = treatment_model.predict_proba(
        X_test
    )[:, 1]

    t_uplift = t_treatment - t_control

    # --------------------------------------------------------
    # X-LEARNER
    # --------------------------------------------------------

    control_model_x = build_model()
    treatment_model_x = build_model()

    control_model_x.fit(
        X_train.iloc[control_positions],
        y_train[control_mask],
    )

    treatment_model_x.fit(
        X_train.iloc[treatment_positions],
        y_train[treatment_mask],
    )

    control_effects = (
        treatment_model_x.predict_proba(
            X_train.iloc[control_positions]
        )[:, 1]
        - y_train[control_mask]
    )

    treatment_effects = (
        y_train[treatment_mask]
        - control_model_x.predict_proba(
            X_train.iloc[treatment_positions]
        )[:, 1]
    )

    control_effect_model = build_effect_model()
    treatment_effect_model = build_effect_model()

    control_effect_model.fit(
        X_train.iloc[control_positions],
        control_effects,
    )

    treatment_effect_model.fit(
        X_train.iloc[treatment_positions],
        treatment_effects,
    )

    propensity = treatment_train.mean()

    x_uplift = (
        propensity * control_effect_model.predict(X_test)
        + (1 - propensity) * treatment_effect_model.predict(X_test)
    )

    # --------------------------------------------------------
    # EVALUATION AND MLFLOW TRACKING
    # --------------------------------------------------------

    model_predictions = {
        "S-Learner": s_uplift,
        "T-Learner": t_uplift,
        "X-Learner": x_uplift,
    }

    result_rows = []

    for model_name, uplift_predictions in model_predictions.items():
        metrics = evaluate_model(
            model_name,
            uplift_predictions,
            treatment_test,
            y_test,
        )

        result_rows.append(metrics)

        log_model_run(
            model_name=model_name,
            treatment_name=treatment_name,
            control_name=control_name,
            metrics=metrics,
            uplift_predictions=uplift_predictions,
            treatment_test=treatment_test,
            y_test=y_test,
        )

    results = pd.DataFrame(result_rows)

    print(f"\n=== {treatment_name} vs {control_name} ===")
    print(results.round(6).to_string(index=False))

    return results


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_data()

    if df.empty:
        raise ValueError("Hillstrom dataset is empty.")

    if df[FEATURES + ["segment", "conversion"]].isna().any().any():
        raise ValueError("Missing values found in model data.")

    if not df["conversion"].isin([0, 1]).all():
        raise ValueError("Conversion must be binary.")

    all_results = []

    for treatment_name, control_name in TREATMENT_COMPARISONS:
        results = run_experiment(
            df,
            treatment_name,
            control_name,
        )

        results["treatment_comparison"] = (
            f"{treatment_name} vs {control_name}"
        )

        all_results.append(results)

    final_results = pd.concat(
        all_results,
        ignore_index=True,
    )

    print("\n=== UPLIFT MODEL COMPARISON ===")
    print(final_results.round(6).to_string(index=False))

    with mlflow.start_run(run_name="Uplift Model Comparison Summary"):
        mlflow.set_tags({
            "project": "RetailIQ",
            "task": "causal_uplift_modeling_summary",
        })

        mlflow.log_table(
            final_results,
            "uplift_model_comparison.json",
        )


if __name__ == "__main__":
    main()
