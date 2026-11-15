
import numpy as np
import pandas as pd

from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold


from retailiq.config import DATABASE_URL
engine = create_engine(DATABASE_URL)

FEATURES = [
    "recency",
    "history",
    "mens",
    "womens",
    "newbie",
    "history_segment",
    "zip_code",
    "channel",
]

TREATMENT = "Mens E-Mail"
CONTROL = "No E-Mail"

ALLOCATION_RATE = 0.30
EMAIL_COST = 0.01  # Illustrative GBP cost per email
FOLDS = 5
SEED = 42
BOOTSTRAPS = 2000


def load_data():
    query = """
        SELECT
            recency,
            history,
            mens,
            womens,
            newbie,
            history_segment,
            zip_code,
            channel,
            segment,
            spend
        FROM hillstrom.campaign_data
        WHERE segment IN ('Mens E-Mail', 'No E-Mail')
    """

    with engine.connect() as connection:
        df = pd.read_sql(text(query), connection)

    if df.empty:
        raise ValueError("No campaign observations were loaded.")

    required_columns = FEATURES + ["segment", "spend"]

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found in campaign data.")

    if not np.isfinite(df["spend"].to_numpy(dtype=float)).all():
        raise ValueError("Spend contains non-finite values.")

    if not df["spend"].ge(0).all():
        raise ValueError("Spend cannot be negative.")

    if not df["segment"].isin([TREATMENT, CONTROL]).all():
        raise ValueError("Unexpected treatment assignment.")

    return df.reset_index(drop=True)


def prepare_features(df):
    X = pd.get_dummies(
        df[FEATURES],
        columns=["history_segment", "zip_code", "channel"],
        drop_first=False,
        dtype=int,
    )

    if X.empty:
        raise ValueError("No model features were generated.")

    return X


def outer_cross_fitted_policy(X, spend, treatment):
    """
    Generate out-of-sample uplift predictions and campaign decisions.

    Each outer test fold is excluded from model training. The top
    ALLOCATION_RATE customers are selected separately within each
    held-out fold, without using that fold's observed spend outcomes.
    """
    n = len(X)

    uplift_predictions = np.full(n, np.nan, dtype=float)
    selected_policy = np.zeros(n, dtype=int)
    fold_ids = np.full(n, -1, dtype=int)
    fold_allocation_rates = []

    cv = StratifiedKFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, treatment),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = spend.iloc[train_idx].to_numpy(dtype=float)
        t_train = treatment[train_idx]

        treated_idx = np.flatnonzero(t_train == 1)
        control_idx = np.flatnonzero(t_train == 0)

        if len(treated_idx) < 2 or len(control_idx) < 2:
            raise ValueError(
                f"Fold {fold} does not contain enough observations "
                "in both treatment groups."
            )

        treated_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=30,
            random_state=SEED + fold,
            n_jobs=-1,
        )

        control_model = RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=30,
            random_state=SEED + fold + FOLDS,
            n_jobs=-1,
        )

        treated_model.fit(
            X_train.iloc[treated_idx],
            y_train[treated_idx],
        )

        control_model.fit(
            X_train.iloc[control_idx],
            y_train[control_idx],
        )

        fold_uplift = (
            treated_model.predict(X_test)
            - control_model.predict(X_test)
        )

        if not np.isfinite(fold_uplift).all():
            raise ValueError(
                f"Non-finite uplift predictions in fold {fold}."
            )

        uplift_predictions[test_idx] = fold_uplift
        fold_ids[test_idx] = fold

        contact_count = int(
            np.floor(len(test_idx) * ALLOCATION_RATE)
        )

        if contact_count < 1:
            raise ValueError(
                f"Fold {fold} is too small for the target allocation."
            )

        selected_local = np.argsort(
            fold_uplift,
            kind="stable",
        )[-contact_count:]

        selected_global = test_idx[selected_local]
        selected_policy[selected_global] = 1

        fold_allocation_rates.append(
            selected_policy[test_idx].mean()
        )

        print(
            f"Fold {fold}: "
            f"training customers={len(train_idx):,}, "
            f"held-out customers={len(test_idx):,}, "
            f"selected={contact_count:,}, "
            f"allocation={selected_policy[test_idx].mean():.3%}"
        )

    if np.isnan(uplift_predictions).any():
        raise ValueError("Some customers did not receive an out-of-fold prediction.")

    if (fold_ids < 1).any():
        raise ValueError("Some customers were not assigned to an outer fold.")

    return (
        uplift_predictions,
        selected_policy,
        fold_ids,
        fold_allocation_rates,
    )


def policy_value_contributions(
    spend,
    treatment,
    policy,
    treatment_probability,
):
    """
    Customer-level inverse-propensity weighted revenue contributions.

    policy is the probability of contacting a customer:
      0.0 = do not contact
      0.3 = random 30% allocation
      1.0 = contact the selected customer
    """
    spend = np.asarray(spend, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    policy = np.asarray(policy, dtype=float)

    if not (
        len(spend) == len(treatment) == len(policy)
    ):
        raise ValueError("Spend, treatment and policy lengths differ.")

    if not 0 < treatment_probability < 1:
        raise ValueError("Treatment probability must be between 0 and 1.")

    if not np.isfinite(policy).all():
        raise ValueError("Policy contains non-finite values.")

    if ((policy < 0) | (policy > 1)).any():
        raise ValueError("Policy probabilities must be between 0 and 1.")

    if not np.isfinite(spend).all():
        raise ValueError("Spend contains non-finite values.")

    if not np.isin(treatment, [0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    p = treatment_probability

    return (
        treatment * spend * policy / p
        + (1 - treatment) * spend * (1 - policy) / (1 - p)
    )


def evaluate_policies(
    spend,
    treatment,
    selected_policy,
    treatment_probability,
):
    policies = {
        "No campaign": np.zeros(len(spend), dtype=float),
        "Random 30% allocation": np.full(
            len(spend),
            ALLOCATION_RATE,
            dtype=float,
        ),
        "Uplift-based 30% allocation": selected_policy.astype(float),
    }

    rows = []
    profit_contributions = {}

    no_campaign_revenue_contributions = policy_value_contributions(
        spend,
        treatment,
        policies["No campaign"],
        treatment_probability,
    )

    no_campaign_profit_contributions = (
        no_campaign_revenue_contributions
    )

    profit_contributions["No campaign"] = (
        no_campaign_profit_contributions
    )

    for name, policy in policies.items():
        revenue_contributions = policy_value_contributions(
            spend,
            treatment,
            policy,
            treatment_probability,
        )

        contact_rate = float(policy.mean())
        campaign_cost_per_customer = contact_rate * EMAIL_COST

        net_profit_contributions = (
            revenue_contributions - policy * EMAIL_COST
        )

        profit_contributions[name] = net_profit_contributions

        revenue_per_customer = float(
            revenue_contributions.mean()
        )

        net_profit_per_customer = float(
            net_profit_contributions.mean()
        )

        rows.append(
            {
                "strategy": name,
                "contact_rate": contact_rate,
                "revenue_per_customer": revenue_per_customer,
                "campaign_cost_per_customer": campaign_cost_per_customer,
                "net_profit_per_customer": net_profit_per_customer,
            }
        )

    results = pd.DataFrame(rows)

    baseline_profit = results.loc[
        results["strategy"] == "No campaign",
        "net_profit_per_customer",
    ].iloc[0]

    results["incremental_profit_per_1000"] = (
        results["net_profit_per_customer"] - baseline_profit
    ) * 1000

    return results, profit_contributions


def bootstrap_profit_intervals(profit_contributions):
    """
    Paired customer-level bootstrap of incremental net profit.

    The outer-fold models and customer-level policy decisions are held
    fixed. This interval does not include model-retraining uncertainty.
    """
    rng = np.random.default_rng(SEED)

    no_campaign = profit_contributions["No campaign"]

    strategies = [
        "Random 30% allocation",
        "Uplift-based 30% allocation",
    ]

    differences = {
        name: profit_contributions[name] - no_campaign
        for name in strategies
    }

    n = len(no_campaign)

    if any(len(values) != n for values in differences.values()):
        raise ValueError("Bootstrap contribution lengths do not match.")

    bootstrap_results = {
        name: np.empty(BOOTSTRAPS, dtype=float)
        for name in strategies
    }

    for iteration in range(BOOTSTRAPS):
        sample_idx = rng.integers(0, n, size=n)

        for name in strategies:
            bootstrap_results[name][iteration] = (
                differences[name][sample_idx].mean() * 1000
            )

    return {
        name: np.percentile(values, [2.5, 97.5])
        for name, values in bootstrap_results.items()
    }


def main():
    df = load_data()

    X = prepare_features(df)

    spend = df["spend"].astype(float).reset_index(drop=True)

    treatment = (
        df["segment"] == TREATMENT
    ).astype(int).to_numpy()

    if np.unique(treatment).size != 2:
        raise ValueError(
            "Both treatment and control observations are required."
        )

    treatment_counts = np.bincount(treatment)

    if treatment_counts.min() < FOLDS:
        raise ValueError(
            "Not enough observations in one group for the selected folds."
        )

    treatment_probability = float(treatment.mean())

    print("\nOUTER CROSS-FITTED CAMPAIGN POLICY EVALUATION")
    print(f"Treatment: {TREATMENT}")
    print(f"Control: {CONTROL}")
    print(f"Treatment assignment probability: {treatment_probability:.4f}")
    print(f"Target allocation rate: {ALLOCATION_RATE:.0%}")
    print(f"Assumed email cost: £{EMAIL_COST:.2f}")

    (
        uplift_predictions,
        selected_policy,
        fold_ids,
        fold_allocation_rates,
    ) = outer_cross_fitted_policy(
        X,
        spend,
        treatment,
    )

    results, profit_contributions = evaluate_policies(
        spend.to_numpy(dtype=float),
        treatment,
        selected_policy,
        treatment_probability,
    )

    intervals = bootstrap_profit_intervals(
        profit_contributions
    )

    print("\nPROFIT-BASED CAMPAIGN OPTIMIZATION")
    print(
        results.to_string(
            index=False,
            formatters={
                "contact_rate": "{:.4f}".format,
                "revenue_per_customer": "{:.4f}".format,
                "campaign_cost_per_customer": "{:.4f}".format,
                "net_profit_per_customer": "{:.4f}".format,
                "incremental_profit_per_1000": "{:.4f}".format,
            },
        )
    )

    print("\nINCREMENTAL PROFIT CONFIDENCE INTERVALS")
    for name, (lower, upper) in intervals.items():
        print(
            f"{name}: £{lower:.2f} to £{upper:.2f} "
            "per 1,000 customers"
        )

    print("\nPOLICY EVALUATION DETAILS")
    print(f"Customers evaluated: {len(df):,}")
    print(f"Uplift-targeted customers: {selected_policy.sum():,}")
    print(f"Outer cross-fitting folds: {FOLDS}")
    print(
        "Average held-out fold allocation: "
        f"{np.mean(fold_allocation_rates):.2%}"
    )
    print(f"Bootstrap repetitions: {BOOTSTRAPS}")
    print(
        "Customers with out-of-fold uplift predictions: "
        f"{np.isfinite(uplift_predictions).sum():,}"
    )
    print(
        "Unique outer folds represented: "
        f"{len(np.unique(fold_ids))}"
    )

    print("\nLIMITATIONS:")
    print("- Email cost is an illustrative assumption, not a verified cost.")
    print(
        "- Each customer's uplift prediction and policy decision are "
        "generated without training on that customer's outer fold."
    )
    print(
        "- Bootstrap intervals hold the outer-fold policy decisions fixed "
        "and do not include model-retraining uncertainty."
    )
    print(
        "- The inverse-propensity estimator relies on randomized treatment "
        "assignment and the treatment probability used above."
    )
    print(
        "- Results estimate policy value in this experiment; they do not "
        "guarantee future campaign profit."
    )


if __name__ == "__main__":
    main()
