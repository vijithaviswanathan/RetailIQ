
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import chi2_contingency, norm, kruskal
from statsmodels.stats.proportion import (
    proportions_ztest,
    proportion_confint,
)
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import chisquare, ttest_ind
import statsmodels.formula.api as smf

from retailiq.config import DATABASE_URL

engine = create_engine(DATABASE_URL)

QUERY = """
SELECT
    segment,
    visit,
    conversion,
    spend,
    recency,
    history,
    history_segment,
    mens,
    womens,
    newbie,
    channel
FROM hillstrom.campaign_data;
"""


def load_data():
    with engine.connect() as connection:
        return pd.read_sql(text(QUERY), connection)


def validate_data(df):
    required = {
        "segment", "visit", "conversion", "spend",
        "recency", "history", "mens", "womens",
        "newbie", "channel"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Hillstrom dataset is empty.")

    if not df["segment"].isin(
        ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
    ).all():
        raise ValueError("Unexpected campaign segment values.")

    for column in ["visit", "conversion"]:
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 and 1.")

    if (df["conversion"] > df["visit"]).any():
        raise ValueError("A conversion cannot occur without a visit.")

    if df["spend"].isna().any() or (df["spend"] < 0).any():
        raise ValueError("Invalid spend values.")

    return True


def campaign_summary(df):
    summary = (
        df.groupby("segment")
        .agg(
            customers=("segment", "size"),
            visits=("visit", "sum"),
            conversions=("conversion", "sum"),
            total_spend=("spend", "sum"),
            average_spend=("spend", "mean"),
        )
    )

    summary["conversion_rate"] = (
        summary["conversions"] / summary["customers"]
    )

    summary["visit_rate"] = (
        summary["visits"] / summary["customers"]
    )

    summary["revenue_per_customer"] = (
        summary["total_spend"] / summary["customers"]
    )

    return summary


def conversion_test(df, group_a, group_b):
    a = df[df["segment"] == group_a]["conversion"]
    b = df[df["segment"] == group_b]["conversion"]

    counts = [int(a.sum()), int(b.sum())]
    sample_sizes = [len(a), len(b)]

    statistic, p_value = proportions_ztest(
        counts,
        sample_sizes,
        alternative="two-sided",
    )

    rate_a = a.mean()
    rate_b = b.mean()

    difference = rate_a - rate_b

    pooled_rate = sum(counts) / sum(sample_sizes)
    standard_error = (
        pooled_rate * (1 - pooled_rate)
        * (1 / sample_sizes[0] + 1 / sample_sizes[1])
    ) ** 0.5

    ci_low = difference - norm.ppf(0.975) * (
        rate_a * (1 - rate_a) / sample_sizes[0]
        + rate_b * (1 - rate_b) / sample_sizes[1]
    ) ** 0.5

    ci_high = difference + norm.ppf(0.975) * (
        rate_a * (1 - rate_a) / sample_sizes[0]
        + rate_b * (1 - rate_b) / sample_sizes[1]
    ) ** 0.5

    return {
        "comparison": f"{group_a} vs {group_b}",
        "group_a_rate": rate_a,
        "group_b_rate": rate_b,
        "absolute_difference": difference,
        "relative_lift": (
            difference / rate_b if rate_b != 0 else None
        ),
        "confidence_interval_low": ci_low,
        "confidence_interval_high": ci_high,
        "z_statistic": statistic,
        "p_value": p_value,
    }


def conversion_tests(df):
    comparisons = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]

    results = pd.DataFrame([
        conversion_test(df, a, b)
        for a, b in comparisons
    ])

    results["adjusted_p_value"] = multipletests(
        results["p_value"],
        method="holm",
    )[1]

    results["statistically_significant"] = (
        results["adjusted_p_value"] < 0.05
    )

    return results

def visit_tests(df):
    comparisons = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]

    rows = []

    for group_a, group_b in comparisons:
        a = df.loc[df["segment"] == group_a, "visit"]
        b = df.loc[df["segment"] == group_b, "visit"]

        counts = [int(a.sum()), int(b.sum())]
        sample_sizes = [len(a), len(b)]

        statistic, p_value = proportions_ztest(
            counts,
            sample_sizes,
            alternative="two-sided",
        )

        rate_a = a.mean()
        rate_b = b.mean()
        difference = rate_a - rate_b

        standard_error = np.sqrt(
            rate_a * (1 - rate_a) / len(a)
            + rate_b * (1 - rate_b) / len(b)
        )

        rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "visit_rate_a": rate_a,
            "visit_rate_b": rate_b,
            "difference": difference,
            "confidence_interval_low": (
                difference - norm.ppf(0.975) * standard_error
            ),
            "confidence_interval_high": (
                difference + norm.ppf(0.975) * standard_error
            ),
            "p_value": p_value,
        })

    results = pd.DataFrame(rows)

    results["adjusted_p_value"] = multipletests(
        results["p_value"],
        method="holm",
    )[1]

    results["statistically_significant"] = (
        results["adjusted_p_value"] < 0.05
    )

    return results


def revenue_tests(df):
    comparisons = [
        ("Mens E-Mail", "No E-Mail"),
        ("Womens E-Mail", "No E-Mail"),
        ("Mens E-Mail", "Womens E-Mail"),
    ]

    rows = []

    rng = np.random.default_rng(42)
    bootstrap_iterations = 5000

    for group_a, group_b in comparisons:
        a = df.loc[df["segment"] == group_a, "spend"].to_numpy()
        b = df.loc[df["segment"] == group_b, "spend"].to_numpy()

        test = ttest_ind(a, b, equal_var=False)

        bootstrap_differences = np.empty(bootstrap_iterations)

        for i in range(bootstrap_iterations):
            sample_a = rng.choice(a, size=len(a), replace=True)
            sample_b = rng.choice(b, size=len(b), replace=True)

            bootstrap_differences[i] = (
                sample_a.mean() - sample_b.mean()
            )

        ci_low, ci_high = np.percentile(
            bootstrap_differences, [2.5, 97.5]
        )

        rows.append({
            "comparison": f"{group_a} vs {group_b}",
            "mean_spend_a": a.mean(),
            "mean_spend_b": b.mean(),
            "difference": a.mean() - b.mean(),
            "confidence_interval_low": ci_low,
            "confidence_interval_high": ci_high,
            "welch_t_statistic": test.statistic,
            "p_value": test.pvalue,
        })

    results = pd.DataFrame(rows)

    results["adjusted_p_value"] = multipletests(
        results["p_value"],
        method="holm",
    )[1]

    results["statistically_significant"] = (
        results["adjusted_p_value"] < 0.05
    )

    return results

def covariate_balance(df):
    """
    Assess observed pre-treatment covariate balance
    across the three randomized campaign groups.
    """

    numeric_covariates = ["recency", "history"]

    categorical_covariates = [
        "history_segment",
        "mens",
        "womens",
        "newbie",
        "channel",
    ]

    rows = []

    for column in numeric_covariates:
        groups = [
            group[column].dropna().to_numpy()
            for _, group in df.groupby("segment")
        ]

        statistic, p_value = kruskal(*groups)

        n = sum(len(group) for group in groups)
        k = len(groups)

        epsilon_squared = max(
            0,
            (statistic - k + 1) / (n - k)
        )

        rows.append({
            "covariate": column,
            "test": "Kruskal-Wallis",
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": epsilon_squared,
            "effect_size_type": "Epsilon-squared",
        })

    for column in categorical_covariates:
        contingency = pd.crosstab(
            df[column],
            df["segment"],
        )

        statistic, p_value, _, _ = chi2_contingency(
            contingency
        )

        n = contingency.to_numpy().sum()

        minimum_dimension = min(
            contingency.shape[0] - 1,
            contingency.shape[1] - 1,
        )

        cramers_v = (
            np.sqrt(statistic / (n * minimum_dimension))
            if minimum_dimension > 0 and n > 0
            else 0
        )

        rows.append({
            "covariate": column,
            "test": "Chi-square",
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": cramers_v,
            "effect_size_type": "Cramer's V",
        })

    results = pd.DataFrame(rows)

    results["adjusted_p_value"] = multipletests(
        results["p_value"],
        method="holm",
    )[1]

    results["imbalance_detected"] = (
        results["adjusted_p_value"] < 0.05
    )

    return results

def adjusted_treatment_effects(df):
    data = df.copy()

    data["mens_email"] = (
        data["segment"] == "Mens E-Mail"
    ).astype(int)

    data["womens_email"] = (
        data["segment"] == "Womens E-Mail"
    ).astype(int)

    covariates = (
        "recency + history + C(history_segment) "
        "+ mens + womens + newbie + C(channel)"
    )

    conversion_model = smf.logit(
        f"conversion ~ mens_email + womens_email + {covariates}",
        data=data,
    ).fit(disp=False, cov_type="HC3")

    spend_model = smf.ols(
        f"spend ~ mens_email + womens_email + {covariates}",
        data=data,
    ).fit(cov_type="HC3")

    rows = []

    for treatment, label in [
        ("mens_email", "Mens E-Mail vs No E-Mail"),
        ("womens_email", "Womens E-Mail vs No E-Mail"),
    ]:
        for outcome, model in [
            ("Conversion (log-odds)", conversion_model),
            ("Spend per customer (£)", spend_model),
        ]:
            estimate = model.params[treatment]
            confidence_interval = model.conf_int().loc[treatment]

            rows.append({
                "comparison": label,
                "outcome": outcome,
                "adjusted_estimate": estimate,
                "confidence_interval_low": confidence_interval.iloc[0],
                "confidence_interval_high": confidence_interval.iloc[1],
                "p_value": model.pvalues[treatment],
            })

    return pd.DataFrame(rows)


def experiment_diagnostics(df):
    counts = df["segment"].value_counts()

    expected_counts = np.repeat(
        len(df) / 3, 3
    )

    ordered_counts = counts.reindex([
        "Mens E-Mail",
        "Womens E-Mail",
        "No E-Mail",
    ]).to_numpy()

    statistic, p_value = chisquare(
        ordered_counts,
        f_exp=expected_counts,
    )

    return {
        "group_counts": counts.to_dict(),
        "group_proportions": (
            counts / len(df)
        ).to_dict(),
        "expected_allocation": "1:1:1",
        "srm_chi_square": statistic,
        "srm_p_value": p_value,
        "srm_detected": p_value < 0.001,
    }

def aa_test_simulation(
    iterations=1000,
    sample_size=10000,
    baseline_rate=0.10,
    alpha=0.05,
    seed=42,
):
    rng = np.random.default_rng(seed)
    p_values = []

    for _ in range(iterations):
        group_a = rng.binomial(
            1, baseline_rate, sample_size
        )
        group_b = rng.binomial(
            1, baseline_rate, sample_size
        )

        _, p_value = proportions_ztest(
            [
                group_a.sum(),
                group_b.sum(),
            ],
            [
                sample_size,
                sample_size,
            ],
        )

        p_values.append(p_value)

    false_positive_rate = np.mean(
        np.array(p_values) < alpha
    )

    return {
        "iterations": iterations,
        "sample_size_per_group": sample_size,
        "baseline_conversion_rate": baseline_rate,
        "significance_level": alpha,
        "false_positive_rate": false_positive_rate,
        "expected_false_positive_rate": alpha,
    }


def main():
    df = load_data()
    validate_data(df)

    print("\n=== CAMPAIGN SUMMARY ===")
    print(campaign_summary(df).round(4))

    print("\n=== CONVERSION TESTS ===")
    print(conversion_tests(df).round(6))

    print("\n=== VISIT RATE GUARDRAIL ANALYSIS ===")
    print(visit_tests(df).round(6))

    print("\n=== REVENUE COMPARISONS ===")
    print(revenue_tests(df).round(4))

    print("\n=== PRE-TREATMENT COVARIATE BALANCE ===")
    print(covariate_balance(df).round(6))

    print("\n=== REGRESSION-ADJUSTED TREATMENT EFFECTS ===")
    print(adjusted_treatment_effects(df).round(6))

    print("\n=== EXPERIMENT DIAGNOSTICS ===")
    print(experiment_diagnostics(df))

    print("\n=== SYNTHETIC A/A TEST VALIDATION ===")
    print(aa_test_simulation())


if __name__ == "__main__":
    main()
