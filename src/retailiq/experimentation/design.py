
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_sample_size(
    baseline_rate,
    minimum_detectable_effect,
    alpha=0.05,
    power=0.80,
):
    """
    Required sample size per group for a two-sided
    comparison of two independent conversion proportions.

    minimum_detectable_effect is an absolute difference.
    """

    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")

    if minimum_detectable_effect <= 0:
        raise ValueError("MDE must be positive.")

    treatment_rate = baseline_rate + minimum_detectable_effect

    if not 0 < treatment_rate < 1:
        raise ValueError("Baseline + MDE must be between 0 and 1.")

    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("Alpha and power must be between 0 and 1.")

    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    pooled_rate = (baseline_rate + treatment_rate) / 2

    numerator = (
        z_alpha * np.sqrt(2 * pooled_rate * (1 - pooled_rate))
        + z_power * np.sqrt(
            baseline_rate * (1 - baseline_rate)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2

    sample_size = numerator / minimum_detectable_effect**2

    return int(np.ceil(sample_size))


def minimum_detectable_effect(
    baseline_rate,
    sample_size_per_group,
    alpha=0.05,
    power=0.80,
):
    """Find an approximate absolute MDE by numerical search."""

    if not 0 < baseline_rate < 1:
        raise ValueError("Invalid baseline conversion rate.")

    if sample_size_per_group <= 0:
        raise ValueError("Sample size must be positive.")

    low = 1e-6
    high = 1 - baseline_rate

    for _ in range(100):
        mid = (low + high) / 2

        required = calculate_sample_size(
            baseline_rate,
            mid,
            alpha,
            power,
        )

        if required > sample_size_per_group:
            low = mid
        else:
            high = mid

    return high


def assign_experiment_groups(
    df,
    id_column,
    allocation=None,
    seed=42,
):
    """
    Randomly assign unique subjects to experiment groups.
    Default allocation: 50% control, 50% treatment.
    """

    if id_column not in df.columns:
        raise ValueError(f"Missing ID column: {id_column}")

    if df[id_column].isna().any():
        raise ValueError("Subject IDs cannot be missing.")

    if df[id_column].duplicated().any():
        raise ValueError("Each subject must appear only once.")

    if allocation is None:
        allocation = {
            "control": 0.5,
            "treatment": 0.5,
        }

    if any(value <= 0 for value in allocation.values()):
        raise ValueError("Allocation proportions must be positive.")

    if not np.isclose(sum(allocation.values()), 1.0):
        raise ValueError("Allocation proportions must sum to 1.")

    rng = np.random.default_rng(seed)

    subjects = df[id_column].to_numpy().copy()
    rng.shuffle(subjects)

    proportions = np.array(list(allocation.values()))
    expected_counts = proportions * len(subjects)

    counts = np.floor(expected_counts).astype(int)

    remaining = len(subjects) - counts.sum()

    if remaining:
        fractions = expected_counts - counts
        priority = np.argsort(-fractions)
        counts[priority[:remaining]] += 1

    labels = []

    for group, count in zip(allocation.keys(), counts):
        labels.extend([group] * count)

    assignment = pd.DataFrame({
        id_column: subjects,
        "experiment_group": labels,
    })

    return assignment


def main():
    baseline_conversion = 0.00573
    mde = 0.002

    required = calculate_sample_size(
        baseline_rate=baseline_conversion,
        minimum_detectable_effect=mde,
        alpha=0.05,
        power=0.80,
    )

    print("=== EXPERIMENT SAMPLE SIZE ===")
    print(f"Baseline conversion: {baseline_conversion:.4%}")
    print(f"Minimum detectable effect: {mde:.4%}")
    print(f"Required sample size per group: {required:,}")
    print(f"Total required sample size: {required * 2:,}")

    print("\n=== MDE FOR HILLSTROM-SIZED GROUPS ===")

    detectable = minimum_detectable_effect(
        baseline_rate=baseline_conversion,
        sample_size_per_group=21306,
    )

    print(f"Approximate detectable effect: {detectable:.4%}")


if __name__ == "__main__":
    main()
