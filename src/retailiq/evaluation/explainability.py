
import pandas as pd
from sklearn.inspection import permutation_importance


def calculate_permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42,
):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    return importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def print_top_features(importance, top_n=15):
    print("\nTOP FEATURES")
    print(
        importance.head(top_n).to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}",
        )
    )
