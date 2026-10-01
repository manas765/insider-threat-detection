from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
LDAP_DIR = ROOT / "data" / "raw" / "LDAP"

ACTIVE_FEATURES = [
    "login_hour",
    "after_hours_flag",
    "session_duration_mins",
    "usb_events_count",
    "files_accessed_count",
    "email_count",
    "unique_domains_visited",
    "email_ext_recipient_count",
    "usb_events_count_zscore",
    "files_accessed_count_zscore",
    "email_count_zscore",
    "session_duration_mins_zscore",
    "days_since_last_spike",
]

ROLE_NUMERIC_FEATURES = [
    "usb_events_count",
    "files_accessed_count",
    "email_count",
    "session_duration_mins",
]

ROLE_FEATURES = [
    "role_usb_events_count_zscore",
    "role_files_accessed_count_zscore",
    "role_email_count_zscore",
    "role_session_duration_mins_zscore",
    "role_after_hours_deviation",
]


def load_role_snapshots():
    exact_roles = {}
    cumulative_roles = {}
    current_roles = {}

    for path in sorted(LDAP_DIR.glob("*.csv")):
        month = path.stem

        ldap = pd.read_csv(
            path,
            usecols=["user_id", "role"],
        ).dropna(subset=["user_id", "role"])

        ldap["user_id"] = ldap["user_id"].astype(str).str.strip()
        ldap["role"] = ldap["role"].astype(str).str.strip()
        ldap = ldap.drop_duplicates("user_id", keep="last")

        snapshot_roles = dict(zip(ldap["user_id"], ldap["role"]))

        exact_roles[month] = snapshot_roles
        current_roles.update(snapshot_roles)
        cumulative_roles[month] = current_roles.copy()

    return exact_roles, cumulative_roles


def attach_roles(df, exact_roles, cumulative_roles):
    df = df.copy()
    df["user"] = df["user"].astype(str).str.strip()
    df["day"] = pd.to_datetime(df["day"])
    df["month"] = df["day"].dt.to_period("M").astype(str)

    roles = []
    fallback_count = 0

    for user, month in zip(df["user"], df["month"]):
        role = exact_roles.get(month, {}).get(user)

        if role is None:
            role = cumulative_roles.get(month, {}).get(user)
            fallback_count += 1

        roles.append(role)

    df["role"] = roles

    missing = df["role"].isna().sum()
    if missing:
        raise ValueError(f"Could not assign a role to {missing} user-day rows.")

    df = df.drop(columns="month")

    return df, fallback_count


def add_training_role_features(train_df):
    train_df = train_df.copy()
    benign = train_df[train_df["is_malicious"] == 0].copy()

    for feature in ROLE_NUMERIC_FEATURES:
        daily = (
            benign.groupby(["role", "day"])[feature]
            .agg(
                count="count",
                total="sum",
                sumsq=lambda s: np.square(s).sum(),
            )
            .reset_index()
            .sort_values(["role", "day"])
        )

        grouped = daily.groupby("role", group_keys=False)

        daily["prior_count"] = (
            grouped["count"].cumsum() - daily["count"]
        )
        daily["prior_total"] = (
            grouped["total"].cumsum() - daily["total"]
        )
        daily["prior_sumsq"] = (
            grouped["sumsq"].cumsum() - daily["sumsq"]
        )

        daily["prior_mean"] = (
            daily["prior_total"] / daily["prior_count"]
        )

        variance = (
            daily["prior_sumsq"]
            - (daily["prior_total"] ** 2 / daily["prior_count"])
        ) / (daily["prior_count"] - 1)

        daily["prior_std"] = np.sqrt(
            variance.clip(lower=0)
        )

        stats = daily[
            ["role", "day", "prior_count", "prior_mean", "prior_std"]
        ]

        train_df = train_df.merge(
            stats,
            on=["role", "day"],
            how="left",
        )

        output_name = f"role_{feature}_zscore"

        train_df[output_name] = np.where(
            (train_df["prior_count"] >= 2)
            & (train_df["prior_std"] > 0),
            (
                train_df[feature] - train_df["prior_mean"]
            ) / train_df["prior_std"],
            0.0,
        )

        train_df = train_df.drop(
            columns=["prior_count", "prior_mean", "prior_std"]
        )

    daily_after_hours = (
        benign.groupby(["role", "day"])["after_hours_flag"]
        .agg(
            count="count",
            total="sum",
        )
        .reset_index()
        .sort_values(["role", "day"])
    )

    grouped = daily_after_hours.groupby("role", group_keys=False)

    daily_after_hours["prior_count"] = (
        grouped["count"].cumsum() - daily_after_hours["count"]
    )
    daily_after_hours["prior_total"] = (
        grouped["total"].cumsum() - daily_after_hours["total"]
    )

    daily_after_hours["prior_rate"] = (
        daily_after_hours["prior_total"]
        / daily_after_hours["prior_count"]
    )

    train_df = train_df.merge(
        daily_after_hours[
            ["role", "day", "prior_count", "prior_rate"]
        ],
        on=["role", "day"],
        how="left",
    )

    train_df["role_after_hours_deviation"] = np.where(
        train_df["prior_count"] > 0,
        train_df["after_hours_flag"] - train_df["prior_rate"],
        0.0,
    )

    train_df = train_df.drop(
        columns=["prior_count", "prior_rate"]
    )

    return train_df


def add_test_role_features(test_df, training_df, test_start_day):
    test_df = test_df.copy()

    reference = training_df[
        (training_df["day"] < test_start_day)
        & (training_df["is_malicious"] == 0)
    ].copy()

    for feature in ROLE_NUMERIC_FEATURES:
        stats = (
            reference.groupby("role")[feature]
            .agg(["mean", "std"])
            .reset_index()
        )

        test_df = test_df.merge(
            stats,
            on="role",
            how="left",
        )

        output_name = f"role_{feature}_zscore"

        test_df[output_name] = np.where(
            test_df["std"].notna() & (test_df["std"] > 0),
            (test_df[feature] - test_df["mean"]) / test_df["std"],
            0.0,
        )

        test_df = test_df.drop(columns=["mean", "std"])

    after_hours_stats = (
        reference.groupby("role")["after_hours_flag"]
        .mean()
        .rename("role_after_hours_rate")
        .reset_index()
    )

    test_df = test_df.merge(
        after_hours_stats,
        on="role",
        how="left",
    )

    test_df["role_after_hours_deviation"] = np.where(
        test_df["role_after_hours_rate"].notna(),
        test_df["after_hours_flag"]
        - test_df["role_after_hours_rate"],
        0.0,
    )

    test_df = test_df.drop(columns=["role_after_hours_rate"])

    return test_df


def select_output_columns(df):
    return df[
        [
            "user",
            "day",
            "role",
            *ACTIVE_FEATURES,
            *ROLE_FEATURES,
            "is_malicious",
        ]
    ]


def main():
    train = pd.read_csv(PROCESSED / "train_with_ids.csv")
    test = pd.read_csv(PROCESSED / "test_with_ids.csv")

    exact_roles, cumulative_roles = load_role_snapshots()

    train, train_fallbacks = attach_roles(
        train,
        exact_roles,
        cumulative_roles,
    )

    test, test_fallbacks = attach_roles(
        test,
        exact_roles,
        cumulative_roles,
    )

    train = add_training_role_features(train)

    test_start_day = test["day"].min()

    test = add_test_role_features(
        test,
        train,
        test_start_day,
    )

    train_output = select_output_columns(train)
    test_output = select_output_columns(test)

    train_output.to_csv(
        PROCESSED / "train_role_features.csv",
        index=False,
    )

    test_output.to_csv(
        PROCESSED / "test_role_features.csv",
        index=False,
    )

    print("Role feature generation complete.")
    print(f"Train rows: {len(train_output)}")
    print(f"Test rows: {len(test_output)}")
    print(f"Train role fallbacks: {train_fallbacks}")
    print(f"Test role fallbacks: {test_fallbacks}")
    print(f"Role count: {train_output['role'].nunique()}")
    print("New role features:")
    for feature in ROLE_FEATURES:
        print(f"  - {feature}")
    print("Saved:")
    print("  data/processed/train_role_features.csv")
    print("  data/processed/test_role_features.csv")


if __name__ == "__main__":
    main()