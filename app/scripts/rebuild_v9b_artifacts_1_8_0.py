from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

BASE_RANDOM_STATE = 42

DATA_PATH = REPO_ROOT / "app" / "data" / "compiled_model_ready" / "MR_cities_worldpop_knn_macro_v5.csv"
OUT_DIR = REPO_ROOT / "app" / "models" / "safety" / "artifacts" / "geo_safety_model_v9b_torch_mlp"

TARGET_COL = "safety_index"

knn_feature_cols = [
    "dist_nearest_labeled_city",
    "log1p_dist_nearest_labeled_city",
    "crime_nearest_labeled_city",
    "safety_nearest_labeled_city",
    "same_country_as_nearest_labeled",
    "avg_crime_k5",
    "avg_safety_k5",
    "avg_crime_k10",
    "avg_safety_k10",
    "wavg_crime_k5",
    "wavg_safety_k5",
    "log1p_num_labeled_within_50km",
    "log1p_num_labeled_within_100km",
    "log1p_num_labeled_within_250km",
    "avg_crime_same_country_k5",
    "avg_safety_same_country_k5",
    "log1p_num_same_country_within_250km",
]

density_gravity_feature_cols = [
    "log1p_num_cities_50km",
    "log1p_sum_pop_50km",
    "log1p_pop_gravity_50km",
    "log1p_num_cities_100km",
    "log1p_sum_pop_100km",
    "log1p_pop_gravity_100km",
    "dist_to_nearest_large_city",
    "log1p_dist_to_nearest_large_city",
    "log1p_pop_of_nearest_large_city",
]

base_feature_cols = [
    "lat",
    "lon",
    "crimeindex_2020",
    "crimeindex_2023",
    "safetyindex_2020",
    "age_0_14",
    "age_15_64",
    "age_65_plus",
    "population",
    "density_per_km2",
]

macro_cols_v5 = [
    "gdp",
    "gdp_per_capita",
    "unemployment",
    "homicide_rate",
    "life_expectancy_male",
    "life_expectancy_female",
    "infant_mortality",
    "urban_population_growth",
    "tourists",
]

all_feature_cols = list(
    dict.fromkeys(
        knn_feature_cols + density_gravity_feature_cols + base_feature_cols + macro_cols_v5
    )
)

BEST_FEATURE_SET_NAME = "geo_knn_macro_plus_city_crime_only"
BEST_FEATURE_LIST = [
    "lat",
    "lon",
    "dist_nearest_labeled_city",
    "log1p_dist_nearest_labeled_city",
    "crime_nearest_labeled_city",
    "safety_nearest_labeled_city",
    "same_country_as_nearest_labeled",
    "avg_crime_k5",
    "avg_safety_k5",
    "avg_crime_k10",
    "avg_safety_k10",
    "wavg_crime_k5",
    "wavg_safety_k5",
    "log1p_num_labeled_within_50km",
    "log1p_num_labeled_within_100km",
    "log1p_num_labeled_within_250km",
    "avg_crime_same_country_k5",
    "avg_safety_same_country_k5",
    "log1p_num_same_country_within_250km",
    "gdp",
    "gdp_per_capita",
    "unemployment",
    "homicide_rate",
    "life_expectancy_male",
    "life_expectancy_female",
    "infant_mortality",
    "urban_population_growth",
    "tourists",
    "crimeindex_2020",
    "crimeindex_2023",
]


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' missing")

    avail_feature_cols = [c for c in all_feature_cols if c in df.columns]

    missing_best = [c for c in BEST_FEATURE_LIST if c not in avail_feature_cols]
    if missing_best:
        raise ValueError(
            f"Some best feature columns not found in dataset: {missing_best}"
        )

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    for col in avail_feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X_all = df[avail_feature_cols].copy()
    y_all = df[TARGET_COL].astype(float).copy()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=BASE_RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=BASE_RANDOM_STATE
    )

    Xtr_best = X_train[BEST_FEATURE_LIST].copy()

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    _ = scaler.fit(imputer.fit_transform(Xtr_best))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    imputer_path = OUT_DIR / "v9b_best_mlp_imputer.joblib"
    scaler_path = OUT_DIR / "v9b_best_mlp_scaler.joblib"
    features_path = OUT_DIR / "v9b_best_mlp_features.joblib"

    joblib.dump(imputer, imputer_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(BEST_FEATURE_LIST, features_path)

    print(f"[DONE] Rebuilt v9b artifacts under current sklearn version.")
    print(f"       Imputer  -> {imputer_path}")
    print(f"       Scaler   -> {scaler_path}")
    print(f"       Features -> {features_path}")


if __name__ == "__main__":
    main()