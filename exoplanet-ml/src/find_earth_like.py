"""
find_earth_like.py
------------------
Identify Earth-like planets in the exoplanet dataset using scikit-learn.

Strict Earth-like criteria:
  - Planet mass:      0.5 - 2.0 Earth masses   (rocky, Earth-sized)
  - Planet radius:    0.8 - 1.25 Earth radii   (Earth-sized, not a mini-Neptune)
  - Equilibrium temp: 180 - 310 K              (roughly liquid-water / habitable range)

scikit-learn is used to:
  - Impute missing values  (SimpleImputer, median)
  - Scale features         (StandardScaler)
  - Rank ALL planets by scaled Euclidean distance to Earth
    (smaller distance = more Earth-like)

Outputs the count of strict Earth-like planets, lists them, and lists the
top candidates ranked by distance to Earth.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# Resolve data path relative to this script so it works from any CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # exoplanet-ml folder
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "exoplanets.csv"

# Conversion factors: Jupiter units -> Earth units
MJ_TO_MEARTH = 317.8   # 1 Jupiter mass  = 317.8 Earth masses
RJ_TO_REARTH = 11.209  # 1 Jupiter radius = 11.209 Earth radii

# Earth reference point (in Earth units)
EARTH_MASS = 1.0        # Earth mass
EARTH_RADIUS = 1.0      # Earth radius
EARTH_TEMP_K = 288.0    # Earth mean surface temperature (K)

# Strict Earth-like thresholds
MASS_MIN, MASS_MAX = 0.5, 2.0
RADIUS_MIN, RADIUS_MAX = 0.8, 1.25
TEMP_MIN, TEMP_MAX = 180.0, 310.0

TOP_N = 15


def load_data(path):
    """Load the raw CSV and convert Jupiter units to Earth units."""
    df = pd.read_csv(path)
    num_cols = ["Mass (MJ)", "Radius (RJ)", "Temp. (K)", "Period (days)",
                "Semi-major axis (AU)", "Disc. Year", "Distance (ly)"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Mass_Earth"] = df["Mass (MJ)"] * MJ_TO_MEARTH
    df["Radius_Earth"] = df["Radius (RJ)"] * RJ_TO_REARTH
    return df


def main():
    df = load_data(RAW_PATH)

    # Use only rows with observed Mass, Radius and Temperature (do not fabricate data).
    model_df = df.dropna(subset=["Mass_Earth", "Radius_Earth", "Temp. (K)"]).copy()

    print("Total planets in raw dataset        :", len(df))
    print("Planets with Mass, Radius & Temp observed:", len(model_df), "\n")

    features = ["Mass_Earth", "Radius_Earth", "Temp. (K)"]
    X = model_df[features].values

    # scikit-learn: impute missing values, then standard-scale features.
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    # Earth reference vector, scaled the same way as the data.
    earth_vec = scaler.transform(np.array([[EARTH_MASS, EARTH_RADIUS, EARTH_TEMP_K]]))

    # scikit-learn: scaled Euclidean distance from every planet to Earth.
    dist = euclidean_distances(X_scaled, earth_vec).ravel()
    model_df["earth_distance"] = dist

    # ---- Strict Earth-like filter ----
    strict = model_df[
        model_df["Mass_Earth"].between(MASS_MIN, MASS_MAX) &
        model_df["Radius_Earth"].between(RADIUS_MIN, RADIUS_MAX) &
        model_df["Temp. (K)"].between(TEMP_MIN, TEMP_MAX)
    ].sort_values("earth_distance")

    print("=" * 70)
    print(f"STRICT EARTH-LIKE PLANETS: {len(strict)}")
    print(" Criteria: 0.5-2.0 MEarth, 0.8-1.25 REarth, 180-310 K")
    print("=" * 70)
    if len(strict):
        print(strict[["Name", "Mass_Earth", "Radius_Earth", "Temp. (K)",
                      "Discovery method", "Disc. Year", "Distance (ly)",
                      "earth_distance"]].round(3).to_string(index=False))
    else:
        print("(none found)")

    # ---- Top candidates ranked by distance to Earth (scikit-learn) ----
    print("\n" + "=" * 70)
    print(f"TOP {TOP_N} MOST EARTH-LIKE CANDIDATES")
    print(" (ranked by scikit-learn scaled Euclidean distance to Earth)")
    print("=" * 70)
    top = model_df.sort_values("earth_distance").head(TOP_N)
    print(top[["Name", "Mass_Earth", "Radius_Earth", "Temp. (K)",
               "Discovery method", "Disc. Year", "Distance (ly)",
               "earth_distance"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()