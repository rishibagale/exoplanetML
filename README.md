# exoplanetML

Finding Earth-like planets in an exoplanet dataset using **scikit-learn**.

## Goal

Answer the question: **"How many Earth-like planets are in this data set, and which ones are they?"**

The project loads a catalog of ~4,855 confirmed exoplanets, converts Jupiter units to Earth units, and uses scikit-learn (`SimpleImputer`, `StandardScaler`, `euclidean_distances`) to rank every planet by its scaled Euclidean distance to Earth. It then applies a strict "Earth-like" filter to isolate the closest matches.

---

## Key result

**3 strict Earth-like planets** were found in the data set — all three orbit the star **TRAPPIST-1**, 39 light-years away:

| Planet | Mass (M⊕) | Radius (R⊕) | Temp. (K) | Distance (ly) | Discovery method | Distance to Earth (scaled) |
|--------|-----------|-------------|-----------|---------------|------------------|----------------------------|
| TRAPPIST-1e | 0.620 | 0.919 | 251 | 39 | transit | 0.069 |
| TRAPPIST-1f | 0.680 | 1.042 | 219 | 39 | transit | 0.126 |
| TRAPPIST-1g | 1.341 | 1.132 | 199 | 39 | transit | 0.164 |

All three were discovered by the **transit** method. TRAPPIST-1e is the closest match to Earth.

### Top 15 most Earth-like candidates (ranked by scikit-learn)

| Rank | Planet | Mass (M⊕) | Radius (R⊕) | Temp. (K) | Dist. (ly) | Scaled distance to Earth |
|------|--------|-----------|-------------|-----------|------------|--------------------------|
| 1 | TRAPPIST-1d | 0.410 | 0.773 | 288 | 39 | 0.040 |
| 2 | TRAPPIST-1e | 0.620 | 0.919 | 251 | 39 | 0.069 |
| 3 | LHS 1140 b | 6.642 | 1.435 | 254 | 40.7 | 0.099 |
| 4 | TRAPPIST-1c | 1.379 | 1.054 | 342 | 39 | 0.099 |
| 5 | TRAPPIST-1f | 0.680 | 1.042 | 219 | 39 | 0.126 |
| 6 | TRAPPIST-1g | 1.341 | 1.132 | 199 | 39 | 0.164 |
| 7 | TRAPPIST-1b | 0.849 | 1.087 | 400 | 39 | 0.205 |
| 8 | TOI-270 d | 4.767 | 2.130 | 340 | 73.2 | 0.221 |
| 9 | TOI-732 c | 8.581 | 2.298 | 323 | 72 | 0.238 |
| 10 | Kepler-22 b | 35.911 | 2.376 | 262 | 619.7 | 0.250 |
| 11 | Kepler-138 d | 0.639 | 1.211 | 431 | 218.5 | 0.263 |
| 12 | LTT 1445 A b | 2.193 | 1.379 | 433 | 22.4 | 0.273 |
| 13 | BD+20 594 b | 16.300 | 2.231 | 386 | 496 | 0.282 |
| 14 | Kepler-26 c | 6.200 | 2.724 | 381 | 1,104 | 0.348 |
| 15 | TOI-270 c | 6.134 | 2.421 | 424 | 73.2 | 0.353 |

*(TRAPPIST-1d ranks #1 overall because its temperature matches Earth almost exactly at 288 K, but it falls just outside the strict mass/radius cutoffs used above.)*

---

## What "Earth-like" means (criteria)

A planet is counted as strictly Earth-like only if it satisfies **all three** conditions:

| Attribute | Earth-like range | Rationale |
|-----------|------------------|-----------|
| Mass | 0.5 – 2.0 M⊕ | Rocky, Earth-sized (not a gas giant or sub-Earth) |
| Radius | 0.8 – 1.25 R⊕ | Earth-sized, not an inflated mini-Neptune |
| Equilibrium temperature | 180 – 310 K | Roughly the liquid-water / habitable range |

Earth itself is the reference point: **1 M⊕, 1 R⊕, 288 K**. Planets are ranked by scaled Euclidean distance to this reference after scikit-learn `StandardScaler` normalization (so mass, radius, and temperature contribute comparably despite different units/scales).

> **Assumption:** Temperature is treated as equilibrium temperature. True habitability also depends on atmosphere, stellar activity, tidal locking, etc., none of which are in this data set.

---

## Dataset

- **Source:** `exoplanet-ml/data/raw/exoplanets.csv` — 4,855 confirmed exoplanets with 12 columns:
  `Name, Mass (MJ), Radius (RJ), Period (days), Semi-major axis (AU), Temp. (K), Discovery method, Disc. Year, Distance (ly), Host star mass (M☉), Host star temp. (K), Remarks`
- **Cleaned subset:** `exoplanet-ml/data/processed/exoplanets_cleaned.csv` — 70 planets with no missing values, but planet names were dropped and the smallest is ~3 M⊕, so it contains **no Earth-sized worlds**. The analysis above therefore uses the **raw** file (which keeps names and has 356 planets with Mass, Radius, and Temperature all observed).
- **356-planet detail file:** `exoplanet-ml/src/earth_like_candidates.csv` — all 356 planets that have Mass, Radius, **and** Temperature observed, with every original column plus computed `Mass_Earth` and `Radius_Earth` columns. This is the full candidate pool used for the Earth-like search.
- **Unit conversion:** 1 MJ = 317.8 M⊕; 1 RJ = 11.209 R⊕.

Only rows with observed Mass, Radius, and Temperature are used (no values are fabricated). Missing values in the feature matrix are imputed with the median via scikit-learn's `SimpleImputer`.

---

## How to run

### Prerequisites
Python 3 with the packages in `exoplanet-ml/requirements.txt`:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

### Install
```bash
# from the repo root
cd exoplanet-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### Run the Earth-like finder
```bash
python src/find_earth_like.py
```

Output:
- Count and list of strict Earth-like planets
- Top-N most Earth-like candidates ranked by scaled distance to Earth

---

## Project structure

```
exoplanet-ml/
├── data/
│   ├── raw/exoplanets.csv           # 4,855 planets, 12 columns (with names)
│   └── processed/exoplanets_cleaned.csv  # 70 planets, no missing values (names dropped)
├── notebooks/
│   └── exploration.ipynb            # exploratory data analysis (EDA)
├── src/
│   ├── preprocess.py                # placeholder
│   ├── train.py                     # placeholder
│   ├── evaluate.py                  # placeholder
│   ├── find_earth_like.py           # Earth-like planet finder (scikit-learn)
│   └── earth_like_candidates.csv    # 356 planets with Mass, Radius & Temp observed
├── requirements.txt
└── README.md
```

---

## Limitations

- Only 356 of 4,855 planets have Mass, Radius, **and** Temperature observed; the rest are excluded (no fabrication).
- Temperature is equilibrium temperature — atmospheric composition, greenhouse effects, and stellar activity are not modeled.
- The strict thresholds (0.5–2.0 M⊕, 0.8–1.25 R⊕, 180–310 K) are a chosen heuristic; relaxing them yields more candidates (see the Top-15 list).
- Distance values with uncertainty notation (e.g. `218.5±0.3`) come straight from the source CSV.