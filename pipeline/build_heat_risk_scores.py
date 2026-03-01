import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency `geopandas`. Install it with `pip install geopandas`.") from exc

try:
    import folium
    from branca.colormap import LinearColormap
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependencies `folium`/`branca`. Install them with `pip install folium branca`.") from exc


DEFAULT_WEIGHTS_PATH = Path("src/config/weights.yml")
DEFAULT_OUT_SCORES_PATH = Path("data/processed/neighbourhood_heat_risk_scores.csv")
DEFAULT_OUT_MAP_PATH = Path("outputs/maps/kingston_equity_adjusted_heat_risk.html")


def log(message: str) -> None:
    print(message)


def normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    for token in ("_", "-", "/", "(", ")", "%"):
        text = text.replace(token, " ")
    return " ".join(text.split())


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Weights config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Weights config must be a YAML mapping.")
    return data


def discover_climate_path(explicit_path: Path | None) -> Path:
    if explicit_path:
        return explicit_path
    parquet_path = Path("data/processed/neighbourhood_climate_features.parquet")
    csv_path = Path("data/processed/neighbourhood_climate_features.csv")
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError("Could not find a climate features table in data/processed.")


def discover_ses_path(explicit_path: Path | None) -> Path:
    if explicit_path:
        return explicit_path
    preferred = Path("data/processed/neighbourhood_ses_features_2021.csv")
    fallback = Path("data/processed/kingston_neighbourhood_census_2021.csv")
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find an SES features table in data/processed.")


def discover_geo_path(explicit_path: Path | None) -> Path:
    if explicit_path:
        return explicit_path
    candidates = [
        Path("data/processed/kingston_neighbourhoods.gpkg"),
        Path("data/raw/kingston_neighbourhoods.geojson"),
        Path("data/processed/kingston_neighbourhoods_canonical.geojson"),
        Path("data/raw/administrative/kingston_neighbourhoods.geojson"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find a neighbourhood geometry file.")


def load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table format: {path}")
    if "neigh_id" not in df.columns:
        raise ValueError(f"`neigh_id` is required in {path}.")
    df = df.copy()
    df["neigh_id"] = df["neigh_id"].astype(str).str.strip()
    if "neigh_name" in df.columns:
        df["neigh_name"] = df["neigh_name"].astype(str).str.strip()
    return df


def load_geometry(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Geometry layer is empty: {path}")

    columns = gdf.columns.tolist()
    id_col = "neigh_id" if "neigh_id" in columns else None
    name_col = "neigh_name" if "neigh_name" in columns else None

    if id_col is None:
        for candidate in ("OBJECTID", "objectid", "id"):
            if candidate in columns:
                id_col = candidate
                break
    if name_col is None:
        for candidate in ("HOODNAME", "hoodname", "name"):
            if candidate in columns:
                name_col = candidate
                break

    if id_col is None:
        raise ValueError(f"Could not detect `neigh_id` in geometry layer {path}.")
    if name_col is None:
        raise ValueError(f"Could not detect `neigh_name` in geometry layer {path}.")

    output = gdf[[id_col, name_col, "geometry"]].copy()
    output = output.rename(columns={id_col: "neigh_id", name_col: "neigh_name"})
    output["neigh_id"] = output["neigh_id"].astype(str).str.strip()
    output["neigh_name"] = output["neigh_name"].astype(str).str.strip()
    return output


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized_map = {column: normalize_text(column) for column in columns}
    for alias in aliases:
        alias_norm = normalize_text(alias)
        for column, column_norm in normalized_map.items():
            if column_norm == alias_norm:
                return column
        for column, column_norm in normalized_map.items():
            if alias_norm in column_norm:
                return column
    return None


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({"": np.nan, "..": np.nan, "...": np.nan, "x": np.nan, "X": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_minmax(series: pd.Series, invert: bool = False) -> tuple[pd.Series, pd.Series]:
    numeric = to_numeric(series)
    original = numeric.copy()
    fill_value = numeric.median(skipna=True)
    if pd.isna(fill_value):
        raise ValueError("Cannot normalize a column with all values missing.")
    scoring = numeric.fillna(fill_value)

    min_value = scoring.min()
    max_value = scoring.max()
    if pd.isna(min_value) or pd.isna(max_value):
        raise ValueError("Cannot normalize a column with invalid numeric values.")

    if np.isclose(max_value, min_value):
        scored_norm = pd.Series(0.5, index=scoring.index, dtype="float64")
        output_norm = pd.Series(np.where(original.notna(), 0.5, np.nan), index=series.index, dtype="float64")
    else:
        scored_norm = (scoring - min_value) / (max_value - min_value)
        output_norm = (original - min_value) / (max_value - min_value)

    if invert:
        scored_norm = 1.0 - scored_norm
        output_norm = 1.0 - output_norm

    return output_norm.astype("float64"), scored_norm.astype("float64")


def compute_weighted_score(scored_norms: dict[str, pd.Series], weights: dict[str, float]) -> pd.Series:
    available = {key: float(weight) for key, weight in weights.items() if key in scored_norms and float(weight) > 0}
    if not available:
        raise ValueError("No available indicators for weighted score.")
    weight_sum = sum(available.values())
    if weight_sum <= 0:
        raise ValueError("Indicator weights must sum to a positive value.")

    score = pd.Series(0.0, index=next(iter(scored_norms.values())).index, dtype="float64")
    for indicator, weight in available.items():
        score = score + scored_norms[indicator] * (weight / weight_sum)
    return score


def build_map(
    gdf: gpd.GeoDataFrame,
    out_map_path: Path,
) -> None:
    map_gdf = gdf.to_crs(epsg=4326).copy()
    centroid_proj = gdf.to_crs(epsg=26918).geometry.centroid
    centroid_wgs84 = gpd.GeoSeries(centroid_proj, crs=26918).to_crs(epsg=4326)
    center_lat = float(centroid_wgs84.y.mean())
    center_lon = float(centroid_wgs84.x.mean())

    risk_values = map_gdf["equity_adjusted_risk"].fillna(0.0)
    vmin = float(risk_values.min())
    vmax = float(risk_values.max())
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    colormap = LinearColormap(
        colors=["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"],
        vmin=vmin,
        vmax=vmax,
    )
    colormap.caption = "Equity-Adjusted Heat Risk (0-1)"

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

    style_function = lambda feature: {
        "fillColor": colormap(feature["properties"].get("equity_adjusted_risk", 0.0) or 0.0),
        "color": "#444444",
        "weight": 1,
        "fillOpacity": 0.75,
    }

    tooltip = folium.GeoJsonTooltip(
        fields=[
            "neigh_name",
            "equity_adjusted_risk",
            "baseline_heat_risk",
            "hazard_score",
            "vulnerability_score",
            "rank_equity",
        ],
        aliases=[
            "Neighbourhood",
            "Equity-Adjusted Risk",
            "Baseline Risk",
            "Hazard Score",
            "Vulnerability Score",
            "Equity Rank",
        ],
        localize=True,
        sticky=False,
    )

    folium.GeoJson(
        data=map_gdf.to_json(),
        style_function=style_function,
        tooltip=tooltip,
    ).add_to(m)

    colormap.add_to(m)
    out_map_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_map_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Kingston heat risk scores and an equity-adjusted choropleth map.")
    parser.add_argument("--climate_path", type=Path, default=None)
    parser.add_argument("--ses_path", type=Path, default=None)
    parser.add_argument("--geo_path", type=Path, default=None)
    parser.add_argument("--weights_path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--out_scores_path", type=Path, default=DEFAULT_OUT_SCORES_PATH)
    parser.add_argument("--out_map_path", type=Path, default=DEFAULT_OUT_MAP_PATH)
    args = parser.parse_args()

    climate_path = discover_climate_path(args.climate_path)
    ses_path = discover_ses_path(args.ses_path)
    geo_path = discover_geo_path(args.geo_path)
    weights_cfg = load_yaml(args.weights_path)

    climate_df = load_table(climate_path)
    ses_df = load_table(ses_path)
    geo_gdf = load_geometry(geo_path)

    log(f"Climate table: {climate_path}")
    log(f"SES table: {ses_path}")
    log(f"Geometry: {geo_path}")
    log(f"Neighbourhood geometry rows: {len(geo_gdf)}")

    climate_lookup = {
        "lst_day": ["lst_day_mean_summer_2015_2025", "lst_day_mean_summer", "lst day mean summer"],
        "lst_night": ["lst_night_mean_summer_2015_2025", "lst_night_mean_summer", "lst night mean summer"],
        "population_density": ["population_density", "population density", "pop_density"],
    }
    ses_lookup = {
        "pct_low_income": ["pct_low_income", "low_income_pct", "prevalence_of_low_income"],
        "pct_seniors": ["pct_seniors", "seniors_pct"],
        "pct_seniors_living_alone": ["pct_seniors_living_alone", "seniors_living_alone_pct"],
        "pct_renters": ["pct_renters", "renters_pct"],
        "median_household_income": ["median_household_income", "avg_household_income", "average_household_income"],
        "pct_recent_immigrants": ["pct_recent_immigrants", "recent_immigrants_pct"],
        "pct_visible_minority": ["pct_visible_minority", "visible_minority_pct", "racialized_pct"],
    }

    climate_detected = {key: find_column(climate_df.columns.tolist(), aliases) for key, aliases in climate_lookup.items()}
    ses_detected = {key: find_column(ses_df.columns.tolist(), aliases) for key, aliases in ses_lookup.items()}

    log(f"Detected climate columns: {climate_detected}")
    log(f"Detected SES columns: {ses_detected}")

    merged = geo_gdf[["neigh_id", "neigh_name", "geometry"]].copy()
    merged = merged.merge(
        climate_df.drop(columns=[column for column in ["neigh_name"] if column in climate_df.columns]),
        on="neigh_id",
        how="left",
    )
    merged = merged.merge(
        ses_df.drop(columns=[column for column in ["neigh_name"] if column in ses_df.columns]),
        on="neigh_id",
        how="left",
        suffixes=("", "_ses"),
    )

    climate_feature_cols = [column for column in climate_df.columns if column != "neigh_id"]
    ses_feature_cols = [column for column in ses_df.columns if column != "neigh_id"]
    matched_climate = int(merged[climate_feature_cols].notna().any(axis=1).sum()) if climate_feature_cols else 0
    matched_ses = int(merged[ses_feature_cols].notna().any(axis=1).sum()) if ses_feature_cols else 0
    log(f"Join success: climate {matched_climate}/{len(merged)}, ses {matched_ses}/{len(merged)}")

    protective_indicators = set(weights_cfg.get("protective_indicators", []))
    norm_output_columns: dict[str, pd.Series] = {}
    hazard_scored: dict[str, pd.Series] = {}
    vulnerability_scored: dict[str, pd.Series] = {}
    exposure_scored: dict[str, pd.Series] = {}

    hazard_weights = {key: float(value) for key, value in (weights_cfg.get("hazard", {}) or {}).items()}
    vulnerability_weights = {key: float(value) for key, value in (weights_cfg.get("vulnerability", {}) or {}).items()}
    exposure_cfg = weights_cfg.get("exposure", {}) or {}
    exposure_enabled = bool(exposure_cfg.get("enabled", False))
    exposure_weights = {key: float(value) for key, value in (exposure_cfg.get("indicators", {}) or {}).items()}

    missing_hazard = []
    for indicator, weight in hazard_weights.items():
        if weight <= 0:
            continue
        source_col = climate_detected.get(indicator) or ses_detected.get(indicator)
        if not source_col:
            missing_hazard.append(indicator)
            continue
        output_norm, scored_norm = normalize_minmax(merged[source_col], invert=indicator in protective_indicators)
        norm_output_columns[f"norm_{indicator}"] = output_norm
        hazard_scored[indicator] = scored_norm

    missing_vulnerability = []
    for indicator, weight in vulnerability_weights.items():
        if weight <= 0:
            continue
        source_col = ses_detected.get(indicator) or climate_detected.get(indicator)
        if not source_col:
            missing_vulnerability.append(indicator)
            continue
        output_norm, scored_norm = normalize_minmax(merged[source_col], invert=indicator in protective_indicators)
        norm_output_columns[f"norm_{indicator}"] = output_norm
        vulnerability_scored[indicator] = scored_norm

    missing_exposure = []
    if exposure_enabled:
        for indicator in exposure_weights:
            if exposure_weights[indicator] <= 0:
                continue
            source_col = climate_detected.get(indicator) or ses_detected.get(indicator)
            if not source_col:
                missing_exposure.append(indicator)
                continue
            output_norm, scored_norm = normalize_minmax(merged[source_col], invert=indicator in protective_indicators)
            norm_output_columns[f"norm_{indicator}"] = output_norm
            exposure_scored[indicator] = scored_norm

    if missing_hazard:
        log(f"Missing hazard indicators: {missing_hazard}")
    if missing_vulnerability:
        log(f"Missing vulnerability indicators: {missing_vulnerability}")
    if exposure_enabled and missing_exposure:
        log(f"Missing exposure indicators: {missing_exposure}")

    if not hazard_scored:
        raise ValueError("No usable hazard indicators were found. Need at least one hazard indicator.")
    if not vulnerability_scored:
        raise ValueError("No usable vulnerability indicators were found. Need at least one vulnerability indicator.")

    merged["hazard_score"] = compute_weighted_score(hazard_scored, hazard_weights)
    merged["vulnerability_score"] = compute_weighted_score(vulnerability_scored, vulnerability_weights)

    if exposure_enabled and exposure_scored:
        merged["exposure_score"] = compute_weighted_score(exposure_scored, exposure_weights)
    else:
        merged["exposure_score"] = 1.0
        log("Exposure indicators unavailable or disabled; using neutral exposure_score = 1.0.")

    merged["baseline_heat_risk"] = merged["hazard_score"] * merged["exposure_score"]
    merged["equity_adjusted_risk"] = merged["baseline_heat_risk"] * merged["vulnerability_score"]
    merged["rank_baseline"] = merged["baseline_heat_risk"].rank(method="dense", ascending=False).astype(int)
    merged["rank_equity"] = merged["equity_adjusted_risk"].rank(method="dense", ascending=False).astype(int)

    for column_name, series in norm_output_columns.items():
        merged[column_name] = series
        log(f"Missingness {column_name}: {int(series.isna().sum())}/{len(series)}")

    component_columns = sorted(norm_output_columns.keys())
    score_columns = [
        "neigh_id",
        "neigh_name",
        "hazard_score",
        "vulnerability_score",
        "exposure_score",
        "baseline_heat_risk",
        "equity_adjusted_risk",
        "rank_baseline",
        "rank_equity",
    ] + component_columns

    scores_df = merged[score_columns].copy()
    scores_df = scores_df.sort_values(["rank_equity", "neigh_name"]).reset_index(drop=True)
    institutional_df = scores_df[
        ["neigh_id", "neigh_name", "baseline_heat_risk", "equity_adjusted_risk", "rank_baseline", "rank_equity"]
    ].copy()

    args.out_scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(args.out_scores_path, index=False)
    institutional_path = args.out_scores_path.parent / "neighbourhood_heat_risk_scores_institutional.csv"
    institutional_df.to_csv(institutional_path, index=False)

    build_map(
        merged[
            [
                "neigh_id",
                "neigh_name",
                "hazard_score",
                "vulnerability_score",
                "baseline_heat_risk",
                "equity_adjusted_risk",
                "rank_equity",
                "geometry",
            ]
        ].copy(),
        args.out_map_path,
    )

    log(f"Scored neighbourhoods: {len(scores_df)}")
    top_10 = scores_df.nsmallest(10, "rank_equity")[["neigh_name", "equity_adjusted_risk"]]
    log("Top 10 neighbourhoods by equity-adjusted risk:")
    for row in top_10.itertuples(index=False):
        log(f"  {row.neigh_name}: {row.equity_adjusted_risk:.4f}")


if __name__ == "__main__":
    main()
