"""
Aggregate Kingston 2021 DA census polygons to neighbourhood-level metrics.

Inputs
- `data/raw/administrative/kingston_neighbourhoods.geojson`
- `data/raw/administrative/kingston_da_2021.geojson`

Method
- Load both polygon layers and standardize identifiers / numeric fields.
- Reproject to EPSG:26918 for area calculations.
- Intersect DAs with neighbourhoods and compute area-share weights.
- Allocate DA counts and weights by overlap proportion.
- Aggregate neighbourhood metrics using the requested formulas.

Output
- `data/processed/kingston_neighbourhood_census_2021.csv`

Run
- `python pipeline/build_neighbourhood_census_from_da.py`
"""

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_NEIGH_PATH = Path("data/raw/administrative/kingston_neighbourhoods.geojson")
DEFAULT_DA_PATH = Path("data/raw/administrative/kingston_da_2021.geojson")
DEFAULT_OUTPUT_PATH = Path("data/processed/kingston_neighbourhood_census_2021.csv")
PROJECTED_EPSG = 26918
PCT_COLUMNS = [
    "pct_low_income",
    "pct_seniors",
    "pct_seniors_living_alone",
    "pct_renters",
    "pct_recent_immigrants",
    "pct_visible_minority",
]
SENIOR_COLUMNS = [
    "F65_to_69_years",
    "F70_to_74_years",
    "F75_to_79_years",
    "F80_to_84_years",
    "F85_to_89_years",
    "F90_to_94_years",
    "F95_to_99_years",
    "F100_years_and_over",
]
NUMERIC_NULL_TOKENS = {"", " ", "..", "...", "x", "X", "-", "--"}


def normalize_text(value) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def choose_column(columns: list[str], exact: list[str] | None = None, contains_all: list[list[str] | str] | None = None) -> str | None:
    normalized = {column: normalize_text(column) for column in columns}

    for candidate in exact or []:
        candidate_norm = normalize_text(candidate)
        for column, column_norm in normalized.items():
            if column_norm == candidate_norm:
                return column

    for parts in contains_all or []:
        if isinstance(parts, str):
            parts = [parts]
        normalized_parts = [normalize_text(part) for part in parts]
        for column, column_norm in normalized.items():
            if all(part in column_norm for part in normalized_parts):
                return column

    return None


def coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.strip()
        .replace({token: np.nan for token in NUMERIC_NULL_TOKENS})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_neighbourhoods(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Neighbourhood layer is empty.")
    if gdf.crs is None:
        raise ValueError("Neighbourhood layer has no CRS.")

    name_col = choose_column(
        gdf.columns.tolist(),
        exact=["neigh_name", "HOODNAME", "hoodname"],
        contains_all=[["hood", "name"], ["neigh", "name"]],
    )
    id_col = choose_column(
        gdf.columns.tolist(),
        exact=["neigh_id", "OBJECTID", "objectid"],
        contains_all=[["neigh", "id"], ["object", "id"]],
    )

    if not name_col:
        raise ValueError("Could not detect a neighbourhood name column.")
    if not id_col:
        raise ValueError("Could not detect a neighbourhood id column.")

    output = gdf[[id_col, name_col, "geometry"]].copy()
    output = output.rename(columns={id_col: "neigh_id", name_col: "neigh_name"})
    output["neigh_id"] = output["neigh_id"].astype(str).str.strip()
    output["neigh_name"] = output["neigh_name"].astype(str).str.strip()

    print(f"Neighbourhood file: {path}")
    print(f"  CRS: {gdf.crs}")
    print(f"  id column: {id_col}")
    print(f"  name column: {name_col}")
    print(f"  rows: {len(output)}")
    return output


def detect_da_columns(columns: list[str]) -> dict:
    mapping = {
        "da_id": choose_column(columns, exact=["DAUID", "dauid"], contains_all=[["da", "uid"]]),
        "population": choose_column(columns, exact=["Population_2021"], contains_all=[["population", "2021"]]),
        "owner": choose_column(columns, exact=["Owner"], contains_all=["owner"]),
        "renter": choose_column(columns, exact=["Renter"], contains_all=["renter"]),
        "one_person_households": choose_column(
            columns,
            exact=["Oneperson_households"],
            contains_all=[["oneperson", "households"], ["one", "person", "households"]],
        ),
        "total_households": choose_column(
            columns,
            exact=["Total_Private_households_by_ten"],
            contains_all=[["total", "private", "households", "ten"]],
        ),
        "median_income": choose_column(
            columns,
            exact=["Median_total_income_of_househol"],
            contains_all=[["median", "income", "househol"]],
        ),
        "low_income_pct": choose_column(
            columns,
            exact=["Prevalence_of_low_income_based_"],
            contains_all=[["prevalence", "low", "income"]],
        ),
        "total_immigrant": choose_column(
            columns,
            exact=["Total_Immigrant_status_and_peri"],
            contains_all=[["total", "immigrant", "status"]],
        ),
        "recent_immigrants": choose_column(
            columns,
            exact=["F2016_to_2021"],
            contains_all=[["2016", "2021"]],
        ),
        "seniors_living_alone_direct": choose_column(
            columns,
            contains_all=[["senior", "alone"], ["65", "alone"]],
        ),
        "visible_minority": choose_column(
            columns,
            contains_all=[["visible", "minority"], ["racialized"]],
        ),
    }

    senior_bins = []
    missing_senior_bins = []
    for column_name in SENIOR_COLUMNS:
        if column_name in columns:
            senior_bins.append(column_name)
        else:
            missing_senior_bins.append(column_name)
    mapping["senior_bins"] = senior_bins
    mapping["missing_senior_bins"] = missing_senior_bins
    return mapping


def load_da_geojson(path: Path) -> tuple[gpd.GeoDataFrame, dict, list[str]]:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("DA layer is empty.")
    if gdf.crs is None:
        raise ValueError("DA layer has no CRS.")

    mapping = detect_da_columns(gdf.columns.tolist())
    missing_columns: list[str] = []
    required = ["da_id", "population", "median_income", "low_income_pct"]
    for key in required:
        if not mapping.get(key):
            raise ValueError(f"Missing required DA column for `{key}`.")

    if mapping["missing_senior_bins"]:
        raise ValueError(f"Missing required senior age-bin columns: {mapping['missing_senior_bins']}")

    if not mapping.get("owner"):
        missing_columns.append("Owner")
    if not mapping.get("renter"):
        missing_columns.append("Renter")
    if not mapping.get("one_person_households"):
        missing_columns.append("Oneperson_households")
    if not mapping.get("total_households"):
        missing_columns.append("Total_Private_households_by_ten")
    if not mapping.get("recent_immigrants"):
        missing_columns.append("F2016_to_2021")
    if not mapping.get("total_immigrant"):
        missing_columns.append("Total_Immigrant_status_and_peri")
    if not mapping.get("visible_minority"):
        missing_columns.append("visible minority column")

    output = gdf[[mapping["da_id"], "geometry"]].copy()
    output = output.rename(columns={mapping["da_id"]: "da_id"})
    output["da_id"] = output["da_id"].astype(str).str.strip()

    output["population"] = coerce_numeric(gdf[mapping["population"]])
    output["owner"] = coerce_numeric(gdf[mapping["owner"]]) if mapping.get("owner") else np.nan
    output["renter"] = coerce_numeric(gdf[mapping["renter"]]) if mapping.get("renter") else np.nan
    output["one_person_households"] = (
        coerce_numeric(gdf[mapping["one_person_households"]]) if mapping.get("one_person_households") else np.nan
    )
    output["total_households"] = (
        coerce_numeric(gdf[mapping["total_households"]]) if mapping.get("total_households") else np.nan
    )
    output["median_household_income"] = coerce_numeric(gdf[mapping["median_income"]])
    output["low_income_pct"] = coerce_numeric(gdf[mapping["low_income_pct"]])
    output["recent_immigrants"] = (
        coerce_numeric(gdf[mapping["recent_immigrants"]]) if mapping.get("recent_immigrants") else np.nan
    )
    output["total_immigrant_status"] = (
        coerce_numeric(gdf[mapping["total_immigrant"]]) if mapping.get("total_immigrant") else np.nan
    )
    output["visible_minority_count"] = (
        coerce_numeric(gdf[mapping["visible_minority"]]) if mapping.get("visible_minority") else np.nan
    )
    output["seniors_living_alone_direct"] = (
        coerce_numeric(gdf[mapping["seniors_living_alone_direct"]]) if mapping.get("seniors_living_alone_direct") else np.nan
    )

    output["seniors_count"] = 0.0
    for column_name in mapping["senior_bins"]:
        output["seniors_count"] = output["seniors_count"] + coerce_numeric(gdf[column_name]).fillna(0.0)

    print(f"DA file: {path}")
    print(f"  CRS: {gdf.crs}")
    print(f"  rows: {len(output)}")
    print("  detected columns:")
    for key in [
        "da_id",
        "population",
        "owner",
        "renter",
        "one_person_households",
        "total_households",
        "median_income",
        "low_income_pct",
        "recent_immigrants",
        "total_immigrant",
        "seniors_living_alone_direct",
        "visible_minority",
    ]:
        print(f"    {key}: {mapping.get(key)}")
    if missing_columns:
        print(f"  warnings: missing optional columns -> {missing_columns}")

    return output, mapping, missing_columns


def reproject_layers(neighbourhoods: gpd.GeoDataFrame, da_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    return neighbourhoods.to_crs(epsg=PROJECTED_EPSG), da_gdf.to_crs(epsg=PROJECTED_EPSG)


def build_intersections_area_weighted(
    neighbourhoods_proj: gpd.GeoDataFrame,
    da_proj: gpd.GeoDataFrame,
) -> pd.DataFrame:
    da_base = da_proj.copy()
    da_base["da_area"] = da_base.geometry.area
    if (da_base["da_area"] <= 0).any():
        raise ValueError("DA polygons contain zero-area geometries.")

    intersections = gpd.overlay(
        da_base,
        neighbourhoods_proj[["neigh_id", "neigh_name", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )
    if intersections.empty:
        raise ValueError("No DA/neighbourhood intersections were produced.")

    intersections["intersection_area"] = intersections.geometry.area
    intersections["weight"] = intersections["intersection_area"] / intersections["da_area"]
    intersections = intersections.loc[intersections["weight"] > 0].copy()
    if intersections.empty:
        raise ValueError("All computed DA/neighbourhood weights were zero.")

    area_check = intersections.groupby("da_id")["weight"].sum()
    if (area_check > 1.001).any():
        raise ValueError("Some DAs were allocated more than 100% by area; check geometry validity.")

    return pd.DataFrame(intersections.drop(columns="geometry"))


def allocate_counts_and_weights(intersections: pd.DataFrame) -> pd.DataFrame:
    allocated = intersections.copy()

    alloc_columns = [
        "population",
        "seniors_count",
        "owner",
        "renter",
        "one_person_households",
        "total_households",
        "recent_immigrants",
        "total_immigrant_status",
        "seniors_living_alone_direct",
        "visible_minority_count",
    ]

    for column in alloc_columns:
        allocated[f"allocated_{column}"] = allocated[column] * allocated["weight"]

    allocated["allocated_low_income_weight"] = allocated["population"] * allocated["weight"]
    allocated["allocated_low_income_weighted_pct"] = allocated["allocated_low_income_weight"] * allocated["low_income_pct"]

    income_weight = allocated["total_households"]
    if income_weight.isna().all():
        income_weight = allocated["population"]
        print("Warning: total household counts missing; median income uses population weights.")
    else:
        print("Median income uses household-weighted average approximation.")

    allocated["income_weight_base"] = income_weight
    allocated["allocated_income_weight"] = allocated["income_weight_base"] * allocated["weight"]
    allocated["allocated_income_weighted_value"] = allocated["allocated_income_weight"] * allocated["median_household_income"]

    if allocated["seniors_living_alone_direct"].notna().any():
        print("Using direct seniors-living-alone field.")
    else:
        print(
            "Warning: no direct seniors-living-alone field found; using fallback approximation based on "
            "one-person households share and seniors share."
        )

    if allocated["visible_minority_count"].isna().all():
        print("Warning: no visible minority column found; pct_visible_minority will be NA.")

    return allocated


def safe_ratio(numerator: pd.Series, denominator: pd.Series, multiplier: float = 1.0) -> pd.Series:
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    valid = denominator.notna() & (denominator > 0) & numerator.notna()
    result.loc[valid] = (numerator.loc[valid] / denominator.loc[valid]) * multiplier
    return result


def compute_metrics(
    neighbourhoods: gpd.GeoDataFrame,
    allocated: pd.DataFrame,
) -> pd.DataFrame:
    grouped = allocated.groupby(["neigh_id", "neigh_name"], dropna=False).sum(numeric_only=True)

    output = neighbourhoods[["neigh_id", "neigh_name"]].copy()
    output = output.merge(grouped.reset_index(), on=["neigh_id", "neigh_name"], how="left")

    output["pct_low_income"] = safe_ratio(
        output["allocated_low_income_weighted_pct"],
        output["allocated_low_income_weight"],
    )
    output["pct_seniors"] = safe_ratio(
        output["allocated_seniors_count"],
        output["allocated_population"],
        multiplier=100.0,
    )

    if allocated["seniors_living_alone_direct"].notna().any():
        output["pct_seniors_living_alone"] = safe_ratio(
            output["allocated_seniors_living_alone_direct"],
            output["allocated_population"],
            multiplier=100.0,
        )
    else:
        one_person_share = safe_ratio(output["allocated_one_person_households"], output["allocated_total_households"])
        seniors_share = safe_ratio(output["allocated_seniors_count"], output["allocated_population"])
        output["pct_seniors_living_alone"] = one_person_share * seniors_share * 100.0

    tenure_total = output["allocated_owner"] + output["allocated_renter"]
    output["pct_renters"] = safe_ratio(
        output["allocated_renter"],
        tenure_total,
        multiplier=100.0,
    )

    output["median_household_income"] = safe_ratio(
        output["allocated_income_weighted_value"],
        output["allocated_income_weight"],
    )

    output["pct_recent_immigrants"] = safe_ratio(
        output["allocated_recent_immigrants"],
        output["allocated_population"],
        multiplier=100.0,
    )

    if allocated["visible_minority_count"].isna().all():
        output["pct_visible_minority"] = np.nan
    else:
        output["pct_visible_minority"] = safe_ratio(
            output["allocated_visible_minority_count"],
            output["allocated_population"],
            multiplier=100.0,
        )

    final_columns = [
        "neigh_id",
        "neigh_name",
        "pct_low_income",
        "pct_seniors",
        "pct_seniors_living_alone",
        "pct_renters",
        "median_household_income",
        "pct_recent_immigrants",
        "pct_visible_minority",
    ]
    final = output[final_columns].copy()

    for column in PCT_COLUMNS:
        final[column] = final[column].round(3)
    final["median_household_income"] = final["median_household_income"].round(2)
    return final


def write_output_csv(output: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    print(f"Wrote output: {path}")


def run_qa(
    output: pd.DataFrame,
    neighbourhoods: gpd.GeoDataFrame,
    allocated: pd.DataFrame,
    missing_columns: list[str],
) -> None:
    if len(output) != len(neighbourhoods):
        raise ValueError(f"Expected {len(neighbourhoods)} neighbourhood rows, found {len(output)}.")

    for column in PCT_COLUMNS:
        series = output[column].dropna()
        if ((series < -0.001) | (series > 100.001)).any():
            raise ValueError(f"{column} has values outside [0, 100].")

    pop_by_neigh = allocated.groupby("neigh_id")["allocated_population"].sum(min_count=1)
    zero_pop = output.loc[output["neigh_id"].isin(pop_by_neigh[pop_by_neigh <= 0].index), "neigh_name"].tolist()
    if zero_pop:
        print(f"Warning: neighbourhoods with zero allocated population -> {zero_pop}")

    missing_metrics = [column for column in output.columns[2:] if output[column].isna().all()]
    if missing_metrics:
        print(f"Warning: fully missing output metrics -> {missing_metrics}")

    print("QA summary:")
    print(f"  intersections: {len(allocated)}")
    print(f"  neighbourhood rows: {len(output)}")
    print(f"  neighbourhoods with zero allocated population: {len(zero_pop)}")
    print(f"  missing optional source columns: {missing_columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Kingston 2021 DA census polygons to neighbourhood metrics.")
    parser.add_argument("--neigh_path", type=Path, default=DEFAULT_NEIGH_PATH, help="Neighbourhood GeoJSON path.")
    parser.add_argument("--da_path", type=Path, default=DEFAULT_DA_PATH, help="DA GeoJSON path.")
    parser.add_argument("--output_path", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output CSV path.")
    args = parser.parse_args()

    neighbourhoods = load_neighbourhoods(args.neigh_path)
    da_gdf, mapping, missing_columns = load_da_geojson(args.da_path)

    neighbourhoods_proj, da_proj = reproject_layers(neighbourhoods, da_gdf)
    intersections = build_intersections_area_weighted(neighbourhoods_proj, da_proj)
    allocated = allocate_counts_and_weights(intersections)
    output = compute_metrics(neighbourhoods, allocated)

    run_qa(output, neighbourhoods, allocated, missing_columns)
    write_output_csv(output, args.output_path)

    _ = mapping


if __name__ == "__main__":
    main()
