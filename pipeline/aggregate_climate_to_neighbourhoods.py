import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping


DATE_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass(frozen=True)
class VariableSpec:
    key: str
    folder: str
    file_glob: str
    yearly_metric: str
    final_metric: str
    yearly_agg: str
    final_agg: str


VARIABLE_SPECS = (
    VariableSpec(
        key="lst_day",
        folder="daytime lst",
        file_glob="MOD11A2.061_LST_Day_1km_*.tif",
        yearly_metric="lst_day_mean_summer",
        final_metric="lst_day_mean_summer",
        yearly_agg="mean",
        final_agg="mean",
    ),
    VariableSpec(
        key="lst_night",
        folder="nighttime lst",
        file_glob="MOD11A2.061_LST_Night_1km_*.tif",
        yearly_metric="lst_night_mean_summer",
        final_metric="lst_night_mean_summer",
        yearly_agg="mean",
        final_agg="mean",
    ),
    VariableSpec(
        key="clear_sky_days",
        folder="clear sky days",
        file_glob="MOD11A2.061_Clear_sky_days_*.tif",
        yearly_metric="clear_sky_days_total_summer",
        final_metric="clear_sky_days_total",
        yearly_agg="sum",
        final_agg="sum",
    ),
    VariableSpec(
        key="clear_sky_nights",
        folder="clear sky nights",
        file_glob="MOD11A2.061_Clear_sky_nights_*.tif",
        yearly_metric="clear_sky_nights_total_summer",
        final_metric="clear_sky_nights_total",
        yearly_agg="sum",
        final_agg="sum",
    ),
    VariableSpec(
        key="qc_day",
        folder="qc day",
        file_glob="MOD11A2.061_QC_Day_*.tif",
        yearly_metric="pct_pixels_good_qc_day_mean_summer",
        final_metric="pct_pixels_good_qc_day_mean_summer",
        yearly_agg="mean",
        final_agg="mean",
    ),
    VariableSpec(
        key="qc_night",
        folder="qc night",
        file_glob="MOD11A2.061_QC_Night_*.tif",
        yearly_metric="pct_pixels_good_qc_night_mean_summer",
        final_metric="pct_pixels_good_qc_night_mean_summer",
        yearly_agg="mean",
        final_agg="mean",
    ),
)


def parse_date_from_name(path: Path) -> pd.Timestamp:
    match = DATE_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"Could not parse YYYY-MM-DD from filename: {path}")
    return pd.Timestamp(
        year=int(match.group(1)),
        month=int(match.group(2)),
        day=int(match.group(3)),
    )


def discover_rasters(
    climate_dir: Path,
    spec: VariableSpec,
    start_year: int,
    end_year: int,
    summer_months: set[int],
) -> list[tuple[pd.Timestamp, Path]]:
    folder = climate_dir / spec.folder
    files: list[tuple[pd.Timestamp, Path]] = []

    for path in sorted(folder.glob(spec.file_glob)):
        date = parse_date_from_name(path)
        if date.year < start_year or date.year > end_year:
            continue
        if date.month not in summer_months:
            continue
        files.append((date, path))

    if not files:
        raise ValueError(f"No rasters found for '{spec.key}' in {folder}")

    return files


def load_neighbourhoods(path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    required = {"neigh_id", "neigh_name", "geometry"}
    missing = required.difference(gdf.columns)
    if missing:
        raise ValueError(f"Neighbourhood file missing required columns: {sorted(missing)}")
    if gdf.crs is None:
        raise ValueError("Neighbourhood file has no CRS.")
    if gdf.empty:
        raise ValueError("Neighbourhood file is empty.")
    return gdf[["neigh_id", "neigh_name", "geometry"]].copy()


def build_mask_cache(
    src: rasterio.io.DatasetReader,
    neighbourhoods: gpd.GeoDataFrame,
    cache: dict[tuple, list[np.ndarray]],
) -> list[np.ndarray]:
    key = (
        src.width,
        src.height,
        src.transform.a,
        src.transform.b,
        src.transform.c,
        src.transform.d,
        src.transform.e,
        src.transform.f,
        str(src.crs),
    )
    if key in cache:
        return cache[key]

    masks = []
    for geom in neighbourhoods.geometry:
        masks.append(
            geometry_mask(
                [mapping(geom)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
                all_touched=True,
            )
        )
    cache[key] = masks
    return masks


def aggregate_lst(array: np.ma.MaskedArray, masks: list[np.ndarray]) -> list[float]:
    values_c = array.astype("float64") * 0.02 - 273.15
    outputs: list[float] = []

    for zone_mask in masks:
        zone_values = values_c[zone_mask]
        outputs.append(float(zone_values.mean()) if zone_values.count() else np.nan)

    return outputs


def aggregate_sum(array: np.ma.MaskedArray, masks: list[np.ndarray]) -> list[float]:
    values = array.astype("float64")
    outputs: list[float] = []

    for zone_mask in masks:
        zone_values = values[zone_mask]
        outputs.append(float(zone_values.mean()) if zone_values.count() else np.nan)

    return outputs


def aggregate_qc(array: np.ma.MaskedArray, masks: list[np.ndarray]) -> list[float]:
    filled = array.filled(0).astype("uint8")
    valid_global = ~np.ma.getmaskarray(array)
    mandatory_qa = np.bitwise_and(filled, 0b11)
    good_global = mandatory_qa <= 1
    outputs: list[float] = []

    for zone_mask in masks:
        valid_zone = valid_global & zone_mask
        valid_count = int(valid_zone.sum())
        if valid_count == 0:
            outputs.append(np.nan)
            continue
        good_count = int((good_global & valid_zone).sum())
        outputs.append((good_count / valid_count) * 100.0)

    return outputs


def reduce_values(values: list[float], mode: str) -> float:
    valid = [float(v) for v in values if pd.notna(v)]
    if not valid:
        return np.nan
    if mode == "mean":
        return float(np.mean(valid))
    if mode == "sum":
        return float(np.sum(valid))
    raise ValueError(f"Unsupported reduction mode: {mode}")


def aggregate_variable(
    neighbourhoods: gpd.GeoDataFrame,
    rasters: list[tuple[pd.Timestamp, Path]],
    spec: VariableSpec,
) -> dict[tuple[str, int], list[float]]:
    outputs: dict[tuple[str, int], list[float]] = defaultdict(list)
    mask_cache: dict[tuple, list[np.ndarray]] = {}
    raster_crs = None
    local_neighbourhoods = neighbourhoods

    for date, path in rasters:
        with rasterio.open(path) as src:
            if raster_crs is None:
                raster_crs = src.crs
                if raster_crs is None:
                    raise ValueError(f"Raster has no CRS: {path}")
                if local_neighbourhoods.crs != raster_crs:
                    local_neighbourhoods = local_neighbourhoods.to_crs(raster_crs)
            masks = build_mask_cache(src, local_neighbourhoods, mask_cache)
            band = src.read(1, masked=True)

        if spec.key.startswith("lst_"):
            zonal_values = aggregate_lst(band, masks)
        elif spec.key.startswith("clear_sky_"):
            zonal_values = aggregate_sum(band, masks)
        elif spec.key.startswith("qc_"):
            zonal_values = aggregate_qc(band, masks)
        else:
            raise ValueError(f"Unhandled variable key: {spec.key}")

        for idx, value in enumerate(zonal_values):
            neigh_id = str(local_neighbourhoods.iloc[idx]["neigh_id"])
            outputs[(neigh_id, int(date.year))].append(value)

    return outputs


def build_outputs(
    neighbourhoods: gpd.GeoDataFrame,
    per_variable_rasters: dict[str, list[tuple[pd.Timestamp, Path]]],
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = list(range(start_year, end_year + 1))
    per_year = neighbourhoods[["neigh_id", "neigh_name"]].copy()
    per_year = per_year.loc[per_year.index.repeat(len(years))].reset_index(drop=True)
    per_year["year"] = np.tile(years, len(neighbourhoods))

    aggregated_values: dict[str, dict[tuple[str, int], list[float]]] = {}
    for spec in VARIABLE_SPECS:
        aggregated_values[spec.key] = aggregate_variable(neighbourhoods, per_variable_rasters[spec.key], spec)
        yearly_column = []
        for _, row in per_year.iterrows():
            values = aggregated_values[spec.key].get((row["neigh_id"], int(row["year"])), [])
            yearly_column.append(reduce_values(values, spec.yearly_agg))
        per_year[spec.yearly_metric] = yearly_column

    suffix = f"{start_year}_{end_year}"
    final = neighbourhoods[["neigh_id", "neigh_name"]].copy()

    for spec in VARIABLE_SPECS:
        column_name = f"{spec.final_metric}_{suffix}"
        final_values = []
        for neigh_id in final["neigh_id"]:
            yearly_values = []
            for year in years:
                values = aggregated_values[spec.key].get((str(neigh_id), year), [])
                yearly_values.append(reduce_values(values, spec.yearly_agg))
            final_values.append(reduce_values(yearly_values, spec.final_agg))
        final[column_name] = final_values

    return per_year, final


def round_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    rounded = df.copy()
    for column in rounded.columns:
        if column.startswith("lst_"):
            rounded[column] = rounded[column].round(3)
        elif column.startswith("clear_sky_"):
            rounded[column] = rounded[column].round(2)
        elif column.startswith("pct_pixels_good_qc_"):
            rounded[column] = rounded[column].round(2)
    return rounded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate MODIS summer climate rasters to Kingston neighbourhood polygons."
    )
    parser.add_argument(
        "--neighbourhoods",
        type=Path,
        default=Path("data/processed/kingston_neighbourhoods.gpkg"),
        help="Path to canonical neighbourhood polygons.",
    )
    parser.add_argument(
        "--climate-dir",
        type=Path,
        default=Path("data/raw/climate"),
        help="Directory containing climate raster folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/neighbourhood_climate_features.csv"),
        help="Path for the multi-year aggregated output table.",
    )
    parser.add_argument(
        "--output-yearly",
        type=Path,
        default=Path("data/processed/neighbourhood_climate_features_by_year.csv"),
        help="Path for the per-year output table.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="First year to include.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last year to include.",
    )
    parser.add_argument(
        "--summer-months",
        type=int,
        nargs="+",
        default=[6, 7, 8],
        help="Calendar months to treat as summer.",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year")

    summer_months = set(args.summer_months)
    if not summer_months:
        raise ValueError("--summer-months must include at least one month")

    neighbourhoods = load_neighbourhoods(args.neighbourhoods)

    per_variable_rasters = {}
    for spec in VARIABLE_SPECS:
        per_variable_rasters[spec.key] = discover_rasters(
            climate_dir=args.climate_dir,
            spec=spec,
            start_year=args.start_year,
            end_year=args.end_year,
            summer_months=summer_months,
        )

    yearly_df, final_df = build_outputs(
        neighbourhoods=neighbourhoods,
        per_variable_rasters=per_variable_rasters,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    yearly_df = round_feature_columns(yearly_df)
    final_df = round_feature_columns(final_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_yearly.parent.mkdir(parents=True, exist_ok=True)
    yearly_df.to_csv(args.output_yearly, index=False)
    final_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
