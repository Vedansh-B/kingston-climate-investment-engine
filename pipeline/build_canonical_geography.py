import argparse
import json
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import make_valid


def slugify(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text or "unknown"


def make_unique(values: list[str]) -> list[str]:
    seen = {}
    output = []
    for value in values:
        count = seen.get(value, 0)
        if count == 0:
            output.append(value)
        else:
            output.append(f"{value}-{count + 1}")
        seen[value] = count + 1
    return output


def extract_multipolygon(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = []
        for item in geom.geoms:
            if isinstance(item, Polygon):
                polys.append(item)
            elif isinstance(item, MultiPolygon):
                polys.extend(item.geoms)
        if polys:
            return MultiPolygon(polys)
    return None


def parse_source_date(series: pd.Series) -> str | None:
    if series.empty:
        return None

    non_null = series.dropna()
    if non_null.empty:
        return None

    sample = non_null.iloc[0]
    parsed = None

    if isinstance(sample, (int, float)):
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if numeric.empty:
            return None
        max_value = float(numeric.max())
        unit = "ms" if max_value > 1e11 else "s"
        parsed = pd.to_datetime(max_value, unit=unit, utc=True, errors="coerce")
    else:
        parsed_values = pd.to_datetime(non_null, utc=True, errors="coerce")
        if parsed_values.notna().any():
            parsed = parsed_values.max()

    if parsed is None or pd.isna(parsed):
        return None

    return parsed.date().isoformat()


def choose_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        match = lower_map.get(candidate.lower())
        if match:
            return match
    return None


def build_canonical(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("Input geography is empty.")

    if gdf.crs is None:
        raise ValueError("Input geography has no CRS. Expected EPSG:4326.")

    crs_epsg = gdf.crs.to_epsg()
    if crs_epsg is None:
        raise ValueError("Could not resolve CRS EPSG code from input geometry.")

    name_col = choose_column(gdf.columns.tolist(), ["neigh_name", "hoodname", "name", "neighbourhood", "neighbourhood"])
    if not name_col:
        raise ValueError("Could not find a neighbourhood name column.")

    source_date_col = choose_column(
        gdf.columns.tolist(),
        ["source_date", "last_edited_date", "created_date", "lastupdate", "updated_at"],
    )

    id_col = choose_column(gdf.columns.tolist(), ["neigh_id", "objectid", "globalid", "id"])

    canonical = gdf[[name_col, "geometry"]].copy()
    canonical["neigh_name"] = canonical[name_col].astype(str).str.strip()
    canonical = canonical.drop(columns=[name_col])
    canonical = canonical[canonical["neigh_name"].notna() & (canonical["neigh_name"] != "")].copy()

    canonical["geometry"] = canonical.geometry.apply(make_valid).apply(extract_multipolygon)
    canonical = canonical[canonical.geometry.notna() & ~canonical.geometry.is_empty].copy()

    raw_slugs = canonical["neigh_name"].apply(slugify).tolist()
    canonical["neigh_slug"] = make_unique(raw_slugs)

    if id_col:
        id_series = gdf.loc[canonical.index, id_col].astype(str).str.strip()
        if id_series.notna().all() and id_series.ne("").all() and not id_series.duplicated().any():
            canonical["neigh_id"] = id_series
        else:
            canonical["neigh_id"] = [f"KNG-{slug.upper()}" for slug in canonical["neigh_slug"]]
    else:
        canonical["neigh_id"] = [f"KNG-{slug.upper()}" for slug in canonical["neigh_slug"]]

    if canonical["neigh_id"].duplicated().any() or canonical["neigh_id"].isna().any():
        raise ValueError("neigh_id contains duplicates or nulls after normalization.")

    if canonical["neigh_slug"].duplicated().any() or canonical["neigh_slug"].isna().any():
        raise ValueError("neigh_slug contains duplicates or nulls after normalization.")

    source_date = parse_source_date(gdf[source_date_col]) if source_date_col else None
    canonical["source"] = "City of Kingston Open Data"
    canonical["source_date"] = source_date if source_date else "unknown"
    canonical["crs_epsg"] = int(crs_epsg)

    analysis_epsg = 32618
    projected = canonical.to_crs(epsg=analysis_epsg)
    canonical["area_km2"] = (projected.geometry.area / 1_000_000).round(6)

    centroid_projected = projected.geometry.centroid
    centroid_wgs84 = gpd.GeoSeries(centroid_projected, crs=projected.crs).to_crs(epsg=4326)
    canonical["centroid_lon"] = centroid_wgs84.x.round(6)
    canonical["centroid_lat"] = centroid_wgs84.y.round(6)

    canonical = canonical[
        [
            "neigh_id",
            "neigh_name",
            "neigh_slug",
            "source",
            "source_date",
            "crs_epsg",
            "area_km2",
            "centroid_lon",
            "centroid_lat",
            "geometry",
        ]
    ].sort_values("neigh_id").reset_index(drop=True)

    if not canonical.geometry.is_valid.all():
        raise ValueError("Geometry validation failed after normalization.")

    canonical_path = output_dir / "kingston_neighbourhoods.gpkg"
    projected_path = output_dir / "kingston_neighbourhoods_utm18.gpkg"
    metadata_path = output_dir / "kingston_neighbourhoods_base_metadata.json"

    canonical.to_file(canonical_path, layer="neighbourhoods_base", driver="GPKG")
    projected.to_file(projected_path, layer="neighbourhoods_base_utm18", driver="GPKG")

    metadata = {
        "feature_count": int(len(canonical)),
        "crs_epsg": int(crs_epsg),
        "analysis_crs_epsg": analysis_epsg,
        "total_area_km2": float(canonical["area_km2"].sum()),
        "min_area_km2": float(canonical["area_km2"].min()),
        "max_area_km2": float(canonical["area_km2"].max()),
        "source": "City of Kingston Open Data",
        "source_date": canonical["source_date"].iloc[0],
        "output_files": {
            "canonical": str(canonical_path.as_posix()),
            "projected": str(projected_path.as_posix()),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical Kingston neighbourhood base geography.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/kingston_neighbourhoods.geojson"),
        help="Path to raw neighbourhood boundary GeoJSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for canonical outputs.",
    )
    args = parser.parse_args()

    build_canonical(args.input, args.output_dir)


if __name__ == "__main__":
    main()

