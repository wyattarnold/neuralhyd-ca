"""Compute area-weighted BasinATLAS static attributes for each PourPtID.

Reads the GIS intersect table and calculates weighted averages (or spatial
majority for class attributes) per watershed. Output goes to
data/training/static/<target>/Physical_Attributes_<TARGET>.csv
(or data/eval/static/<target>/... when scope="eval").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.paths import (
    BASIN_ATLAS_INPUT,
    BASIN_ATLAS_OUTPUT,
    WATERSHED_GEOMETRY,
    get_target_paths,
    get_eval_target_paths,
)

GLC_PERCENT_ATTRS = {
    f"glc_pc_u{i:02d}": {
        "description": f"Land-cover class {i:02d} percent upstream of pour point",
        "units": "percent",
    }
    for i in range(1, 23)
}

PNV_PERCENT_ATTRS = {
    f"pnv_pc_u{i:02d}": {
        "description": f"Potential natural vegetation class {i:02d} percent upstream of pour point",
        "units": "percent",
    }
    for i in range(1, 16)
}

BASIN_ATLAS_STATIC_ATTR = {
    "ria_ha_usu": {"description": "River Area (ha) sum upstream of pour point", "units": "hectares"},
    "riv_tc_usu": {"description": "River Volume (thousand m3) sum upstream of pour point", "units": "thousand cubic meters"},
    "DIST_SINK": {"description": "Distance from subbasin outlet to terminal sink", "units": "kilometers"},
    "DIST_MAIN": {"description": "Distance from subbasin outlet to main basin outlet", "units": "kilometers"},
    "dis_m3_pmn": {"description": "Mean natural river discharge", "units": "cubic meters per second"},
    "lka_pc_use": {"description": "Lake area percent upstream of pour point", "units": "percent"},
    "gwt_cm_sav": {"description": "Groundwater table depth subbasin average", "units": "centimeters"},
    "ele_mt_uav": {"description": "Mean Elevation (m) upstream of pour point", "units": "meters"},
    "slp_dg_uav": {"description": "Terrain slope (degrees) upstream of pour point", "units": "degrees"},
    "sgr_dk_sav": {"description": "Stream gradient (decimeters/km) subbasin", "units": "decimeters per kilometer"},
    "clz_cl_smj": {"description": "Climate zone class (18 classes) subbasin", "units": "class"},
    "cmi_ix_uyr": {"description": "Annual climate moisture index (x100) upstream of pour point", "units": "index"},
    **GLC_PERCENT_ATTRS,
    **PNV_PERCENT_ATTRS,
    "wet_cl_smj": {"description": "Wetland class (12 classes) subbasin spatial majority", "units": "class"},
    "for_pc_use": {"description": "Forest cover percent upstream of pour point", "units": "percent"},
    "cly_pc_uav": {"description": "Clay percent upstream of pour point", "units": "percent"},
    "slt_pc_uav": {"description": "Silt percent upstream of pour point", "units": "percent"},
    "snd_pc_uav": {"description": "Sand percent upstream of pour point", "units": "percent"},
    "soc_th_uav": {"description": "Soil organic carbon content upstream of pour point", "units": "tons per hectare"},
    "swc_pc_uyr": {"description": "Annual soil water content upstream of pour point", "units": "percent"},
}


def add_land_cover_rank_attributes(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact top-3 class composition features for land cover and PNV."""
    out = results_df.copy()
    groups = {
        "glc": tuple(GLC_PERCENT_ATTRS),
        "pnv": tuple(PNV_PERCENT_ATTRS),
    }

    for prefix, percent_cols in groups.items():
        available_cols = [col for col in percent_cols if col in out.columns]
        if not available_cols:
            continue

        values = out[available_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, -np.inf)
        class_codes = np.array([int(col.rsplit("u", maxsplit=1)[-1]) for col in available_cols])
        order = np.argsort(-values, axis=1)

        for rank in range(3):
            ranked_idx = order[:, rank]
            ranked_pct = values[np.arange(values.shape[0]), ranked_idx]
            valid = np.isfinite(ranked_pct) & (ranked_pct > 0.0)
            out[f"{prefix}_cl_top{rank + 1}"] = np.where(valid, class_codes[ranked_idx], 0)
            out[f"{prefix}_pc_top{rank + 1}"] = np.where(valid, ranked_pct, 0.0)

        out.drop(columns=available_cols, inplace=True)

    return out


def add_area_rank_class_attributes(
    results_df: pd.DataFrame,
    source_df: pd.DataFrame,
    class_col: str,
    prefix: str,
    group_col: str = "PourPtID",
    area_col: str = "Shape_Area",
    n_ranks: int = 3,
) -> pd.DataFrame:
    """Add top-N class codes and area shares from intersect polygon classes."""
    if class_col not in source_df.columns:
        return results_df

    work = source_df[[group_col, class_col, area_col]].copy()
    work[class_col] = pd.to_numeric(work[class_col], errors="coerce")
    work[area_col] = pd.to_numeric(work[area_col], errors="coerce").fillna(0.0)
    valid = work[class_col].notna() & (work[class_col] != -9999) & (work[area_col] > 0.0)
    work = work.loc[valid, [group_col, class_col, area_col]].copy()
    work[class_col] = work[class_col].astype(int)

    total_work = source_df[[group_col, area_col]].copy()
    total_work[area_col] = pd.to_numeric(total_work[area_col], errors="coerce").fillna(0.0)
    totals = total_work.groupby(group_col, observed=True)[area_col].sum().rename("total_area")
    class_areas = (
        work.groupby([group_col, class_col], observed=True)[area_col]
        .sum()
        .rename("class_area")
        .reset_index()
    )
    if class_areas.empty:
        rank_df = results_df[[group_col]].copy()
        for rank in range(1, n_ranks + 1):
            rank_df[f"{prefix}_cl_top{rank}"] = 0
            rank_df[f"{prefix}_pc_top{rank}"] = 0.0
    else:
        class_areas = class_areas.join(totals, on=group_col)
        class_areas.sort_values(
            [group_col, "class_area", class_col],
            ascending=[True, False, True],
            inplace=True,
        )
        class_areas["rank"] = class_areas.groupby(group_col, observed=True).cumcount() + 1
        class_areas = class_areas[class_areas["rank"] <= n_ranks].copy()
        class_areas[f"{prefix}_pc"] = class_areas["class_area"] / class_areas["total_area"] * 100.0

        rank_df = results_df[[group_col]].copy()
        for rank in range(1, n_ranks + 1):
            ranked = class_areas[class_areas["rank"] == rank][[group_col, class_col, f"{prefix}_pc"]]
            ranked = ranked.rename(
                columns={
                    class_col: f"{prefix}_cl_top{rank}",
                    f"{prefix}_pc": f"{prefix}_pc_top{rank}",
                }
            )
            rank_df = rank_df.merge(ranked, on=group_col, how="left")

    out = results_df.merge(rank_df, on=group_col, how="left")
    for rank in range(n_ranks):
        out[f"{prefix}_cl_top{rank + 1}"] = out[f"{prefix}_cl_top{rank + 1}"].fillna(0).astype(int)
        out[f"{prefix}_pc_top{rank + 1}"] = out[f"{prefix}_pc_top{rank + 1}"].fillna(0.0).round(1)
    return out


def add_derived_routing_attributes(results_df: pd.DataFrame) -> pd.DataFrame:
    """Add routing-friendly aliases and simple river hydraulic proxies."""
    out = results_df.copy()
    out.rename(
        columns={
            "DIST_SINK": "dist_sink_km",
            "DIST_MAIN": "dist_main_km",
            "dis_m3_pmn": "discharge_mean_m3s",
        },
        inplace=True,
    )

    if {"ria_ha_usu", "riv_tc_usu"}.issubset(out.columns):
        river_area_m2 = pd.to_numeric(out["ria_ha_usu"], errors="coerce") * 10_000.0
        river_volume_m3 = pd.to_numeric(out["riv_tc_usu"], errors="coerce") * 1_000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = river_volume_m3 / river_area_m2
        out["river_mean_depth_m"] = depth.replace([np.inf, -np.inf], np.nan)

    if {"riv_tc_usu", "discharge_mean_m3s"}.issubset(out.columns):
        river_volume_m3 = pd.to_numeric(out["riv_tc_usu"], errors="coerce") * 1_000.0
        discharge_m3s = pd.to_numeric(out["discharge_mean_m3s"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            residence_days = river_volume_m3 / (discharge_m3s * 86_400.0)
        residence_days = residence_days.where(discharge_m3s > 0)
        out["river_residence_days"] = residence_days.replace([np.inf, -np.inf], np.nan)

    return out


def identify_basin_atlas_columns(df: pd.DataFrame) -> dict:
    """Identify BasinATLAS columns present in the dataframe."""
    basin_atlas_info = {}
    for col_name, attr_info in BASIN_ATLAS_STATIC_ATTR.items():
        if col_name in df.columns:
            is_class = attr_info.get('units', '').lower() == 'class'
            basin_atlas_info[col_name] = {
                'description': attr_info.get('description', ''),
                'units': attr_info.get('units', ''),
                'is_class': is_class,
            }
    return basin_atlas_info


def calculate_weighted_averages(
    df: pd.DataFrame,
    group_col: str = 'PourPtID',
    area_col: str = 'Shape_Area',
) -> pd.DataFrame:
    """Calculate area-weighted averages for each unique PourPtID."""
    basin_atlas_info = identify_basin_atlas_columns(df)
    print(f"Found {len(basin_atlas_info)} BasinATLAS columns to process")

    class_attrs = [col for col, info in basin_atlas_info.items() if info['is_class']]
    numeric_attrs = [col for col, info in basin_atlas_info.items() if not info['is_class']]
    print(f"  - Class attributes (using largest polygon): {len(class_attrs)} - {class_attrs}")
    print(f"  - Numeric attributes (using weighted average): {len(numeric_attrs)}")

    results = []
    grouped = df.groupby(group_col)
    print(f"\nProcessing {len(grouped)} unique PourPtIDs...")

    for pour_pt_id, group in grouped:
        total_area = group[area_col].sum()
        largest_area_idx = group[area_col].idxmax()
        area_weights = group[area_col] / total_area

        result = {
            group_col: pour_pt_id,
            f'total_{area_col}': total_area,
            'num_basins': len(group),
            'FID_BasinATLAS_count': len(group),
        }

        for col, info in basin_atlas_info.items():
            if group[col].isna().all():
                result[col] = np.nan
                continue

            if info['is_class']:
                class_value = group.loc[largest_area_idx, col]
                result[col] = 0 if class_value == -9999 else class_value
            else:
                if pd.api.types.is_numeric_dtype(group[col]):
                    valid_mask = ~group[col].isna()
                    if valid_mask.any():
                        valid_weights = area_weights[valid_mask]
                        valid_weights = valid_weights / valid_weights.sum()
                        weighted_avg = (group[col][valid_mask] * valid_weights).sum()
                        result[col] = round(weighted_avg, 1)
                    else:
                        result[col] = np.nan
                else:
                    result[col] = group[col].mode()[0] if len(group[col].mode()) > 0 else np.nan

        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(group_col)
    return results_df


def main(target: str = "watersheds", scope: str = "training") -> None:
    tp = get_eval_target_paths(target) if scope == "eval" else get_target_paths(target)
    basin_atlas_input = tp["basin_atlas_input"]
    basin_atlas_output = tp["basin_atlas_output"]

    print("=" * 70)
    print(f"BasinATLAS Weighted Average Calculator  [target={target}]")
    print("=" * 70)

    if not basin_atlas_input.exists():
        print(f"ERROR: Input file not found: {basin_atlas_input}")
        return

    print(f"\nReading input file: {basin_atlas_input.name}")
    df = pd.read_csv(basin_atlas_input)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    required_cols = ['PourPtID', 'Shape_Area', 'FID_BasinATLAS_v10_lev12']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        return

    print(f"\nUnique PourPtIDs: {df['PourPtID'].nunique()}")
    print(f"Unique BasinATLAS IDs: {df['FID_BasinATLAS_v10_lev12'].nunique()}")

    print("\nCalculating weighted averages...")
    results_df = calculate_weighted_averages(df)

    print("\nAdding lithology composition attributes...")
    results_df = add_area_rank_class_attributes(results_df, df, "lit_cl_smj", "lit")

    print("\nAdding land-cover composition attributes...")
    results_df = add_land_cover_rank_attributes(results_df)

    if target == "watersheds":
        # Replace total_Shape_Area with km2 areas from watershed geometry
        print(f"\nLoading watershed areas from: {WATERSHED_GEOMETRY.name}")
        ws_df = pd.read_csv(WATERSHED_GEOMETRY)
        ws_areas = ws_df.set_index('Pour Point ID')['Area Square Kilometers']
        matched = results_df['PourPtID'].isin(ws_areas.index)
        results_df.loc[matched, 'total_Shape_Area'] = results_df.loc[matched, 'PourPtID'].map(ws_areas)
        results_df.rename(columns={'total_Shape_Area': 'total_Shape_Area_km2'}, inplace=True)
        n_matched = matched.sum()
        n_missing = (~matched).sum()
        print(f"  Replaced {n_matched} areas with km2 values from watersheds.csv")
        if n_missing > 0:
            missing_ids = results_df.loc[~matched, 'PourPtID'].tolist()
            print(f"  WARNING: {n_missing} PourPtIDs not found in watersheds.csv: {missing_ids}")
    else:
        # Convert intersect m² → km² for non-watershed targets
        results_df['total_Shape_Area'] = results_df['total_Shape_Area'] / 1e6
        results_df.rename(columns={'total_Shape_Area': 'total_Shape_Area_km2'}, inplace=True)
        print(f"\n  Converted Shape_Area m² → km² for {len(results_df)} polygons")

    print("\nAdding routing/hydraulic derived attributes...")
    results_df = add_derived_routing_attributes(results_df)

    print("\nApplying precision rounding...")
    for col in results_df.columns:
        if col not in ['PourPtID', 'num_basins', 'FID_BasinATLAS_count'] and 'Shape_Area' not in col:
            if pd.api.types.is_numeric_dtype(results_df[col]):
                results_df[col] = results_df[col].round(1)

    basin_atlas_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {basin_atlas_output.name}")
    results_df.to_csv(basin_atlas_output, index=False)

    print(f"\nResults summary:")
    print(f"  - Output rows (unique PourPtIDs): {len(results_df)}")
    print(f"  - Output columns: {len(results_df.columns)}")
    print(f"  - BasinATLAS attribute columns: {len(BASIN_ATLAS_STATIC_ATTR)}")

    print(f"\nFirst few rows of results:")
    print(results_df.head())
    print(f"\nArea statistics:")
    print(results_df[['PourPtID', 'total_Shape_Area_km2', 'num_basins']].describe())

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
