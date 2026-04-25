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

BASIN_ATLAS_STATIC_ATTR = {
    "ria_ha_usu": {"description": "River Area (ha) sum upstream of pour point", "units": "hectares"},
    "riv_tc_usu": {"description": "River Volume (thousand m3) sum upstream of pour point", "units": "thousand cubic meters"},
    "ele_mt_uav": {"description": "Mean Elevation (m) upstream of pour point", "units": "meters"},
    "slp_dg_uav": {"description": "Terrain slope (degrees) upstream of pour point", "units": "degrees"},
    "sgr_dk_sav": {"description": "Stream gradient (decimeters/km) subbasin", "units": "decimeters per kilometer"},
    "clz_cl_smj": {"description": "Climate zone class (18 classes) subbasin", "units": "class"},
    "cmi_ix_uav": {"description": "Average annual climate moisture index (x100) upstream of pour point", "units": "index"},
    "glc_cl_smj": {"description": "Land cover class (12 classes) subbasin spatial majority", "units": "class"},
    "pnv_cl_smj": {"description": "Potential natural vegetation class (15 classes) subbasin spatial majority", "units": "class"},
    "wet_cl_smj": {"description": "Wetland class (12 classes) subbasin spatial majority", "units": "class"},
    "for_pc_use": {"description": "Forest cover percent upstream of pour point", "units": "percent"},
    "cly_pc_uav": {"description": "Clay percent upstream of pour point", "units": "percent"},
    "slt_pc_uav": {"description": "Silt percent upstream of pour point", "units": "percent"},
    "snd_pc_uav": {"description": "Sand percent upstream of pour point", "units": "percent"},
    "lit_cl_smj": {"description": "Lithology class (16 classes) subbasin spatial majority", "units": "class"},
}


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
