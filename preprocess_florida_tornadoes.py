import pandas as pd
import numpy as np
import re
import os
import geopandas as gpd 
from shapely.geometry import Point 

# === Configuration ===
TORNADO_DATA_PATH = r"./tornado_only_post2000.csv"
COUNTY_POP_DENSITY_PATH = r"./county_population_density.csv"
STATE_QUARTERLY_HPI_PATH = r"./florida_hpi_quarterly.csv"
FL_COUNTY_SHAPEFILE_PATH = r"./shape/tl_2022_12_cousub.shp" 

OUTPUT_PATH = r"./florida_tornado_features_countyPop_stateHPI.csv"

print("Loading tornado data...")
df = pd.read_csv(TORNADO_DATA_PATH, low_memory=False)

# === Filter for Florida & Ensure Lat/Lon ===
df_fl = df[df["STATE"] == "FLORIDA"].copy()
df_fl.dropna(subset=['BEGIN_LAT', 'BEGIN_LON'], inplace=True)
# Reset index after filtering/dropping to ensure it's clean
df_fl.reset_index(drop=True, inplace=True)

if df_fl.empty:
    print("Error: No Florida tornado data found or missing lat/lon.")
    exit()

# === 1. Determine County FIPS using Geospatial Lookup ===
print("Performing geospatial lookup for counties...")
try:
    counties_gdf = gpd.read_file(FL_COUNTY_SHAPEFILE_PATH)
    counties_gdf = counties_gdf.to_crs("EPSG:4326")

    # --- fix geometry column in df_fl  ---
    df_fl['geometry'] = [Point(xy) for xy in zip(df_fl['BEGIN_LON'], df_fl['BEGIN_LAT'])]

    # --- fix GeoDataFrame  ---
    tornado_gdf = gpd.GeoDataFrame(df_fl, geometry='geometry', crs="EPSG:4326")

    # --- get county fips column  ---
    COUNTY_FIPS_COL_IN_SHAPEFILE = 'GEOID' 
    if COUNTY_FIPS_COL_IN_SHAPEFILE not in counties_gdf.columns:
         raise ValueError(f"Column '{COUNTY_FIPS_COL_IN_SHAPEFILE}' not found in shapefile. Available: {counties_gdf.columns}")

    # Only select necessary columns from counties_gdf for no dups
    joined_gdf = gpd.sjoin(tornado_gdf, counties_gdf[[COUNTY_FIPS_COL_IN_SHAPEFILE, 'geometry']],
                           how="left", predicate="within")

    # --- assign fips back using the index ---
    fips_map = counties_gdf[COUNTY_FIPS_COL_IN_SHAPEFILE] # Series mapping county index to FIPS
    df_fl['COUNTY_FIPS'] = joined_gdf['index_right'].map(fips_map)

    # --- Rest of the validation and processing ---
    print("Validating mapped COUNTY_FIPS codes...")
    print(f"Unique FIPS count: {df_fl['COUNTY_FIPS'].nunique()}")
    print(f"Sample FIPS: {df_fl['COUNTY_FIPS'].unique()[:10]}")
    fips_lengths = df_fl['COUNTY_FIPS'].astype(str).str.len().value_counts()
    print(f"FIPS Code Lengths:\n{fips_lengths}")
    # Check if any length is not 5 (can happen if mapping failed -> NaN -> 'nan' -> 3 chars)
    if not all(l == 5 for l in fips_lengths.index if not pd.isna(l)):
         print("Warning: Some FIPS codes may not be 5 digits. Check shapefile 'GEOID' column or mapping.")

    # Ensure FIPS is 5-digit string, handling potential NaNs from failed maps
    df_fl['COUNTY_FIPS'] = df_fl['COUNTY_FIPS'].astype(str).str.replace('.0','', regex=False).str.zfill(5) 
    df_fl.loc[df_fl['COUNTY_FIPS'] == 'nan', 'COUNTY_FIPS'] = np.nan

    missing_counties = df_fl['COUNTY_FIPS'].isnull().sum()
    if missing_counties > 0:
        print(f"Warning: Could not map county for {missing_counties} tornadoes. Dropping these rows.")
        df_fl.dropna(subset=['COUNTY_FIPS'], inplace=True)

    print(f"Successfully mapped counties for {len(df_fl)} tornadoes.")

except Exception as e:
    print(f"Error during geospatial processing: {e}")
    exit()

# === deduplicate ===
initial_rows = len(df_fl)
# Check for duplicates based on original identifying columns if possible, or just all columns
id_cols = ['EVENT_ID'] if 'EVENT_ID' in df_fl.columns else ['BEGIN_DATE_TIME', 'BEGIN_LAT', 'BEGIN_LON']
if all(col in df_fl.columns for col in id_cols):
    df_fl.drop_duplicates(subset=id_cols, keep='first', inplace=True)
else:
    print("Warning: Cannot find unique id columns, dropping duplicates based on all columns")
    df_fl.drop_duplicates(keep='first', inplace=True)

final_rows = len(df_fl)
if initial_rows != final_rows:
    print(f"Dropped {initial_rows - final_rows} duplicate rows after merges.")
    print(f"Shape after deduplication: {df_fl.shape}")

# Target Variable
df_fl['DAMAGE_PROPERTY_LOG'] = np.log1p(df_fl["DAMAGE_PROPERTY"].fillna(0))

# EF Scale
def parse_f_scale(scale):
    if pd.isna(scale): return np.nan
    match = re.search(r"\d+", str(scale))
    return int(match.group(0)) if match else np.nan
df_fl["EF_SCALE_NUM"] = df_fl["TOR_F_SCALE"].apply(parse_f_scale)
median_ef_scale = df_fl["EF_SCALE_NUM"].median()
df_fl["EF_SCALE_NUM"] = df_fl["EF_SCALE_NUM"].fillna(median_ef_scale)
print(f"Processed EF_SCALE_NUM. Median used for imputation: {median_ef_scale}")

# TOR Length/Width
median_length = df_fl["TOR_LENGTH"].median()
median_width = df_fl["TOR_WIDTH"].median()
# --- fix no inplace ---
df_fl["TOR_LENGTH"] = df_fl["TOR_LENGTH"].fillna(median_length)
df_fl["TOR_WIDTH"] = df_fl["TOR_WIDTH"].fillna(median_width)
print(f"Imputed TOR_LENGTH (median: {median_length}) and TOR_WIDTH (median: {median_width})")

# Time Features
df_fl["BEGIN_DATE_TIME"] = pd.to_datetime(df_fl["BEGIN_DATE_TIME"], errors="coerce")
df_fl.dropna(subset=["BEGIN_DATE_TIME"], inplace=True) 
df_fl["YEAR"] = df_fl["BEGIN_DATE_TIME"].dt.year.astype(int)
df_fl["MONTH"] = df_fl["BEGIN_DATE_TIME"].dt.month.astype(int)
df_fl["QUARTER"] = df_fl["BEGIN_DATE_TIME"].dt.quarter.astype(int)

# === 3. Load and Merge County Pop Density and State HPI Data ===
print("Loading and merging county pop density and state HPI data...")
try:
    df_county_pop_density = pd.read_csv(COUNTY_POP_DENSITY_PATH)
    df_state_hpi = pd.read_csv(STATE_QUARTERLY_HPI_PATH)

    # Ensure merge keys are correct types
    df_county_pop_density['COUNTY_FIPS'] = df_county_pop_density['COUNTY_FIPS'].astype(str).str.zfill(5)
    df_county_pop_density['YEAR'] = df_county_pop_density['YEAR'].astype(int)
    df_state_hpi['YEAR'] = df_state_hpi['YEAR'].astype(int)
    df_state_hpi['QUARTER'] = df_state_hpi['QUARTER'].astype(int)

    # --- Merge Population Density (County Level) ---
    print(f"Shape before pop density merge: {df_fl.shape}")
    df_fl = pd.merge(df_fl, df_county_pop_density[['COUNTY_FIPS', 'YEAR', 'Population_Density']],
                     on=['COUNTY_FIPS', 'YEAR'], how='left')
    print(f"Shape after pop density merge: {df_fl.shape}") # Check shape

    # --- Merge State HPI (State Level, Quarterly) ---
    print(f"Shape before state HPI merge: {df_fl.shape}")
    df_fl = pd.merge(df_fl, df_state_hpi[['YEAR', 'QUARTER', 'State_HPI']],
                     on=['YEAR', 'QUARTER'], how='left')
    print(f"Shape after state HPI merge: {df_fl.shape}") # Check shape - SHOULD NOT CHANGE

    print("Merge complete. Checking for missing values...")

    # --- Impute missing county pop density data ---
    missing_pop_dens = df_fl['Population_Density'].isnull().sum()
    if missing_pop_dens > 0:
        print(f"Found {missing_pop_dens} missing Population_Density values. Imputing...")
        # Impute missing pop density with county median across years FIRST
        df_fl['Population_Density'] = df_fl.groupby('COUNTY_FIPS')['Population_Density'].transform(lambda x: x.fillna(x.median()))
        # THEN impute remaining NaNs (if a county had NO data) with overall median
        overall_median_pop_dens = df_fl['Population_Density'].median() # Calculate median *after* groupby fill
        if pd.isna(overall_median_pop_dens):
             print("Warning: Overall median pop density is NaN after group imputation. Check input data.")
             overall_median_pop_dens = 0 # Or some other reasonable default
        # --- FIX: Reassign instead of inplace ---
        df_fl['Population_Density'] = df_fl['Population_Density'].fillna(overall_median_pop_dens)
        print(f"Imputed Population_Density. Overall median used: {overall_median_pop_dens:.2f}")

    # --- Impute missing state HPI data ---
    missing_state_hpi = df_fl['State_HPI'].isnull().sum()
    if missing_state_hpi > 0:
        print(f"Found {missing_state_hpi} missing State_HPI values. Imputing...")
        overall_median_state_hpi = df_fl['State_HPI'].median()
        if pd.isna(overall_median_state_hpi):
             print("Warning: Overall median state HPI is NaN. Check input data.")
             overall_median_state_hpi = df_state_hpi['State_HPI'].median() # Fallback to median from source file
        # --- FIX: Reassign instead of inplace ---
        df_fl['State_HPI'] = df_fl['State_HPI'].fillna(overall_median_state_hpi)
        print(f"Imputed State_HPI. Overall median used: {overall_median_state_hpi:.2f}")

except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    exit()
except Exception as e:
    print(f"Error during data merge/imputation: {e}")
    exit()


# === 4. Select Final Features ===
target = "DAMAGE_PROPERTY_LOG"
features = [
    "YEAR",
    "MONTH",
    "EF_SCALE_NUM",
    "TOR_LENGTH",
    "TOR_WIDTH",
    "Population_Density", 
    "State_HPI",          
    "BEGIN_LAT",          
    "BEGIN_LON"           
]

cols_to_keep = features + [target] + ["DAMAGE_PROPERTY", "COUNTY_FIPS"]
cols_to_keep = [col for col in cols_to_keep if col in df_fl.columns]
df_final = df_fl[cols_to_keep].copy()

# === 5. Final Check for Missing Values in Features ===
print("\nChecking for missing values in final feature set:")
missing_final = df_final[features].isnull().sum()
print(missing_final)
if missing_final.sum() > 0:
    print("Warning: Missing values remain in features after final check. Imputing with overall median.")
    for col in features:
        if df_final[col].isnull().any():
            median_val = df_final[col].median()
            if pd.isna(median_val): 
                 print(f"Warning: Median for {col} is NaN during final imputation. Filling with 0.")
                 median_val = 0
            df_final[col] = df_final[col].fillna(median_val)
            print(f"Imputed remaining NaNs in '{col}' with overall median ({median_val})")


# === Save Processed Data ===
df_final.to_csv(OUTPUT_PATH, index=False)
print(f"\nPreprocessed Florida tornado data saved to: {OUTPUT_PATH}")
print(f"   Shape: {df_final.shape}") 
print(f"   Columns: {df_final.columns.tolist()}")
print("\nFirst few rows of final data:")
print(df_final.head())