import pandas as pd
import numpy as np
import re
import os



CENSUS_POP_FILES = [
    r"./counties_pop_2000_2010.csv",
    r"./counties_pop_2020_2010.csv",
    r"./counties_pop_2024_2020.csv",
]
GAZETTEER_FILE = r"./florida_gazetteer_2024.txt" # tf is a gazetteer
OUTPUT_POP_DENSITY_PATH = r"./county_population_density.csv"

FLORIDA_STATE_FIPS = 12 # Numeric FIPS code for Florida
SQ_METERS_TO_SQ_MILES = 3.86102e-7


# 1. Process Gazetteer File for Land Area
print(f"Processing Gazetteer file: {GAZETTEER_FILE}")
if not os.path.exists(GAZETTEER_FILE):
    print(f"Error: Gazetteer file not found at {GAZETTEER_FILE}")
    exit()

try:
    # read the tab-delimited text file
    df_gaz = pd.read_csv(GAZETTEER_FILE, delimiter='\t', dtype={'GEOID': str})
    # filter for florida
    df_gaz_fl = df_gaz[df_gaz['USPS'] == 'FL'].copy()
    # select and rename columns
    df_gaz_fl = df_gaz_fl[['GEOID', 'ALAND']].copy()
    df_gaz_fl.rename(columns={'GEOID': 'COUNTY_FIPS', 'ALAND': 'Land_Area_Meters'}, inplace=True)
    # convert area to square miles
    df_gaz_fl['Land_Area_SQMI'] = df_gaz_fl['Land_Area_Meters'] * SQ_METERS_TO_SQ_MILES
    # keep only necessary columns
    df_land_area = df_gaz_fl[['COUNTY_FIPS', 'Land_Area_SQMI']].copy()
    # ensure fips is 5 digits string
    df_land_area['COUNTY_FIPS'] = df_land_area['COUNTY_FIPS'].astype(str).str.zfill(5)
    print(f"Processed land area for {len(df_land_area)} Florida counties.")

except Exception as e:
    print(f"Error processing gazetteer file: {e}")
    exit()


# 2. Process Population Files
print("Processing Census population files...")
all_county_pop = []

for file_path in CENSUS_POP_FILES:
    if not os.path.exists(file_path):
        print(f"Warning: Population file not found, skipping: {file_path}")
        continue

    try:
        print(f"Loading: {file_path}")
        df_pop_raw = pd.read_csv(file_path, encoding="ISO-8859-1") # trying bc common

        # filter for florida counties (state == 12) and county level (sumlev == 50)
        df_pop_fl = df_pop_raw[(df_pop_raw['STATE'] == FLORIDA_STATE_FIPS) & (df_pop_raw['SUMLEV'] == 50)].copy()

        if df_pop_fl.empty:
            print(f"Warning: No Florida county data found in {file_path}")
            continue

        # create 5-digit county fips (state fips + county fips, zero-padded)
        df_pop_fl['COUNTY_FIPS'] = df_pop_fl['STATE'].astype(str).str.zfill(2) + \
                                   df_pop_fl['COUNTY'].astype(str).str.zfill(3)

        # identify population columns (POPESTIMATE followed by year)
        pop_cols = [col for col in df_pop_fl.columns if col.upper().startswith('POPESTIMATE') and col[-4:].isdigit()]

        if not pop_cols:
             print(f"Warning: No 'POPESTIMATE*' columns found in {file_path}. Skipping.")
             continue

        id_vars = ['COUNTY_FIPS']
        df_melted = pd.melt(df_pop_fl, id_vars=id_vars, value_vars=pop_cols,
                            var_name='Year_Str', value_name='Population')

        # extract numeric year from the column name
        def extract_year(year_str):
            match = re.search(r'(\d{4})', year_str)
            return int(match.group(1)) if match else None

        df_melted['YEAR'] = df_melted['Year_Str'].apply(extract_year)
        df_melted.dropna(subset=['YEAR'], inplace=True)
        df_melted['YEAR'] = df_melted['YEAR'].astype(int)

        # keep only relevant columns
        df_processed = df_melted[['COUNTY_FIPS', 'YEAR', 'Population']].copy()
        all_county_pop.append(df_processed)
        print(f"Processed population data from: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# 3. Combine Population Data
if not all_county_pop:
    print("Error: No population data processed. Exiting.")
    exit()

df_final_pop = pd.concat(all_county_pop, ignore_index=True)
# remove duplicates, keeping the latest estimate if years overlap between files
df_final_pop.drop_duplicates(subset=['COUNTY_FIPS', 'YEAR'], keep='last', inplace=True)
df_final_pop.sort_values(by=['COUNTY_FIPS', 'YEAR'], inplace=True)

# ensure population is numeric
df_final_pop['Population'] = pd.to_numeric(df_final_pop['Population'], errors='coerce')
df_final_pop.dropna(subset=['Population'], inplace=True)
df_final_pop['Population'] = df_final_pop['Population'].astype(int)

print(f"Combined population data shape: {df_final_pop.shape}")

# 4. Merge Population and Land Area
print("Merging population and land area data...")
df_merged = pd.merge(df_final_pop, df_land_area, on='COUNTY_FIPS', how='left')

missing_area = df_merged['Land_Area_SQMI'].isnull().sum()
if missing_area > 0:
    print(f"Warning: Could not find land area for {missing_area} county-year entries.")
    # drop rows where area is missing, as density cannot be calculated
    df_merged.dropna(subset=['Land_Area_SQMI'], inplace=True)

# 5. Calculate Population Density
print("Calculating population density...")
# avoid division by zero if land area is somehow zero
df_merged['Population_Density'] = df_merged.apply(
    lambda row: row['Population'] / row['Land_Area_SQMI'] if row['Land_Area_SQMI'] > 0 else 0,
    axis=1
)

df_output = df_merged[['COUNTY_FIPS', 'YEAR', 'Population_Density']].copy()
df_output.sort_values(by=['COUNTY_FIPS', 'YEAR'], inplace=True)

df_output.to_csv(OUTPUT_POP_DENSITY_PATH, index=False)
print(f"\n✅ County population density data saved to: {OUTPUT_POP_DENSITY_PATH}")
print(f"   Final Shape: {df_output.shape}")
print(df_output.head())

