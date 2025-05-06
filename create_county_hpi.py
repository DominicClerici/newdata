import pandas as pd
import numpy as np
import os


# --- !!! UPDATE THIS PATH !!! ---
FHFA_HPI_FILE_PATH = r"./HPI_AT_BDL_county.csv"

OUTPUT_HPI_PATH = r"./florida_hpi_quarterly.csv"

print(f"Processing FHFA HPI file: {FHFA_HPI_FILE_PATH}")
if not os.path.exists(FHFA_HPI_FILE_PATH):
    print(f"Error: FHFA HPI file not found at {FHFA_HPI_FILE_PATH}")
    exit()

try:
    df_hpi_raw = pd.read_csv(FHFA_HPI_FILE_PATH)
    print(f"Loaded raw HPI data. Shape: {df_hpi_raw.shape}")

    # filter for florida state data: state level, quarterly, all-transactions
    df_filtered = df_hpi_raw[
        (df_hpi_raw['level'] == 'State') &
        (df_hpi_raw['hpi_flavor'] == 'all-transactions') &
        (df_hpi_raw['frequency'] == 'quarterly') &
        (df_hpi_raw['place_id'] == 'FL')
    ].copy()
    print(f"\nFiltered for Florida State/Quarterly/All-Transactions. Shape: {df_filtered.shape}")

    if df_filtered.empty:
        print("Error: No Florida state HPI data found after filtering.")
        exit()

    #  rename columns
    df_output = df_filtered[['place_id', 'yr', 'period', 'index_nsa']].copy()
    df_output.rename(columns={
        'place_id': 'STATE_FIPS',
        'yr': 'YEAR',
        'period': 'QUARTER',
        'index_nsa': 'State_HPI'
    }, inplace=True)

    # ensure correct data types
    df_output['STATE_FIPS'] = df_output['STATE_FIPS'].astype(str)
    df_output['YEAR'] = df_output['YEAR'].astype(int)
    df_output['QUARTER'] = df_output['QUARTER'].astype(int)
    df_output['State_HPI'] = pd.to_numeric(df_output['State_HPI'], errors='coerce')

    # drop rows where cant be converted to numeric
    df_output.dropna(subset=['State_HPI'], inplace=True)

    df_output.sort_values(by=['YEAR', 'QUARTER'], inplace=True)

    df_output.to_csv(OUTPUT_HPI_PATH, index=False)
    print(f"\n✅ Florida state quarterly HPI data saved to: {OUTPUT_HPI_PATH}")
    print(f"   Final Shape: {df_output.shape}")
    print("\nFirst few rows of output:")
    print(df_output.head())

except Exception as e:
    print(f"Error processing FHFA HPI file: {e}")

