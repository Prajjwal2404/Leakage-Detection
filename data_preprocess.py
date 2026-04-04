import pandas as pd
import os


def load_and_preprocess_data(scada_path, leakages_path, resample_freq='1h', rolling_window=3):
    # Load all sheets
    print("Loading SCADA data... This may take a moment.")
    df_pressures = pd.read_excel(scada_path, sheet_name='Pressures (m)')
    df_flows = pd.read_excel(scada_path, sheet_name='Flows (m3_h)')
    df_levels = pd.read_excel(scada_path, sheet_name='Levels (m)')
    df_demands = pd.read_excel(scada_path, sheet_name='Demands (L_h)')

    # 1. Prepare Demands - Sum all individual demand columns
    demand_cols = [col for col in df_demands.columns if col != 'Timestamp']
    df_demands['Total_System_Demand'] = df_demands[demand_cols].sum(axis=1)
    df_demands = df_demands[['Timestamp', 'Total_System_Demand']]

    # 2. Merge all SCADA features
    df_scada = df_pressures.merge(df_flows, on='Timestamp', how='inner')
    df_scada = df_scada.merge(df_levels, on='Timestamp', how='inner')
    df_scada = df_scada.merge(df_demands, on='Timestamp', how='inner')

    # 3. Format DateTime and set as Index
    df_scada['Timestamp'] = pd.to_datetime(df_scada['Timestamp'])
    df_scada.set_index('Timestamp', inplace=True)

    # 4. Resample time intervals by averaging (mean)
    df_resampled = df_scada.resample(resample_freq).mean()

    # 5. Add Time Meta-Features
    print("Adding Time and Rolling features...")
    df_resampled['Hour'] = df_resampled.index.hour
    df_resampled['Is_Daytime'] = df_resampled['Hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)

    # 6. Add Rolling Stats (Moving Average and Std Dev)
    base_sensors = [c for c in df_resampled.columns if c not in ['Hour', 'Is_Daytime']]
    
    for col in base_sensors:
        df_resampled[f'{col}_RollMean'] = df_resampled[col].rolling(window=rolling_window).mean()
        df_resampled[f'{col}_RollStd'] = df_resampled[col].rolling(window=rolling_window).std()

    # Drop NaNs created by rolling calculations
    df_resampled.dropna(inplace=True)

    # 7. Process Leakages as the Target (Y)
    print("Loading and preparing Target (Y)...")
    df_leakages = pd.read_csv(leakages_path, sep=';', low_memory=False)
    df_leakages['Timestamp'] = pd.to_datetime(df_leakages['Timestamp'])
    df_leakages.set_index('Timestamp', inplace=True)

    # Force all pipe columns to numeric (handling European comma decimals or string artifacts if any exist)
    df_leakages = df_leakages.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    # Create Binary Column: Is there ANY leak?
    df_leakages['Any_Leak'] = (df_leakages.max(axis=1) > 0).astype(int)

    # Resample Leakages to the same frequency as SCADA data.
    df_leakages_resampled = df_leakages[['Any_Leak']].resample(resample_freq).max()

    # 8. Final Alignment
    X = df_resampled
    Y = df_leakages_resampled.loc[X.index, 'Any_Leak']

    return X, Y


if __name__ == '__main__':
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    X, Y = load_and_preprocess_data(scada_file, leakages_file, resample_freq='5min', rolling_window=36)
    
    print("Preprocessing Complete!")
    print(f"X shape: {X.shape}")
    print(X.head())
    print(f"\nY shape: {Y.shape}")
    print(Y.head())
    print(f"Total hours with anomalies (leaks): {int(Y.sum())}")
    print(f"Total hours without anomalies (no leaks): {len(Y) - int(Y.sum())}")