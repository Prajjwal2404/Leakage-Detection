import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(scada_path, leakages_path, rolling_window=3, magnitude=1.0, seq=False):
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

    # 4. Add Time Meta-Features
    print("Adding Time Meta-features...")
    df_scada['Hour'] = df_scada.index.hour
    df_scada['Is_Nighttime'] = df_scada['Hour'].apply(lambda x: 1 if 2 <= x <= 6 else 0)

    if not seq:
        # 5. Add Rolling Stats (Moving Average and Std Dev) and Centered Features
        base_sensors = [c for c in df_scada.columns if c not in ['Hour', 'Is_Nighttime']]
        
        for col in base_sensors:
            df_scada[f'{col}_RollMean'] = df_scada[col].rolling(window=rolling_window).mean()
            df_scada[f'{col}_RollStd'] = df_scada[col].rolling(window=rolling_window).std()
            df_scada[f'{col}_Diff'] = df_scada[col] - df_scada[f'{col}_RollMean']

        # Drop NaNs created by rolling calculations
        df_scada.dropna(inplace=True)

    # 6. Process Leakages as the Target (Y)
    print("Loading and preparing Target (Y)...")
    df_leakages = pd.read_csv(leakages_path, sep=';', low_memory=False)
    df_leakages['Timestamp'] = pd.to_datetime(df_leakages['Timestamp'])
    df_leakages.set_index('Timestamp', inplace=True)

    # Force all pipe columns to numeric (handling European comma decimals or string artifacts if any exist)
    df_leakages = df_leakages.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce').fillna(0)

    if magnitude < 0:
        df_leakages['Leak'] = df_leakages.sum(axis=1)
    else:
        df_leakages['Leak'] = (df_leakages.sum(axis=1) > magnitude).astype(int)

    # 7. Final Alignment
    X = df_scada
    Y = df_leakages.loc[X.index, 'Leak']

    if seq:
        # 8. Scale and Sequence Data for RNN
        print("Scaling and sequencing data for RNN...")
        Is_Nighttime = X['Is_Nighttime'].to_numpy()
        X[X.columns] = StandardScaler().fit_transform(X[X.columns])
        return X.to_numpy(), Y.to_numpy(), Is_Nighttime

    return X, Y


if __name__ == '__main__':
    scada_file = os.path.join('Dataset', '2018_SCADA.xlsx')
    leakages_file = os.path.join('Dataset', '2018_Leakages.csv')
    
    X, Y = load_and_preprocess_data(scada_file, leakages_file, rolling_window=36)
    
    print("Preprocessing Complete!")
    print(f"X shape: {X.shape}")
    print(X.head())
    print(f"\nY shape: {Y.shape}")
    print(Y.head())
    print(f"Total hours with anomalies (leaks): {int(Y.sum())}")
    print(f"Total hours without anomalies (no leaks): {len(Y) - int(Y.sum())}")