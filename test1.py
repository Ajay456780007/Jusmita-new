import pandas as pd
import numpy as np


def parse_time_to_seconds(time_str):
    """Converts strings like '00:01.0' or '01:23:45.0' into total seconds."""
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str).strip()
    parts = time_str.split(':')
    try:
        if len(parts) == 3:  # hh:mm:ss.s
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:  # mm:ss.s
            return float(parts[0]) * 60 + float(parts[1])
        else:
            return float(parts[0])
    except ValueError:
        return np.nan


def process_bitrode_file(filepath, flow_rate_ml_min=90.0):
    # 1. Read CSV starting at Row 14 (0-indexed header=13)
    df = pd.read_csv(filepath, skiprows=13)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # 2. Select and rename the required columns based on your sheet
    col_map = {
        'Total Time, (h:m:s)': 'Time_Raw',
        'Voltage, V': 'Voltage_V',
        'Current, A': 'Current_A',
        'Amp-Hours, AH': 'Amp_Hours_AH',
        'Cycle': 'Cycle',
        'Step': 'Step'
    }
    df = df.rename(columns=col_map)

    # 3. Drop empty/blank spacer rows (like Row 15)
    df = df.dropna(subset=['Time_Raw', 'Voltage_V', 'Current_A']).reset_index(drop=True)

    # 4. Convert Time string to numerical seconds
    df['Time_s'] = df['Time_Raw'].apply(parse_time_to_seconds)
    df['Voltage_V'] = pd.to_numeric(df['Voltage_V'], errors='coerce')
    df['Current_A'] = pd.to_numeric(df['Current_A'], errors='coerce')
    df['Amp_Hours_AH'] = pd.to_numeric(df['Amp_Hours_AH'], errors='coerce')

    # Drop any remaining non-numeric conversion artifacts
    df = df.dropna(subset=['Time_s', 'Voltage_V', 'Current_A']).reset_index(drop=True)

    # 5. Compute Continuous Ground-Truth SOC (Coulomb Counting)
    # Using the cycler's integrated Amp-Hours (or integrating Current_A * dt):
    dt_hours = np.diff(df['Time_s'].values, prepend=df['Time_s'].values[0]) / 3600.0
    cumulative_ah = np.cumsum(df['Current_A'].values * dt_hours)

    ah_min = np.min(cumulative_ah)
    ah_max = np.max(cumulative_ah)

    if ah_max > ah_min:
        # Ground-truth SOC scaled between 0 and 1
        df['SOC'] = (cumulative_ah - ah_min) / (ah_max - ah_min)
    else:
        df['SOC'] = 0.5

    # 6. Add Constant Electrolyte Flow Rate
    df['Flow_Rate_mL_min'] = flow_rate_ml_min

    # 7. Final merged dataframe for your CNN-GRU model
    merged_data = df[['Time_s', 'Voltage_V', 'Current_A', 'Flow_Rate_mL_min', 'SOC']].copy()

    return merged_data


# Run on your file:
file_path = "Dataset/VRFB/data/2. Raw data_ Research article 1_Interdigitated flow fields_Life cycle at 90J SF7_THTCp_1500 cm2 area.csv"
dataset = process_bitrode_file(file_path, flow_rate_ml_min=90.0)

print(dataset.head(10))