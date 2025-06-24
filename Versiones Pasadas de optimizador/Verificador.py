import pandas as pd
import numpy as np
from pandas.tseries.offsets import BDay
from datetime import datetime

# 1) Cargar precios diarios del IBC y calcular retorno diario
ibc_df = pd.read_csv('data_acciones/IBC_Daily_2020_2024.csv', parse_dates=['time'])
ibc_df.set_index('time', inplace=True)
ibc_df.sort_index(inplace=True)
ibc_df['ret_m'] = ibc_df['close'].pct_change()

# 2) Cargar precios diarios de BPV y calcular retorno diario
bpv_df = pd.read_csv('data_acciones/BPV.csv', parse_dates=['time'])
bpv_df.set_index('time', inplace=True)
bpv_df.sort_index(inplace=True)
bpv_df['ret_bpv'] = bpv_df['close'].pct_change()

# 3) Definir el período Q1 2022: desde 252 días hábiles antes del 31-mzo-2022
end_dt   = datetime(2022, 3, 31)
start_dt = end_dt - BDay(252)

# 4) Extraer retornos en esa ventana (y eliminar NaN)
ibc_ret_window = ibc_df['ret_m'].loc[start_dt:end_dt].dropna()
bpv_ret_window = bpv_df['ret_bpv'].loc[start_dt:end_dt].dropna()

# 5) Alinear ambas series por fecha
combined = pd.concat([ibc_ret_window, bpv_ret_window], axis=1, join='inner').dropna()
combined.columns = ['ret_m', 'ret_bpv']

# 6) Calcular β sobre esos 252 días:
cov = combined['ret_bpv'].cov(combined['ret_m'])
var = combined['ret_m'].var()
beta = cov / var if var != 0 else np.nan

print(f"β calculado (252 días) para BPV en Q1-2022: {beta:.4f}")
print(f"Varianza IBC (ventana): {var:.6e}")
print(f"Covarianza (BPV vs IBC): {cov:.6e}")

# 7) Identificar días con retorno diario extremo para BPV (> |50%|)
extreme_bpv = combined[np.abs(combined['ret_bpv']) > 0.5]
print("\nDías con retorno diario |BPV| > 50%:")
print(extreme_bpv)
