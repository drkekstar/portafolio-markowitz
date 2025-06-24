import numpy as np
import pandas as pd
import os
import logging
import sys
from datetime import datetime
from scipy.optimize import minimize
from tabulate import tabulate
from decimal import Decimal, getcontext
from pandas.tseries.offsets import BDay
import traceback
import warnings

# ------------------------------------------------------------
# Definir carpetas de salida
# ------------------------------------------------------------
REPORT_DIR = "reporte trimestre"
EXCEL_DIR = "resultados excel"
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(EXCEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Configuro un logger que envía todo a consola y a archivos en "reporte trimestre"
# ------------------------------------------------------------
logger = logging.getLogger("PortfolioOptimizerLogger")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
# ------------------------------------------------------------

warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format')

class PortfolioOptimizer:
    def __init__(self, data_folder='data_acciones', crp_ve_anual=0.25, risk_premium_type='Country'):
        """
        crp_ve_anual: Prima de riesgo país anual (por ejemplo, 0.25 = 25% anual).
        risk_premium_type: 'Country' para Country Risk Premium,
                           'Equity'  para Equity Risk Premium.
        """
        self.data_folder = data_folder
        self.CRP_VE_anual = crp_ve_anual

        # A) Sólo transaction_cost (la prima del país se maneja aparte)
        self.market_params = {
            'transaction_cost': 0.0275  # 2.75%
        }
        # B) Tipo de prima a usar (para las primas leídas de CSV)
        self.risk_premium_type = risk_premium_type

        # C) Lista de tickers
        self.priority_tickers = [
            'BPV','MVZ.A','ABC.A','ENV','EFE','CGQ','CRM.A',
            'DOM','MPA','CCR','GZL','FNC','RST.B'
        ]

        # D) Cargar CSV de primas de riesgo (anual)
        self.risk_df = self.load_risk_premiums()

        # E) Cargar precios diarios del IBC
        self.daily_ibc = self.load_ibc_daily()

        # F) Inicializar diccionarios para fechas filtradas
        self.filtered_dates = {t: [] for t in self.priority_tickers}
        self.filtered_dates_precio = {t: [] for t in self.priority_tickers}

        # G) Calcular betas diarias (usa filtered_dates internamente)
        self.daily_betas = self.calculate_daily_betas(self.priority_tickers, window=252)

        # H) Diccionarios para betas y capitalizaciones al cierre de cada trimestre
        self.current_betas = {}
        self.current_mkt_caps = {}

    # -------------------------------------------------------
    # A) load_risk_premiums: lee RISK_PREMIUM_VENEZUELA.csv
    # -------------------------------------------------------
    def load_risk_premiums(self):
        """Carga 'RISK_PREMIUM_VENEZUELA.csv' con columnas: Year, Country Risk Premium, Equity Risk Premium, PAIS
        Además: convierte valores con '%' a flotantes en [0,1]."""
        
        path = os.path.join(self.data_folder, 'RISK_PREMIUM_VENEZUELA.csv')
        df = pd.read_csv(path)

        # Verificar columnas mínimas
        required = {'Year', 'Country Risk Premium', 'Equity Risk Premium'}
        if not required.issubset(df.columns):
            raise ValueError("RISK_PREMIUM_VENEZUELA.csv debe tener 'Year', "
                             "'Country Risk Premium' y 'Equity Risk Premium'.")

        # Convertir Year a int
        df['Year'] = df['Year'].astype(int)

        # Limpiar el símbolo '%' y convertir a decimal
        for col in ['Country Risk Premium', 'Equity Risk Premium']:
            df[col] = (
                df[col]
                .astype(str)
                .str.rstrip('%')      # "23.58%" → "23.58"
                .replace('', '0')     # en caso de celdas vacías, tratarlas como 0
                .astype(float) / 100  # "23.58" → 23.58 → 0.2358
            )

        # Finalmente, indexar por Year
        return df.set_index('Year').sort_index()

    # -------------------------------------------------------
    # B) get_annual_risk_premium y get_quarter_risk_premium
    # -------------------------------------------------------
    def get_annual_risk_premium(self, year):
        if year not in self.risk_df.index:
            raise ValueError(f"No se encontró prima de riesgo para el año {year}.")
        if self.risk_premium_type == 'Country':
            return float(self.risk_df.at[year, 'Country Risk Premium'])
        else:
            return float(self.risk_df.at[year, 'Equity Risk Premium'])

    def get_quarter_risk_premium(self, year, quarter):
        # Convertir prima anual a trimestral: (1+anual)^(1/4) - 1
        annual_premium = self.get_annual_risk_premium(year)
        return (1 + annual_premium) ** (1/4) - 1

    # -------------------------------------------------------
    # Método para convertir CRP anual a trimestral
    # -------------------------------------------------------
    def get_crp_trimestral(self):
        return (1 + self.CRP_VE_anual) ** (1/4) - 1

    # -------------------------------------------------------
    # C) load_ibc_daily: lee IBC_Daily_2020_2024.csv y retorna DataFrame con 'ret_m'
    # -------------------------------------------------------
    def load_ibc_daily(self):
        path = os.path.join(self.data_folder, 'IBC_Daily_2020_2024.csv')
        df = pd.read_csv(path, parse_dates=['time'])
        df = df.set_index('time').sort_index()
        df['ret_m'] = df['close'].pct_change()
        return df[['close', 'ret_m']]

    def clean_prices(self, df_precios, threshold=0.5, ticker=None):
        """
        Detecta saltos de precio > |threshold| (p. ej. 50 %) en df_precios['close'],
        los marca como NaN y rellena con ffill/bfill. Si ticker!=None,
        guarda en self.filtered_dates_precio[ticker] las fechas que se limpiaron.
        """
        pct = df_precios['close'].pct_change()
        mask = pct.abs() > threshold

        # Guardar fechas de outliers de precio
        if ticker is not None:
            fechas_err = df_precios.index[mask].tolist()
            for f in fechas_err:
                if f not in self.filtered_dates_precio[ticker]:
                    self.filtered_dates_precio[ticker].append(f)

        df_limpio = df_precios.copy()
        df_limpio.loc[mask, 'close'] = np.nan
        df_limpio['close'] = df_limpio['close'].ffill().bfill()
        return df_limpio

    # -------------------------------------------------------
    # D) calculate_daily_betas: calcula β diaria para cada ticker
    # -------------------------------------------------------
    def calculate_daily_betas(self, tickers, window=252):
        """
        Calcula la β diaria de cada ticker usando ventana de 'window' días hábiles.
        β_i(t) = Cov(r_i_window, r_m_window) / Var(r_m_window),
        donde filtramos retornos diarios >|50%| y registramos esas fechas en self.filtered_dates.
        """
        # 1) Retornos diarios del IBC
        df_ibc = self.daily_ibc[['ret_m']].dropna()

        # 2) Leer cada CSV de ticker, limpiar precios, calcular ret diario
        dict_retornos = {}
        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            df_raw = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
            df_t = self.clean_prices(df_raw, threshold=0.5, ticker=t)
            df_t[f'ret_{t}'] = df_t['close'].pct_change()
            dict_retornos[t] = df_t[[f'ret_{t}']]

        # 3) Concatenar retornos (alinear por fecha)
        all_ret = pd.concat([df_ibc] + list(dict_retornos.values()), axis=1).dropna()
        fechas = all_ret.index

        # 4) Preparar DataFrame para β
        df_betas = pd.DataFrame(index=fechas, columns=tickers, dtype=float)

        def beta_ventana(r_i_win, r_m_win):
            cov = r_i_win.cov(r_m_win)
            var = r_m_win.var()
            return cov/var if var != 0 else np.nan

        # 5) Para cada fecha desde window-1 en adelante
        for idx in range(window - 1, len(fechas)):
            ventana = all_ret.iloc[idx - window + 1: idx + 1].copy()
            r_m_win = ventana['ret_m']

            # 5.1) Filtrar outliers en IBC (opcional)
            mask_m = ventana['ret_m'].abs() > 0.5
            if mask_m.any():
                ventana.loc[mask_m, 'ret_m'] = np.nan
                ventana['ret_m'] = ventana['ret_m'].ffill().bfill()

            # 5.2) Filtrar outliers en cada ticker
            for t in tickers:
                col_ret = f'ret_{t}'
                mask_i = ventana[col_ret].abs() > 0.5
                if mask_i.any():
                    fechas_extremos = ventana.index[mask_i].tolist()
                    for f_ext in fechas_extremos:
                        if f_ext not in self.filtered_dates[t]:
                            self.filtered_dates[t].append(f_ext)
                    ventana.loc[mask_i, col_ret] = np.nan

                ventana[col_ret] = ventana[col_ret].ffill().bfill()

            # 5.3) Calcular β para cada ticker en la fecha actual
            for t in tickers:
                r_i_win = ventana[f'ret_{t}']
                df_betas.at[fechas[idx], t] = beta_ventana(r_i_win, ventana['ret_m'])

        # 6) Rellenar posibles NaN iniciales/finales
        df_betas.fillna(method='ffill', inplace=True)
        df_betas.fillna(method='bfill', inplace=True)

        return df_betas

    # -------------------------------------------------------
    # E) Selección top10 y carga precios históricos
    # -------------------------------------------------------
    def get_quarter_end_date(self, year, quarter):
        q_map = {'Q1': (3,31), 'Q2': (6,30), 'Q3': (9,30), 'Q4': (12,31)}
        m, d = q_map[quarter]
        return datetime(year, m, d)

    def select_top_tickers(self, year, quarter):
        end_dt = self.get_quarter_end_date(year, quarter)
        mkt_caps = {}
        for t in self.priority_tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df = pd.read_csv(path, parse_dates=['time'])
                df = df.set_index('time').sort_index()
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                mkt_caps[t] = last_row['Mkt cap']
            except Exception:
                continue
        sorted_tickers = sorted(mkt_caps, key=mkt_caps.get, reverse=True)
        return sorted_tickers[:10]

    def get_historical_prices(self, tickers, year, quarter):
        """
        Carga precios diarios del año anterior (252 BDay) al cierre del trimestre.
        Limpia precios con clean_prices antes de reindexar.
        Además, asigna a self.current_betas[t] la β diaria del día anterior al cierre,
        y a self.current_mkt_caps[t] el Mkt cap en la fecha de cierre.
        """
        end_dt = self.get_quarter_end_date(year, quarter)
        start_dt = end_dt - BDay(252)
        all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        df_total = pd.DataFrame(index=all_dates)

        self.current_betas = {}
        self.current_mkt_caps = {}

        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df_raw = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
                df = self.clean_prices(df_raw, threshold=0.5, ticker=t)

                # β diaria al cierre del trimestre: última fecha ≤ end_dt
                fechas_beta = self.daily_betas.index[self.daily_betas.index <= end_dt]
                if not len(fechas_beta):
                    raise ValueError(f"No hay β diaria para {t} antes de {end_dt.date()}")
                fecha_beta = fechas_beta[-1]
                self.current_betas[t] = self.daily_betas.at[fecha_beta, t]

                # Mkt cap al cierre
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                self.current_mkt_caps[t] = last_row['Mkt cap']

                # Rellenar precios en el rango all_dates
                dfq = df.reindex(all_dates, method='ffill')
                df_total[t] = dfq['close']
            except Exception:
                pass

        if df_total.empty:
            raise ValueError("No se pudieron cargar datos históricos para ningún ticker.")
        df_clean = df_total.dropna(thresh=len(tickers)//2)
        if len(df_clean) < 50:
            raise ValueError(f"Datos insuficientes: sólo {len(df_clean)} días completos.")
        return df_clean, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

    # -------------------------------------------------------
    # Método para promediar los retornos reales del IBC
    # -------------------------------------------------------
    def get_historical_avg_quarterly_rm(self, year, quarter, lookback=4):
        """
        Para (year, quarter), obtiene los retornos reales del IBC
        en los `lookback` trimestres anteriores. Ejemplo:
        si (year, quarter) = (2021, 'Q1'), busca:
          (2020, 'Q4'), (2020, 'Q3'), (2020, 'Q2'), (2020, 'Q1').
        Luego promedia los que encuentre. Si no encuentra ninguno,
        retorna None para activar el fallback.
        """
        prev_map = {
            'Q1': ('Q4', -1),
            'Q2': ('Q1',  0),
            'Q3': ('Q2',  0),
            'Q4': ('Q3',  0),
        }

        retornos = []
        y_prev, q_prev = year, quarter

        for _ in range(lookback):
            q_ant, year_dec = prev_map[q_prev]
            y_prev = y_prev + year_dec
            q_prev = q_ant
            try:
                rm_prev = self.get_quarterly_rm(y_prev, q_prev)
                retornos.append(rm_prev)
            except Exception:
                continue

        if not retornos:
            return None

        return sum(retornos) / len(retornos)

    # -------------------------------------------------------
    # F) Retornos CAPM ex-ante usando Rf=CRP y Rm estimado
    # -------------------------------------------------------
    def get_quarterly_rm(self, year, quarter):
        # Calcular Rm trimestral a partir de precios diarios del IBC
        end_dt = self.get_quarter_end_date(year, quarter)
        q_map_start = {'Q1': (1,1), 'Q2': (4,1), 'Q3': (7,1), 'Q4': (10,1)}
        m_s, d_s = q_map_start[quarter]
        quarter_start = datetime(year, m_s, d_s)

        ibc = self.daily_ibc
        idx_start = ibc.index[ibc.index >= quarter_start]
        if not len(idx_start):
            raise ValueError(f"No hay precios IBC desde {quarter_start.date()}")
        first_day = idx_start[0]

        idx_end = ibc.index[ibc.index <= end_dt]
        if not len(idx_end):
            raise ValueError(f"No hay precios IBC antes de {end_dt.date()}")
        last_day = idx_end[-1]

        price_start = ibc.at[first_day, 'close']
        price_end = ibc.at[last_day, 'close']
        return (price_end / price_start) - 1

    def calculate_expected_returns(self, tickers, year, quarter):
        # 1) Obtengo CRP trimestral
        CRP_trim = self.get_crp_trimestral()

        # 2) Intento estimar E[Rm] como promedio de los 4 trimestres anteriores
        Rm_est = self.get_historical_avg_quarterly_rm(year, quarter, lookback=4)
        if Rm_est is None:
            # Si no hay datos previos, usar CRP_trim como proxy
            Rm_est = CRP_trim

        exp_returns = []
        for t in tickers:
            beta_i = self.current_betas.get(t, np.mean(list(self.current_betas.values())))
            if np.isnan(beta_i):
                beta_i = 1.0

            # CAPM adaptado: E[Ri] = CRP_trim + beta_i * (Rm_est - CRP_trim)
            exp_ret_i = CRP_trim + beta_i * (Rm_est - CRP_trim)
            exp_returns.append(exp_ret_i)
        return np.array(exp_returns)

    # -------------------------------------------------------
    # G) Optimización, Sharpe ex-post y CAPM ex-post
    # -------------------------------------------------------
    def optimize_portfolio(self, tickers, prices, year, quarter, initial_weights=None, capital=10000):
        if len(tickers) != prices.shape[1]:
            raise ValueError(f"Error: {len(tickers)} tickers pero {prices.shape[1]} columnas de precios.")

        try:
            # 1) Rellenar faltantes de precios
            if prices.isna().any().any():
                prices = prices.ffill().bfill().fillna(0)

            # 2) Calcular retornos diarios y filtrar outliers >|50%|
            returns = prices.pct_change().dropna()
            mask_outliers = returns.abs().max(axis=1) > 0.5
            if mask_outliers.any():
                returns = returns.loc[~mask_outliers]
                if returns.empty:
                    raise ValueError("Serie de retornos vacía tras eliminar outliers.")

            if len(returns) < 5:
                raise ValueError("Insuficientes datos para retornos diarios.")
            if returns.shape[1] != len(tickers):
                raise ValueError("Discrepancia entre series y tickers.")

            # 3) Matriz de covarianzas
            cov = returns.cov().values

            # 4) Rendimientos esperados ex-ante (CAPM adaptado)
            exp_ret_arr = self.calculate_expected_returns(tickers, year, quarter)

            # 5) Extraer precios de inicio y fin del trimestre
            end_dt = self.get_quarter_end_date(year, quarter)
            q_map_start = {'Q1': (1, 1), 'Q2': (4, 1), 'Q3': (7, 1), 'Q4': (10, 1)}
            m_s, d_s = q_map_start[quarter]
            quarter_start = datetime(year, m_s, d_s)

            # --- precio de compra (primer día hábil del trimestre) ---
            idx_start = prices.index[prices.index >= quarter_start]
            if not len(idx_start):
                raise ValueError(f"No hay datos para inicio de {year}-{quarter}")
            actual_start = idx_start[0]
            init_prices_sim = prices.loc[actual_start].values

            # --- precio de venta (último día hábil del trimestre) ---
            idx_end = prices.index[prices.index <= end_dt]
            if not len(idx_end):
                raise ValueError(f"No hay datos para fin de {year}-{quarter}")
            actual_end = idx_end[-1]
            final_prices_sim = prices.loc[actual_end].values

            # 6) Prima trimestral del país (CRP_trim) y Rf diario (ya no se usará para Sharpe)
            CRP_trim = self.get_crp_trimestral()
            # Rf_daily = CRP_trim / len(returns)   # Ya NO se usa para calcular Sharpe ex-post

            # 7) Optimización de pesos (maximizar Sharpe ex-ante)
            def neg_sharpe_expected(w):
                port_return_exp = np.dot(w, exp_ret_arr)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return_exp - CRP_trim) / (port_vol + 1e-10)

            w0 = np.ones(len(tickers)) / len(tickers)
            if initial_weights is not None and len(initial_weights) == len(tickers):
                w0 = initial_weights

            bnds = [(0.005, 0.30) for _ in tickers]
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            res = minimize(
                neg_sharpe_expected,
                w0,
                method='SLSQP',
                bounds=bnds,
                constraints=cons,
                options={'ftol': 1e-4, 'maxiter': 1000, 'disp': False}
            )
            w = res.x if res.success else w0

            # 8) Beta de cartera y cálculo de cantidades/comisiones
            beta_p = sum(w[i] * self.current_betas[t] for i, t in enumerate(tickers))

            getcontext().prec = 10
            getcontext().rounding = 'ROUND_HALF_UP'
            tc = self.market_params['transaction_cost']
            capital_ajustado = Decimal(str(capital)) / (Decimal('1') + Decimal(str(tc)))
            qty = []
            for i in range(len(tickers)):
                try:
                    q = (Decimal(str(w[i])) * capital_ajustado /
                         Decimal(str(init_prices_sim[i]))).quantize(Decimal('0.0001'))
                    qty.append(q)
                except:
                    qty.append(Decimal('0'))

            init_prices_dec = [Decimal(str(p)).quantize(Decimal('0.01')) for p in init_prices_sim]
            invested = sum(q * p for q, p in zip(qty, init_prices_dec))
            commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
            total_gastado = invested + commission

            if total_gastado > Decimal(str(capital)):
                ajuste = Decimal(str(capital)) / total_gastado
                qty = [(q * ajuste).quantize(Decimal('0.0001')) for q in qty]
                invested = sum(q * p for q, p in zip(qty, init_prices_dec))
                commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
                total_gastado = invested + commission

            final_val = sum(float(q) * float(p) for q, p in zip(qty, final_prices_sim))
            roi = (final_val - float(total_gastado)) / capital

            # 9) Cálculo de Sharpe ex-post CORREGIDO
            #    VOLATILIDAD ANUALIZADA basada en retornos limpios:
            port_returns = returns.dot(w)                   # Retornos diarios de la cartera
            sigma_anual = port_returns.std() * np.sqrt(252) # Desviación estándar anualizada
            #    Numerador: ROI (trimestral) menos CRP_trim
            sharpe_expost = (roi - CRP_trim) / (sigma_anual + 1e-10)

            # 10) CAPM ex-post usando CRP_trim
            Rm_real = self.get_quarterly_rm(year, quarter)
            capm_expost = CRP_trim + beta_p * (Rm_real - CRP_trim)

            return {
                'weights':             w,
                'quantities':          [float(q) for q in qty],
                'initial_prices':      init_prices_sim.tolist(),
                'final_prices':        final_prices_sim.tolist(),
                'beta_portfolio':      beta_p,
                'expected_return_capm': None,    # Se asigna luego en run_quarterly_analysis
                'realized_sharpe':     sharpe_expost,
                'capm_expost':         capm_expost,
                'risk':                np.sqrt(w @ cov @ w),
                'overall_deviation':   sigma_anual,
                'invested':            float(invested),
                'commission':          float(commission),
                'final_value':         final_val,
                'roi':                 roi,
                'total_gastado':       float(total_gastado),
                'tickers':             tickers
            }

        except Exception as e:
            print(f"❌ Error crítico en optimización: {str(e)}")
            traceback.print_exc()
            return {
                'weights':             np.ones(len(tickers))/len(tickers) if tickers else [],
                'quantities':          [0]*len(tickers),
                'error':               str(e),
                'expected_return_capm': 0,
                'realized_sharpe':     0,
                'capm_expost':         0,
                'risk':                0,
                'overall_deviation':   0,
                'roi':                 0,
                'total_gastado':       0,
                'tickers':             tickers
            }

    def run_quarterly_analysis(self, year, quarter, prev_weights=None, prev_tickers=None):
        try:
            top10 = self.select_top_tickers(year, quarter)
            logger.info(f"🔍 Tickers seleccionados: {top10}")

            prices, sd, ed = self.get_historical_prices(top10, year, quarter)
            logger.info(f"📊 Datos cargados: {prices.shape[0]} periodos para {prices.shape[1]} activos")

            opt = self.optimize_portfolio(
                top10, prices, year, quarter,
                prev_weights if prev_tickers == top10 else None
            )

            if 'error' in opt:
                raise ValueError(opt['error'])

            # Reconstruir expected_return_capm ex-ante:
            CRP_trim = self.get_crp_trimestral()
            Rm_est = self.get_historical_avg_quarterly_rm(year, quarter, lookback=4)
            if Rm_est is None:
                Rm_est = CRP_trim
            beta_p = opt['beta_portfolio']
            exp_ret_portfolio = CRP_trim + beta_p * (Rm_est - CRP_trim)
            opt['expected_return_capm'] = exp_ret_portfolio

            initial_weights = None
            if prev_weights is not None and prev_tickers is not None:
                initial_weights = np.zeros(len(top10))
                for i, ticker in enumerate(top10):
                    if ticker in prev_tickers:
                        idx_prev = prev_tickers.index(ticker)
                        initial_weights[i] = prev_weights[idx_prev]
                new_tickers = set(top10) - set(prev_tickers)
                exited = set(prev_tickers) - set(top10)
                if new_tickers and exited:
                    total_exited = sum(prev_weights[prev_tickers.index(t)] for t in exited)
                    weight_new = total_exited / len(new_tickers)
                    for i, ticker in enumerate(top10):
                        if ticker in new_tickers:
                            initial_weights[i] = weight_new

            if len(prices) < 200:
                logger.info(f"⚠️ Advertencia: Sólo {len(prices)} puntos de datos para {year}-{quarter} (mínimo recomendado: 200)")

            # Segunda corrida con initial_weights ajustados
            opt = self.optimize_portfolio(top10, prices, year, quarter, initial_weights)
            # Recalculamos expected_return_capm tras nueva optimización
            beta_p = opt['beta_portfolio']
            exp_ret_portfolio = CRP_trim + beta_p * (Rm_est - CRP_trim)
            opt['expected_return_capm'] = exp_ret_portfolio

            transactions = []
            if prev_weights is not None and prev_tickers is not None:
                for i, ticker in enumerate(top10):
                    if ticker in prev_tickers:
                        idx_prev = prev_tickers.index(ticker)
                        change = opt['weights'][i] - prev_weights[idx_prev]
                        transactions.append((ticker, change))

            return {
                'trimestre':         f"{year}-{quarter}",
                'acciones':          top10,
                'beta':              beta_p,
                'retorno_esperado':  exp_ret_portfolio,
                'riesgo':            opt['risk'],
                'overall_deviation': opt['overall_deviation'],
                'sharpe':            opt['realized_sharpe'],
                'capm_expost':       opt['capm_expost'],
                'roi':               opt['roi'],
                'optimizacion':      opt,
                'fecha_inicio':      sd,
                'fecha_fin':         ed,
                'rm_trimestral':     self.get_quarterly_rm(year, quarter),
                'transactions':      transactions,
                'weights':           opt['weights']
            }

        except Exception as e:
            print(f"❌ Error en {year}-{quarter}: {str(e)}")
            return {
                'trimestre':     f"{year}-{quarter}",
                'acciones':      [],
                'error':         str(e),
                'optimizacion':  None,
                'weights':       prev_weights if prev_weights is not None else np.array([]),
                'transactions':  [],
                'sharpe':        0,
                'capm_expost':   0
            }

    def print_results(self, results):
        """
        Imprime en consola y log los resultados completos, diferenciando:
        - Métricas que son producto del análisis del trimestre actual.
        - Métricas que provienen del cálculo con precios históricos (ventana 1 año).
        - Fechas filtradas por retorno y precio.
        """
        if not results or 'error' in results:
            logger.info("\n" + "="*80)
            logger.info(f"FALLO EN {results.get('trimestre', 'TRIMESTRE DESCONOCIDO')}")
            logger.info("="*80)
            logger.info(f"Error: {results.get('error', 'Desconocido')}")
            logger.info("="*80 + "\n")
            return

        opt = results['optimizacion']
        capital0 = 10000
        year_q, quarter = results['trimestre'].split('-')
        rm = results['rm_trimestral']
        comm_pct = opt['commission'] * 100 / capital0

        # 1) Composición (solo trimestre actual)
        tabla = []
        for i, t in enumerate(results['acciones']):
            beta_val = self.current_betas.get(t, 'N/A')
            tabla.append([
                t,
                f"{opt['weights'][i]*100:.2f}%",
                f"{opt['quantities'][i]:.4f}",
                f"{opt['initial_prices'][i]:.4f}",
                f"{(opt['quantities'][i]*opt['initial_prices'][i]):,.4f}",
                f"{opt['final_prices'][i]:.4f}",
                f"{(opt['final_prices'][i]/opt['initial_prices'][i]-1)*100:.4f}%",
                f"{beta_val:.4f}" if isinstance(beta_val, float) else beta_val
            ])

        # 2) Valores intermedios (trimestre actual)
        inv = opt['invested']
        comm_bs = opt['commission']
        remanente = capital0 - inv - comm_bs
        final_v = opt['final_value']
        ganancia = final_v - inv - comm_bs
        roi_pct = opt['roi'] * 100

        # Rentabilidad CAPM ex-ante (trimestre actual)
        retcapm = results['retorno_esperado'] * 100

        rm_pct = rm * 100
        alpha = roi_pct - rm_pct
        vol_pct = opt['risk'] * 100
        overall_dev = opt['overall_deviation'] * 100

        # Sharpe ex-post (trimestre actual)
        sharpe_expost = opt['realized_sharpe']

        logger.info("\n" + "="*80)
        logger.info(f"RESULTADOS DEL TRIMESTRE: {results['trimestre']}")
        logger.info("="*80)

        # Sección: Métricas del trimestre actual
        logger.info("\n-- Métricas del trimestre actual --\n")
        logger.info("COMPOSICIÓN ÓPTIMA:")
        logger.info(tabulate(tabla,
                             headers=['Ticker','Peso','Cant','Precio Ini','Inv Ini','Precio Fin','Ret%','Beta'],
                             tablefmt='grid'))
        logger.info(f"\nRESUMEN DEL PORTAFOLIO ({results['trimestre']})")
        logger.info("-"*80)

        # 3) Métricas de rendimiento (trimestre)
        logger.info("MÉTRICAS DE RENDIMIENTO (Trimestre):")
        logger.info(f"{'Rendimiento total (ROI)':<40}: {roi_pct:.4f}%")
        logger.info(f"{'CAPM ex-ante (esperado)':<40}: {retcapm:.4f}%")
        logger.info(f"{'Rendimiento IBC (Trimestre)':<40}: {rm_pct:.4f}%")
        logger.info(f"{'α (Alfa)':<40}: {alpha:+.4f}%")
        logger.info("")

        # 4) Montos y comisiones (trimestre)
        logger.info("Montos y Comisiones (Trimestre):")
        logger.info(f"{'Capital disponible (Bs)':<40}: {capital0:,.2f}")
        logger.info(f"{'Comisión aplicada (%)':<40}: {comm_pct:.2f}%")
        logger.info(f"{'Comisión aplicada (Bs)':<40}: {comm_bs:,.2f}")
        logger.info(f"{'Capital invertido (Bs)':<40}: {inv:,.2f}")
        logger.info(f"{'Dinero no invertido (remanente Bs)':<40}: {remanente:,.2f}")
        logger.info(f"{'Valor Final del Portafolio (Bs)':<40}: {final_v:,.2f}")
        logger.info(f"{'Ganancia neta del periodo (Bs)':<40}: {ganancia:,.2f}")
        logger.info("")

        # 5) Métricas de riesgo (trimestre)
        logger.info("MÉTRICAS DE RIESGO (Trimestre):")
        logger.info(f"{'Beta de cartera':<40}: {results['beta']:.4f}")
        logger.info(f"{'Volatilidad diaria del Portafolio':<40}: {vol_pct:.4f}%")
        logger.info(f"{'Overall Deviation (Desv. Anualizada)':<40}: {overall_dev:.4f}%")
        logger.info(f"{'Sharpe Ratio (Ex-Post)':<40}: {sharpe_expost:.4f}")
        logger.info("")

        # Sección: Métricas basadas en precios históricos (1 año previo)
        logger.info("\n-- Métricas basadas en precios históricos (1 año previo) --\n")
        logger.info(f"{'Fecha de inicio histórica':<40}: {results['fecha_inicio']}")
        logger.info(f"{'Fecha de fin histórica':<40}: {results['fecha_fin']}")
        logger.info("")

        # 6) Fechas filtradas
        logger.info("FECHAS FILTRADAS (|retorno| > 50%) por acción (análisis histórico):")
        for t in results['acciones']:
            fechas_r = self.filtered_dates.get(t, [])
            lista_r = ", ".join(f.date().isoformat() for f in sorted(fechas_r)) if fechas_r else "(ninguna)"
            logger.info(f"  • {t}: {lista_r}")
        logger.info("")

        logger.info("FECHAS FILTRADAS (precio con salto > 50%) por acción (análisis histórico):")
        for t in results['acciones']:
            fechas_p = self.filtered_dates_precio.get(t, [])
            lista_p = ", ".join(f.date().isoformat() for f in sorted(fechas_p)) if fechas_p else "(ninguna)"
            logger.info(f"  • {t}: {lista_p}")
        logger.info("\n" + "="*80 + "\n")


if __name__ == "__main__":
    optimizer = PortfolioOptimizer(data_folder='data_acciones', crp_ve_anual=0.25, risk_premium_type='Country')
    logger.info("\nOptimizador Trimestral - Mercado Venezolano\n")

    quarters = [
        (2021, 'Q1'), (2021, 'Q2'), (2021, 'Q3'), (2021, 'Q4'),
        (2022, 'Q1'), (2022, 'Q2'), (2022, 'Q3'), (2022, 'Q4'),
        (2023, 'Q1'), (2023, 'Q2'), (2023, 'Q3'), (2023, 'Q4'),
        (2024, 'Q1'), (2024, 'Q2'), (2024, 'Q3'), (2024, 'Q4')
    ]

    prev_weights, prev_tickers = None, None
    all_results = []

    # Procesar cada trimestre y recolectar resultados
    for year, quarter in quarters:
        # Eliminar manejadores de archivos anteriores
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        # Construir nombre de archivo de log dentro de "reporte trimestre"
        log_filename = f"log_{year}_{quarter}.txt"
        log_path = os.path.join(REPORT_DIR, log_filename)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        script_name = os.path.basename(__file__)
        logger.info("\n" + "="*50)
        logger.info(f"[{script_name}] PROCESANDO TRIMESTRE: {year}-{quarter}")
        logger.info("="*50)

        try:
            results = optimizer.run_quarterly_analysis(
                year, quarter, prev_weights, prev_tickers
            )
            optimizer.print_results(results)
            all_results.append(results)
            prev_weights = results.get('weights', None)
            prev_tickers = results.get('acciones', None)

        except Exception as e:
            logger.error(f"❌ Error procesando {year}-{quarter}: {str(e)}")
            results = {
                'trimestre': f"{year}-{quarter}",
                'error': str(e)
            }
            optimizer.print_results(results)
            all_results.append(results)
            if "Datos insuficientes" in str(e):
                prev_weights, prev_tickers = None, None

        # ------------------------------------------------------------
        # Crear un archivo de Excel por trimestre en "resultados excel"
        # que contenga los dos cuadros solicitados (datos de "inicio" y "fin").
        # ------------------------------------------------------------
        if 'error' not in results:
            trimestre = results['trimestre']
            year_q, q = year, quarter

            # Mapear trimestre a mes de inicio y mes de fin
            start_month_map = {'Q1': 'enero', 'Q2': 'abril', 'Q3': 'julio', 'Q4': 'octubre'}
            end_month_map = {'Q1': 'marzo', 'Q2': 'junio', 'Q3': 'septiembre', 'Q4': 'diciembre'}

            # Datos para el cuadro "fin"
            cf = {
                'fin': trimestre,
                'Fecha': f"{end_month_map[q]} {year_q}",
                'Portafolio Optimizado': round(results['roi'] * 100, 4),
                'Indice IBC': round(results['rm_trimestral'] * 100, 4),
                'Alpha': round((results['roi'] - results['rm_trimestral']) * 100, 4)
            }

            # Datos para el cuadro "inicio"
            # CAPM ex-ante
            capm_exante = results['retorno_esperado']
            beta_port = results['beta']
            rm_ibc = results['rm_trimestral']
            sharpe = results['sharpe']
            desviacion = results['riesgo']

            ci = {
                'inicio': trimestre,
                'Fecha': f"{start_month_map[q]} {year_q}",
                'CAPM': round(capm_exante * 100, 4),
                'Beta': round(beta_port, 4),
                'Rm(ibc)': round(rm_ibc * 100, 4),
                'Sharpe': round(sharpe, 4),
                'Desviacion': round(desviacion * 100, 4)
            }

            # Crear DataFrames de un solo registro
            df_fin = pd.DataFrame([cf])
            df_inicio = pd.DataFrame([ci])

            excel_path = os.path.join(EXCEL_DIR, f"{trimestre}_resumen.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                df_fin.to_excel(writer, sheet_name="Fin", index=False)
                df_inicio.to_excel(writer, sheet_name="Inicio", index=False)

    # ------------------------------------------------------------
    # Finalmente, crear los dos cuadros resumen de todos los trimestres
    # ------------------------------------------------------------
    resumen_fin = []
    resumen_inicio = []
    for res in all_results:
        if 'error' in res:
            continue
        year_q, q = map(str, res['trimestre'].split('-'))
        start_month_map = {'Q1': 'enero', 'Q2': 'abril', 'Q3': 'julio', 'Q4': 'octubre'}
        end_month_map = {'Q1': 'marzo', 'Q2': 'junio', 'Q3': 'septiembre', 'Q4': 'diciembre'}

        resumen_fin.append({
            'fin': res['trimestre'],
            'Fecha': f"{end_month_map[q]} {year_q}",
            'Portafolio Optimizado': round(res['roi'] * 100, 4),
            'Indice IBC': round(res['rm_trimestral'] * 100, 4),
            'Alpha': round((res['roi'] - res['rm_trimestral']) * 100, 4)
        })

        resumen_inicio.append({
            'inicio': res['trimestre'],
            'Fecha': f"{start_month_map[q]} {year_q}",
            'CAPM': round(res['retorno_esperado'] * 100, 4),
            'Beta': round(res['beta'], 4),
            'Rm(ibc)': round(res['rm_trimestral'] * 100, 4),
            'Sharpe': round(res['sharpe'], 4),
            'Desviacion': round(res['riesgo'] * 100, 4)
        })

    df_resumen_fin = pd.DataFrame(resumen_fin)
    df_resumen_inicio = pd.DataFrame(resumen_inicio)

    # Guardar los cuadros resumen en un archivo Excel dentro de "resultados excel"
    resumen_path = os.path.join(EXCEL_DIR, "resumen_trimestres.xlsx")
    with pd.ExcelWriter(resumen_path) as writer:
        df_resumen_fin.to_excel(writer, sheet_name="Resumen Fin", index=False)
        df_resumen_inicio.to_excel(writer, sheet_name="Resumen Inicio", index=False)

    logger.info(f"✅ Todos los archivos de resumen se guardaron en '{EXCEL_DIR}'.")
    logger.info(f"✅ Todos los logs por trimestre se encuentran en '{REPORT_DIR}'.")
