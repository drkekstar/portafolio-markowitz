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
# Configuro un logger que envía todo a consola y a un archivo
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
    def __init__(self, data_folder='data_acciones', risk_premium_type='Country'):
        """
        risk_premium_type: 'Country' para Country Risk Premium,
                           'Equity'  para Equity Risk Premium.
        """
        self.data_folder = data_folder

        # A) Solo guardamos transaction_cost, la prima vendrá de RISK_PREMIUM_VENEZUELA.csv
        self.market_params = {
            'transaction_cost': 0.0275  # 2.75%
        }

        # B) Elegir entre prima Country o Equity
        self.risk_premium_type = risk_premium_type

        # C) Tickers “universo”
        self.priority_tickers = [
            'BPV','MVZ.A','ABC.A','ENV','EFE','CGQ','CRM.A',
            'DOM','MPA','CCR','GZL','FNC','RST.B'
        ]

        # D) Cargo CSV de primas de riesgo (anual)
        self.risk_df = self.load_risk_premiums()

        # E) Cargo precios diarios del IBC y calculo retornos diarios
        self.daily_ibc = self.load_ibc_daily()

        # F) Calculo betas diarias sobre ventana de 252 días para cada ticker
        self.daily_betas = self.calculate_daily_betas(self.priority_tickers, window=252)

        # G) Diccionarios que llenaré cada trimestre con β y mkt cap del top10
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
            # Si hay cadenas con '%', les quitamos el '%' y dividimos por 100
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
    # C) load_ibc_daily: lee IBC_Daily_2020_2024.csv y retorna DataFrame con 'ret_m'
    # -------------------------------------------------------
    def load_ibc_daily(self):
        path = os.path.join(self.data_folder, 'IBC_Daily_2020_2024.csv')
        df = pd.read_csv(path, parse_dates=['time'])
        df = df.set_index('time').sort_index()
        df['ret_m'] = df['close'].pct_change()
        return df[['close', 'ret_m']]


    # -------------------------------------------------------
    # D) calculate_daily_betas: calcula β diaria para cada ticker
    # -------------------------------------------------------
    def calculate_daily_betas(self, tickers, window=252):
        # Retornos diarios del IBC
        df_ibc = self.daily_ibc[['ret_m']].dropna()

        # Retornos diarios de cada acción
        dict_retornos = {}
        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            df_raw = pd.read_csv(path, parse_dates=['time']).set_index('time').sort_index()
            df_t  = self.clean_prices(df_raw, threshold=0.5, ticker=t)   # <<<<< aplicar limpieza
            df_t[f'ret_{t}'] = df_t['close'].pct_change()
            dict_retornos[t] = df_t[[f'ret_{t}']]

        # Concatenar todos los retornos
        all_ret = pd.concat([df_ibc] + list(dict_retornos.values()), axis=1).dropna()
        fechas = all_ret.index
        df_betas = pd.DataFrame(index=fechas, columns=tickers, dtype=float)

        def beta_ventana(r_i, r_m):
            cov = r_i.cov(r_m)
            var = r_m.var()
            return cov / var if var != 0 else np.nan

        # Para cada fecha desde window-1 en adelante, calcular β
        for idx in range(window - 1, len(fechas)):
            ventana = all_ret.iloc[idx - window + 1 : idx + 1]
            r_m_win = ventana['ret_m']
            for t in tickers:
                r_i_win = ventana[f'ret_{t}']
                df_betas.at[fechas[idx], t] = beta_ventana(r_i_win, r_m_win)

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
        end_dt = self.get_quarter_end_date(year, quarter)
        start_dt = end_dt - BDay(252)
        all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        df_total = pd.DataFrame(index=all_dates)

        self.current_betas = {}
        self.current_mkt_caps = {}

        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df = pd.read_csv(path, parse_dates=['time'])
                df = df.set_index('time').sort_index()
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()

                # β diaria al cierre del trimestre
                fechas_beta = self.daily_betas.index[self.daily_betas.index <= end_dt]
                if not len(fechas_beta):
                    raise ValueError(f"No hay β diaria para {t} antes de {end_dt.date()}")
                fecha_beta = fechas_beta[-1]
                self.current_betas[t] = self.daily_betas.at[fecha_beta, t]

                # Mkt cap al cierre
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                self.current_mkt_caps[t] = last_row['Mkt cap']

                # Rellenar precios diarios
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
    # F) Retornos CAPM ex-ante
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
        Rm = self.get_quarterly_rm(year, quarter)
        Rf = self.get_quarter_risk_premium(year, quarter)
        exp_returns = []
        for t in tickers:
            beta_i = self.current_betas.get(t, np.mean(list(self.current_betas.values())))
            if np.isnan(beta_i):
                beta_i = 1.0
            exp_returns.append(Rf + beta_i * (Rm - Rf))
        return np.array(exp_returns)


    # -------------------------------------------------------
    # G) Optimización, Sharpe ex-post y CAPM ex-post
    # -------------------------------------------------------
    def optimize_portfolio(self, tickers, prices, year, quarter, initial_weights=None, capital=10000):
        if len(tickers) != prices.shape[1]:
            raise ValueError(f"Error: {len(tickers)} tickers pero {prices.shape[1]} columnas de precios.")

        try:
            if prices.isna().any().any():
                prices = prices.ffill().bfill().fillna(0)

            returns = prices.pct_change().dropna()
            if len(returns) < 5:
                raise ValueError("Insuficientes datos para retornos diarios.")
            if returns.shape[1] != len(tickers):
                raise ValueError("Discrepancia entre series y tickers.")

            cov = returns.cov().values
            exp_ret_arr = self.calculate_expected_returns(tickers, year, quarter)

            end_dt = self.get_quarter_end_date(year, quarter)
            q_map_start = {'Q1': (1,1), 'Q2': (4,1), 'Q3': (7,1), 'Q4': (10,1)}
            m_s, d_s = q_map_start[quarter]
            quarter_start = datetime(year, m_s, d_s)

            idx_start = prices.index[prices.index >= quarter_start]
            if not len(idx_start):
                raise ValueError(f"No hay datos para inicio de {year}-{quarter}")
            actual_start = idx_start[0]
            init_prices_sim = prices.loc[actual_start].values

            idx_end = prices.index[prices.index <= end_dt]
            if not len(idx_end):
                raise ValueError(f"No hay datos para fin de {year}-{quarter}")
            actual_end = idx_end[-1]
            final_prices_sim = prices.loc[actual_end].values

            Rf_trimestral = self.get_quarter_risk_premium(year, quarter)
            num_business = len(returns)
            Rf_daily = Rf_trimestral / num_business

            def neg_sharpe_expected(w):
                port_return_exp = np.dot(w, exp_ret_arr)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return_exp - Rf_trimestral) / (port_vol + 1e-10)

            w0 = np.ones(len(tickers)) / len(tickers)
            if initial_weights is not None and len(initial_weights) == len(tickers):
                w0 = initial_weights

            bnds = [(0.005, 0.30) for _ in tickers]
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            res = minimize(neg_sharpe_expected, w0, method='SLSQP',
                           bounds=bnds, constraints=cons,
                           options={'ftol': 1e-4, 'maxiter': 1000, 'disp': False})
            w = res.x if res.success else w0

            beta_p = sum(w[i] * self.current_betas[t] for i, t in enumerate(tickers))
            exp_ret_portfolio = Rf_trimestral + beta_p * (self.get_quarterly_rm(year, quarter) - Rf_trimestral)

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

            port_returns = returns.dot(w)
            ret_excess = port_returns - Rf_daily
            mean_excess = ret_excess.mean()
            std_port = port_returns.std()
            if std_port > 0:
                sharpe_daily = mean_excess / std_port
                sharpe_expost = sharpe_daily * np.sqrt(252)
            else:
                sharpe_expost = 0.0

            Rm_real = self.get_quarterly_rm(year, quarter)
            capm_expost = Rf_trimestral + beta_p * (Rm_real - Rf_trimestral)

            return {
                'weights':             w,
                'quantities':          [float(q) for q in qty],
                'initial_prices':      init_prices_sim.tolist(),
                'final_prices':        final_prices_sim.tolist(),
                'beta_portfolio':      beta_p,
                'expected_return':     exp_ret_portfolio,
                'expected_return_capm': exp_ret_portfolio,
                'realized_sharpe':     sharpe_expost,
                'sharpe':              sharpe_expost,
                'capm_expost':         capm_expost,
                'risk':                np.sqrt(w @ cov @ w),
                'overall_deviation':   port_returns.std() * np.sqrt(252),
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
                'expected_return':     0,
                'expected_return_capm': 0,
                'realized_sharpe':     0,
                'sharpe':              0,
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

            opt = self.optimize_portfolio(top10, prices, year, quarter, initial_weights)

            transactions = []
            if prev_weights is not None and prev_tickers is not None:
                for i, ticker in enumerate(top10):
                    if ticker in prev_tickers:
                        idx_prev = prev_tickers.index(ticker)
                        change = opt['weights'][i] - prev_weights[idx_prev]
                        transactions.append((ticker, change))

            return {
                'trimestre':        f"{year}-{quarter}",
                'acciones':         top10,
                'beta':             opt['beta_portfolio'],
                'retorno_esperado': opt['expected_return_capm'],
                'riesgo':           opt['risk'],
                'overall_deviation':opt['overall_deviation'],
                'sharpe':           opt['realized_sharpe'],
                'capm_expost':      opt['capm_expost'],
                'roi':              opt['roi'],
                'optimizacion':     opt,
                'fecha_inicio':     sd,
                'fecha_fin':        ed,
                'rm_trimestral':    self.get_quarterly_rm(year, quarter),
                'transactions':     transactions,
                'weights':          opt['weights']
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
        if not results or 'error' in results:
            logger.info("\n" + "="*80)
            logger.info(f"FALLO EN {results.get('trimestre', 'TRIMESTRE DESCONOCIDO')}")
            logger.info("="*80)
            logger.info(f"Error: {results.get('error', 'Desconocido')}")
            logger.info("="*80 + "\n")
            return

        opt = results['optimizacion']
        capital0 = 10000
        rf_tr = self.get_quarter_risk_premium(
            int(results['trimestre'].split('-')[0]),
            results['trimestre'].split('-')[1]
        )
        rm = results['rm_trimestral']
        comm_pct = opt['commission'] * 100 / capital0

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

        inv = opt['invested']
        comm_bs = opt['commission']
        remanente = capital0 - inv - comm_bs
        final_v = opt['final_value']
        ganancia = final_v - inv - comm_bs
        roi_pct = opt['roi'] * 100
        retcapm = results['retorno_esperado'] * 100
        rm_pct = rm * 100

        alpha = roi_pct - rm_pct

        vol_pct = opt['risk'] * 100
        overall_dev = opt['overall_deviation'] * 100
        sharpe_expost = opt['realized_sharpe']
        capm_expost_pct = results['capm_expost'] * 100

        logger.info("\n" + "="*80)
        logger.info(f"RESULTADOS {results['trimestre']}  {results['fecha_inicio']} → {results['fecha_fin']}")
        logger.info("="*80)
        logger.info("COMPOSICIÓN ÓPTIMA:")
        logger.info(tabulate(tabla,
                            headers=['Ticker','Peso','Cant','Precio Ini','Inv Ini','Precio Fin','Ret%','Beta'],
                            tablefmt='grid'))

        logger.info(f"\nRESUMEN DEL PORTAFOLIO ({results['trimestre']})")
        logger.info("-"*80)

        logger.info("MÉTRICAS DE RENDIMIENTO (Trimestre):")
        logger.info(f"{'Rendimiento total (ROI)':<40}: {roi_pct:.4f}%")
        logger.info(f"{'CAPM ex-ante (esperado)':<40}: {retcapm:.4f}%")
        logger.info(f"{'Rendimiento IBC (Trimestre)':<40}: {rm_pct:.4f}%")
        logger.info(f"{'α (Alfa)':<40}: {alpha:+.4f}%")
        logger.info("")

        logger.info("Montos y Comisiones:")
        logger.info(f"{'Capital disponible (Bs)':<40}: {capital0:,.2f}")
        logger.info(f"{'Comisión aplicada (%)':<40}: {comm_pct:.2f}%")
        logger.info(f"{'Comisión aplicada (Bs)':<40}: {comm_bs:,.2f}")
        logger.info(f"{'Capital invertido (Bs)':<40}: {inv:,.2f}")
        logger.info(f"{'Dinero no invertido (remanente Bs)':<40}: {remanente:,.2f}")
        logger.info(f"{'Valor Final del Portafolio (Bs)':<40}: {final_v:,.2f}")
        logger.info(f"{'Ganancia neta del periodo (Bs)':<40}: {ganancia:,.2f}")
        logger.info("")

        logger.info("MÉTRICAS DE RIESGO (Ventana histórica 1 año):")
        logger.info(f"{'Beta de cartera':<40}: {results['beta']:.4f}")
        logger.info(f"{'Volatilidad diaria del Portafolio':<40}: {vol_pct:.4f}%")
        logger.info(f"{'Overall Deviation (Desv. Anualizada)':<40}: {overall_dev:.4f}%")
        logger.info(f"{'Sharpe Ratio (Ex-Post)':<40}: {sharpe_expost:.4f}")
        logger.info(f"{'CAPM ex-post (realizado)':<40}: {capm_expost_pct:.4f}%")
        logger.info("")

        logger.info("FÓRMULA CAPM UTILIZADA:")
        logger.info("  E(Ri) = Rf + βi × (E(Rm) − Rf)")
        logger.info(f"  Rf (trimestral usado): {rf_tr*100:.2f}%")
        logger.info(f"  E(Rm) IBC (real):     {rm_pct:.2f}%")

        if results.get('transactions'):
            trans_table = []
            for ticker, change in results['transactions']:
                trans_table.append([
                    ticker,
                    "COMPRAR" if change > 0 else "VENDER",
                    f"{abs(change)*100:.4f}%"
                ])
            logger.info("\nCAMBIOS DE CARTERA RESPECTO AL TRIMESTRE ANTERIOR:")
            logger.info(tabulate(trans_table,
                                headers=['Acción','Operación','Cambio de Ponderación'],
                                tablefmt='grid'))
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    # Puedes cambiar a 'Equity' si prefieres usar la Equity Risk Premium en lugar de Country.
    optimizer = PortfolioOptimizer(data_folder='data_acciones', risk_premium_type='Country')
    logger.info("\nOptimizador Trimestral - Mercado Venezolano\n")

    quarters = [
        (2021, 'Q1'), (2021, 'Q2'), (2021, 'Q3'), (2021, 'Q4'),
        (2022, 'Q1'), (2022, 'Q2'), (2022, 'Q3'), (2022, 'Q4'),
        (2023, 'Q1'), (2023, 'Q2'), (2023, 'Q3'), (2023, 'Q4'),
        (2024, 'Q1'), (2024, 'Q2'), (2024, 'Q3'), (2024, 'Q4')
    ]

    prev_weights, prev_tickers = None, None
    all_results = []

    for year, quarter in quarters:
        # Cada trimestre tendrá su propio archivo de log
        for handler in list(logger.handlers):
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        log_filename = f"log_{year}-{quarter}.txt"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info("\n" + "="*50)
        logger.info(f"PROCESANDO TRIMESTRE: {year}-{quarter}")
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
