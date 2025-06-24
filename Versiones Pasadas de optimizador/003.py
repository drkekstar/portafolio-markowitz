import numpy as np
import pandas as pd
import os
import logging
import sys
from datetime import datetime, timedelta
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

# Handler para escribir en pantalla (stdout)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# (El FileHandler lo añadiremos dinámicamente dentro del bucle
#  principal, para cada trimestre, de modo que cada trimestre tenga su propio archivo)
# ------------------------------------------------------------

# Ignorar warnings específicos de pandas
warnings.filterwarnings('ignore', category=UserWarning, message='Could not infer format')

class PortfolioOptimizer:
    def __init__(self, data_folder='data_acciones'):
        # Parámetros del mercado
        self.market_params = {
            'Rf_VEN': 0.069775,      # 6.9775% trimestral # 27.91% anualizado
            'transaction_cost': 0.0275  # 2.75%
        }
        self.priority_tickers = ['BPV', 'MVZ.A', 'ABC.A', 'ENV', 'EFE', 'CGQ', 'CRM.A', 'DOM', 'MPA', 'CCR', 'GZL', 'FNC', 'RST.B']
        self.data_folder = data_folder
        self.ibc_data = self.load_ibc_data()
        self.current_betas = {}
        self.current_mkt_caps = {}

    def load_ibc_data(self):
        """Carga los retornos trimestrales del IBC desde CSV."""
        path = os.path.join(self.data_folder, 'IBC_Tri_2020_2024.csv')
        df = pd.read_csv(path)
        if 'TIME' not in df.columns or 'RM Trimestral' not in df.columns:
            raise ValueError("IBC_Tri_2020_2024.csv debe tener columnas 'TIME' y 'RM Trimestral'")
        df['RM Trimestral'] = df['RM Trimestral'].str.rstrip('%').astype(float) / 100
        return df

    def get_quarterly_rm(self, year, quarter):
        """Retorna E(Rm) del IBC para el trimestre dado."""
        qkey = f"{year}{quarter}"
        row = self.ibc_data[self.ibc_data['TIME'] == qkey]
        if row.empty:
            raise ValueError(f"No se encontró Rm para {qkey}")
        return row.iloc[0]['RM Trimestral']

    def get_quarter_end_date(self, year, quarter):
        """Retorna datetime (no string)"""
        q_map = {'Q1': (3, 31), 'Q2': (6, 30), 'Q3': (9, 30), 'Q4': (12, 31)}
        month, day = q_map[quarter]
        return datetime(year, month, day)

    from pandas.tseries.offsets import BDay

    def get_historical_prices(self, tickers, year, quarter):
        """Carga precios de cierre del año anterior al trimestre dado, usando días hábiles."""
        # 1) Determinar fecha final del trimestre (último día hábil de ese Q)
        end_dt = self.get_quarter_end_date(year, quarter)

        # 2) Restar 252 días hábiles para aproximar un año de datos de trading
        start_dt = end_dt - BDay(252)

        # 3) Crear índice de días hábiles desde start_dt hasta end_dt
        all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        df_total = pd.DataFrame(index=all_dates)

        # Convertir a string (para mensajes) y timestamp (no cambia casi nada aquí)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # (Aunque start_ts y end_ts ya son prácticamente iguales a start_dt y end_dt, 
        #  los mantenemos por compatibilidad con el resto del código)

        # Diccionarios para almacenar betas y mkt caps
        self.current_betas = {}
        self.current_mkt_caps = {}

        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df = pd.read_csv(path, parse_dates=['time'])
                df = df.set_index('time').sort_index()
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index()

                # 4) Obtener el último registro válido hasta end_dt
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                self.current_betas[t] = last_row['Beta']
                self.current_mkt_caps[t] = last_row['Mkt cap']

                # 5) Rellenar precios hacia adelante para tener datos en cada día hábil
                dfq = df.reindex(all_dates, method='ffill')
                df_total[t] = dfq['close']

                if dfq.empty:
                    print(f"⚠️ Advertencia: no hay datos para {t} en {start_date} a {end_date}")
                    continue

                # 6) Asegurar continuidad: ya hicimos reindex + ffill, así que basta
                df_total[t] = dfq['close']

            except Exception as e:
                print(f"⚠️ Error procesando {t}: {str(e)}")

        if df_total.empty:
            raise ValueError("No se pudieron cargar datos históricos para ningún ticker.")

        # 7) Limpiar filas donde falten demasiados valores
        threshold = len(tickers) * 0.3  # Permitir hasta 70% de datos faltantes por fila
        df_clean = df_total.dropna(thresh=len(tickers)//2)

        # 8) Verificar si hay al menos 50 días hábiles completos
        if len(df_clean) < 50:
            raise ValueError(f"Datos insuficientes. Solo {len(df_clean)} días completos")

        return df_clean, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

    
    def validate_loaded_data(self, prices, year, quarter):
        """Valida que el DataFrame 'prices' (ventana de un año) esté correctamente cargado:
        - Muestra fecha de inicio/fin real (índices).
        - Imprime las 3 primeras y últimas filas.
        - Cuenta valores faltantes por ticker.
        - Verifica que existan fechas al inicio y fin del trimestre.

        prices: DataFrame resultante de get_historical_prices(...)
        year, quarter: para comprobar que el índice cubre el trimestre
        """
        # 1) Mostrar rango de fechas del DataFrame
        fecha_min = prices.index.min()
        fecha_max = prices.index.max()
        print("🗓 Ventana completa de precios:")
        print(f"   Fecha mínima en índice: {fecha_min.date()}")
        print(f"   Fecha máxima en índice: {fecha_max.date()}")
        print(f"   Total de días hábiles cargados: {len(prices)}\n")

        # 2) Mostrar primeras 3 filas y últimas 3 filas
        print("📋 Primeras 3 filas de 'prices':")
        print(prices.head(3).to_string())
        print("\n📋 Últimas 3 filas de 'prices':")
        print(prices.tail(3).to_string())
        print("\n")

        # 3) Contar valores faltantes por columna
        missing_per_ticker = prices.isna().sum()
        print("⚠️ Valores faltantes por ticker (debe ser 0 o muy pocos):")
        for ticker, missing in missing_per_ticker.items():
            print(f"   {ticker}: {missing}")
        print("\n")

        # 4) Verificar que la primera fecha del trimestre esté en el índice
        #    y que la última fecha del trimestre esté en el índice
        end_dt = self.get_quarter_end_date(year, quarter)
        q_map_start = {'Q1': (1, 1), 'Q2': (4, 1), 'Q3': (7, 1), 'Q4': (10, 1)}
        m_s, d_s = q_map_start[quarter]
        quarter_start_dt = datetime(year, m_s, d_s)

        # Buscar fecha de compra (primer día hábil >= quarter_start_dt)
        idx_compra = prices.index[prices.index >= quarter_start_dt]
        if len(idx_compra) == 0:
            print(f"❌ No se encontró ninguna fecha ≥ {quarter_start_dt.date()} en el índice.")
        else:
            print(f"✅ Fecha de inicio del trimestre {year}-{quarter} en 'prices': {idx_compra[0].date()}")

        # Buscar fecha de venta (último día hábil ≤ end_dt)
        idx_venta = prices.index[prices.index <= end_dt]
        if len(idx_venta) == 0:
            print(f"❌ No se encontró ninguna fecha ≤ {end_dt.date()} en el índice.")
        else:
            print(f"✅ Fecha de fin del trimestre {year}-{quarter} en 'prices': {idx_venta[-1].date()}")
        print("\n— Fin de validación —\n")

    def select_top_tickers(self, year, quarter):
        """Selecciona las 10 acciones con mayor capitalización de mercado."""
        end_dt = self.get_quarter_end_date(year, quarter)
        mkt_caps = {}
        
        for t in self.priority_tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df = pd.read_csv(path, parse_dates=['time'])
                df = df.set_index('time').sort_index()
                
                # Encontrar el valor más cercano al final del trimestre
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                mkt_caps[t] = last_row['Mkt cap']
            except Exception as e:
                print(f"⚠️ Excluyendo {t}: {str(e)}")
                continue   
            except Exception as e:
                print(f"⚠️ Error obteniendo capitalización para {t}: {str(e)}")
        # Aceptar incluso si tenemos menos de 10 acciones
        sorted_tickers = sorted(mkt_caps, key=mkt_caps.get, reverse=True)
        return sorted_tickers[:min(10, len(sorted_tickers))]  # Tomar hasta 10

    def calculate_expected_returns(self, tickers, year, quarter):
        """Calcula E(Ri) por CAPM local usando betas actuales."""
        Rf = self.market_params['Rf_VEN']
        Rm = self.get_quarterly_rm(year, quarter)
        
        exp_returns = []
        for t in tickers:
            beta = self.current_betas.get(t, np.mean(list(self.current_betas.values())))
            # Si beta llega como NaN o None, lo forzamos a 1.0
            if np.isnan(beta) or beta is None:
                beta = 1.0
            exp_returns.append(Rf + beta * (Rm - Rf))
        
        return np.array(exp_returns)

    def optimize_portfolio(self, tickers, prices, year, quarter, initial_weights=None, capital=10000):
        """Optimiza el portafolio maximizando el ratio de Sharpe con validación robusta de dimensiones"""
            
        # 1. Validación inicial de dimensiones
        if len(tickers) != prices.shape[1]:
            raise ValueError(f"Error crítico: {len(tickers)} tickers pero {prices.shape[1]} columnas de precios. Tickers: {tickers}")

        try:
            # 2. Manejo de datos faltantes
            if prices.isna().any().any():
                print("⚠️ Advertencia: Datos faltantes detectados. Aplicando imputación...")
                prices = prices.ffill().bfill().fillna(0)

            # 3. Cálculo de retornos con validación
            returns = prices.pct_change().dropna()
            if len(returns) < 5:
                raise ValueError("Insuficientes datos para calcular retornos (mínimo 5 puntos requeridos)")

            # 4. Verificación final de dimensiones antes de optimización
            if returns.shape[1] != len(tickers):
                raise ValueError(f"Discrepancia final: {returns.shape[1]} series de retorno vs {len(tickers)} tickers")

            cov = returns.cov().values
            exp_ret = self.calculate_expected_returns(tickers, year, quarter)

            # =====================================================
            # 5. — NUEVO BLOQUE: extraer precios de compra/venta
            end_dt = self.get_quarter_end_date(year, quarter)
            q_map_start = {'Q1': (1, 1), 'Q2': (4, 1), 'Q3': (7, 1), 'Q4': (10, 1)}
            m_s, d_s = q_map_start[quarter]
            quarter_start_dt = datetime(year, m_s, d_s)

            # Precio de compra: primer día hábil del trimestre
            _idx_compra = prices.index[prices.index >= quarter_start_dt]
            if len(_idx_compra) == 0:
                raise ValueError(f"No hay datos de precios para el inicio de {year}-{quarter}")
            actual_start = _idx_compra[0]
            init_prices_sim = prices.loc[actual_start].values

            # Precio de venta: último día hábil del trimestre
            _idx_venta = prices.index[prices.index <= end_dt]
            if len(_idx_venta) == 0:
                raise ValueError(f"No hay datos de precios para el fin de {year}-{quarter}")
            actual_end = _idx_venta[-1]
            final_prices_sim = prices.loc[actual_end].values
            # =====================================================

            Rf = self.market_params['Rf_VEN']
            tc = self.market_params['transaction_cost']

            # 6. Configuración de pesos iniciales con validación
            w0 = np.ones(len(tickers)) / len(tickers)  # Default equal weights
            if initial_weights is not None:
                if len(initial_weights) == len(tickers):
                    w0 = initial_weights
                else:
                    print(f"⚠️ initial_weights ignorado: dimensión {len(initial_weights)} != {len(tickers)}")

            # 7. Función objetivo con protección contra divisiones por cero
            def neg_sharpe(w):
                port_return = np.dot(w, exp_ret)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return - Rf) / (port_vol + 1e-10)  # Evita división por cero

            # 8. Configuración de restricciones
            bnds = [(0.005, 0.30) for _ in tickers]
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            # 9. Optimización con manejo de fallos
            res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bnds, constraints=cons,
                        options={'ftol': 1e-4, 'maxiter': 1000, 'disp': False})

            if not res.success:
                print(f"⚠️ Optimización subóptima: {res.message}. Usando pesos iniciales como fallback.")
                w = w0
            else:
                w = res.x

            # ————————————————————————————————
            # >>> NUEVO: calcular β de cartera (β_p)
            beta_p = sum(w[i] * self.current_betas[tickers[i]] for i in range(len(tickers)))
            # y a partir de ahí, el retorno CAPM “de cartera”:
            Rf = self.market_params['Rf_VEN']
            Rm = self.get_quarterly_rm(year, quarter)
            exp_ret_portfolio = Rf + beta_p * (Rm - Rf)
            # ————————————————————————————————   

            # 10. Cálculo preciso de cantidades con Decimal (USANDO init_prices_sim)
            getcontext().prec = 10
            getcontext().rounding = 'ROUND_HALF_UP'
            capital_ajustado = Decimal(str(capital)) / (Decimal('1') + Decimal(str(tc)))
            qty = []
            for i in range(len(tickers)):
                try:
                    q = (Decimal(str(w[i])) * capital_ajustado / 
                        Decimal(str(init_prices_sim[i]))).quantize(Decimal('0.0001'))
                    qty.append(q)
                except:
                    print(f"⚠️ Error cálculo cantidad para {tickers[i]}. Usando 0.")
                    qty.append(Decimal('0'))

            # 11. Cálculo de montos invertidos (USANDO init_prices_sim)
            init_prices_dec = [Decimal(str(p)).quantize(Decimal('0.01')) for p in init_prices_sim]
            invested = sum(q * p for q, p in zip(qty, init_prices_dec))
            commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
            total_gastado = invested + commission

            # 12. Ajuste final de capital si es necesario
            if total_gastado > Decimal(str(capital)):
                adjustment = Decimal(str(capital)) / total_gastado
                qty = [q * adjustment for q in qty]
                qty = [q.quantize(Decimal('0.0001')) for q in qty]
                invested = sum(q * p for q, p in zip(qty, init_prices_dec))
                commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
                total_gastado = invested + commission

            # 13. Validación final
            if float(total_gastado) > capital + 0.01:
                print(f"⚠️ Ajuste de capital insuficiente: {float(total_gastado):.2f} > {capital}")

            # 14. Cálculo de métricas de desempeño
            #     • Usamos final_prices_sim para la venta al final del trimestre
            final_val = sum(float(q) * float(p) for q, p in zip(qty, final_prices_sim))
            roi = (final_val - float(total_gastado)) / capital

            port_returns = returns.dot(w)
            overall_deviation = port_returns.std() * np.sqrt(252)  # Anualizada

            return {
                'weights': w,
                'quantities': [float(q) for q in qty],
                'initial_prices': init_prices_sim.tolist(),   # ← previo: prices.iloc[0]
                'final_prices': final_prices_sim.tolist(),     # ← previo: prices.iloc[-1]
                'beta_portfolio': beta_p,                  # Nuevo campo: β de cartera
                'expected_return': exp_ret_portfolio,      # Retorno CAPM usando β_p
                'expected_return': np.dot(w, exp_ret),
                'risk': np.sqrt(w @ cov @ w),
                'overall_deviation': overall_deviation,
                'sharpe': -neg_sharpe(w),
                'invested': float(invested),
                'commission': float(commission),
                'final_value': final_val,
                'roi': roi,
                'total_gastado': float(total_gastado),
                'tickers': tickers  # Para referencia de debugging
            }

        except Exception as e:
            print(f"❌ Error crítico en optimización: {str(e)}")
            traceback.print_exc()  # Para diagnóstico
            return {
                'weights': np.ones(len(tickers))/len(tickers) if tickers else [],
                'quantities': [0]*len(tickers),
                'error': str(e),
                'expected_return': 0,
                'risk': 0,
                'overall_deviation': 0,
                'sharpe': 0,
                'invested': 0,
                'commission': 0,
                'final_value': 0,
                'roi': 0,
                'total_gastado': 0,
                'tickers': tickers
            }

    def run_quarterly_analysis(self, year, quarter, prev_weights=None, prev_tickers=None):
        try:
            top10 = self.select_top_tickers(year, quarter)
            print(f"🔍 Tickers seleccionados: {top10}")
            
            prices, sd, ed = self.get_historical_prices(top10, year, quarter)
            print(f"📊 Datos cargados: {prices.shape[0]} periodos para {prices.shape[1]} activos")
            
            opt = self.optimize_portfolio(top10, prices, year, quarter, 
                                        prev_weights if prev_tickers == top10 else None)
            
            if 'error' in opt:
                raise ValueError(opt['error'])
            # 1. Selección robusta de tickers
            top10 = self.select_top_tickers(year, quarter)
            if not top10:
                raise ValueError("No se pudo seleccionar acciones para optimización")
    
            # 2. Manejo de pesos iniciales para continuidad entre trimestres
            initial_weights = None
            if prev_weights is not None and prev_tickers is not None:
                initial_weights = np.zeros(len(top10))
                
                # Mapeo de acciones persistentes
                for i, ticker in enumerate(top10):
                    if ticker in prev_tickers:
                        idx_prev = prev_tickers.index(ticker)
                        initial_weights[i] = prev_weights[idx_prev]
                
                # Manejo de nuevas acciones
                new_tickers = set(top10) - set(prev_tickers)
                exited_tickers = set(prev_tickers) - set(top10)
                
                if new_tickers and exited_tickers:
                    # Redistribuir pesos de acciones que salieron
                    total_exited_weight = sum(prev_weights[prev_tickers.index(t)] for t in exited_tickers)
                    weight_per_new = total_exited_weight / len(new_tickers)
                    
                    for i, ticker in enumerate(top10):
                        if ticker in new_tickers:
                            initial_weights[i] = weight_per_new
            
            # 3. Obtener precios históricos con manejo de errores integrado
            prices, sd, ed = self.get_historical_prices(top10, year, quarter)
            
            if prices.empty:
                raise ValueError("No hay datos suficientes para optimizar")

             # 3. Optimización con fallback
            opt_result = self.optimize_portfolio(top10, prices, year, quarter, 
                                           prev_weights if prev_tickers == top10 else None)
        
            if 'error' in opt_result:
                raise ValueError(f"Error en optimización: {opt_result['error']}")
            
            # 4. Validación de datos mínimos
            min_data_points = 200
            if len(prices) < min_data_points:
                print(f"⚠️ Advertencia: Solo {len(prices)} puntos de datos para {year}-{quarter} (mínimo recomendado: {min_data_points})")
            
            # 5. Optimizar portafolio
            opt = self.optimize_portfolio(top10, prices, year, quarter, initial_weights)
            
            # 6. Cálculo de transacciones necesarias
            transactions = []
            if prev_weights is not None:
                for i, ticker in enumerate(top10):
                    if ticker in prev_tickers:
                        idx_prev = prev_tickers.index(ticker)
                        change = opt['weights'][i] - prev_weights[idx_prev]
                        transactions.append((ticker, change))
            
            # 7. Calcular beta del portafolio
            beta_p = sum(opt['weights'][i] * self.current_betas[t] for i, t in enumerate(top10))
            
            # 8. Retornar resultados completos
            return {
                'trimestre': f"{year}-{quarter}",
                'acciones': top10,
                'beta': opt['beta_portfolio'],   
                'retorno_esperado': opt['expected_return'],
                'riesgo': opt['risk'],
                'overall_deviation': opt['overall_deviation'],
                'sharpe': opt['sharpe'],
                'roi': opt['roi'],
                'optimizacion': opt,
                'fecha_inicio': sd,
                'fecha_fin': ed,
                'rm_trimestral': self.get_quarterly_rm(year, quarter),
                'transactions': transactions,
                'weights': opt['weights']  # Guardar para siguiente trimestre
            }
        
        except Exception as e:
            print(f"❌ Error en {year}-{quarter}: {str(e)}")
            return {
                'trimestre': f"{year}-{quarter}",
                'acciones': [],
                'error': str(e),
                'optimizacion': None,
                'weights': prev_weights if prev_weights is not None else np.array([]),
                'transactions': []
        }
    def print_results(self, results):
        """Imprime en consola (y en log) el detalle completo, con trimestre y comparación IBC vs portafolio."""
        if not results or 'error' in results:
            logger.info("\n" + "="*80)
            logger.info(f"FALLO EN {results.get('trimestre', 'TRIMESTRE DESCONOCIDO')}")
            logger.info("="*80)
            logger.info(f"Error: {results.get('error', 'Desconocido')}")
            logger.info("="*80 + "\n")
            return

        opt = results['optimizacion']
        capital0 = 10000
        rf = self.market_params['Rf_VEN']
        rm = results['rm_trimestral']                   # Retorno trimestral IBC en forma decimal
        comm_pct = opt['commission'] * 100 / capital0

        # 1) Composición del portafolio (sin cambios)
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

        # 2) Cálculo de valores intermedios
        inv = opt['invested']
        comm_bs = opt['commission']
        remanente = capital0 - inv - comm_bs
        final_v = opt['final_value']
        ganancia = final_v - inv - comm_bs
        roi_pct = opt['roi'] * 100                    # ROI del portafolio en %
        retcapm = results['retorno_esperado'] * 100    # Retorno CAPM en %
        rm_pct = rm * 100                              # Retorno real IBC en %

        vol_pct = opt['risk'] * 100                    # Volatilidad diaria en %
        overall_dev = opt['overall_deviation'] * 100   # Desv. anualizada en %
        shar = opt['sharpe']                           # Sharpe ratio

        # ============================================================
        # 3) Imprimir composición (igual que antes)
        logger.info("\n" + "="*80)
        logger.info(f"RESULTADOS {results['trimestre']}  {results['fecha_inicio']} → {results['fecha_fin']}")
        logger.info("="*80)
        logger.info("COMPOSICIÓN ÓPTIMA:")
        logger.info(tabulate(tabla,
                            headers=['Ticker','Peso','Cant','Precio Ini','Inv Ini','Precio Fin','Ret%','Beta'],
                            tablefmt='grid'))

        # ============================================================
        # 4) RESUMEN DEL PORTAFOLIO, ahora incluyendo trimestre en el título
        logger.info(f"\nRESUMEN DEL PORTAFOLIO ({results['trimestre']})")
        logger.info("-"*80)

        # 4.1) Métricas de rendimiento para EL TRIMESTRE
        logger.info("MÉTRICAS DE RENDIMIENTO (Trimestre):")
        logger.info(f"{'Rendimiento total (ROI)':<40}: {roi_pct:.4f}%")
        logger.info(f"{'Retorno Esperado (CAPM)':<40}: {retcapm:.4f}%")
        logger.info(f"{'Rendimiento IBC (Trimestre)':<40}: {rm_pct:.4f}%")
        logger.info(f"{'Diferencia ROI – IBC':<40}: {roi_pct - rm_pct:+.4f}%")
        logger.info("")  # línea en blanco para separar

        # 4.2) Montos invertidos y comisiones (también aplican al trimestre)
        logger.info("Montos y Comisiones:")
        logger.info(f"{'Capital disponible (Bs)':<40}: {capital0:,.2f}")
        logger.info(f"{'Comisión aplicada (%)':<40}: {comm_pct:.2f}%")
        logger.info(f"{'Comisión aplicada (Bs)':<40}: {comm_bs:,.2f}")
        logger.info(f"{'Capital invertido (Bs)':<40}: {inv:,.2f}")
        logger.info(f"{'Dinero no invertido (remanente Bs)':<40}: {remanente:,.2f}")
        logger.info(f"{'Valor Final del Portafolio (Bs)':<40}: {final_v:,.2f}")
        logger.info(f"{'Ganancia neta del periodo (Bs)':<40}: {ganancia:,.2f}")
        logger.info("")  # línea en blanco para separar

        # 4.3) Métricas de riesgo (basadas en la ventana histórica de 1 año)
        logger.info("MÉTRICAS DE RIESGO (Ventana histórica 1 año):")
        logger.info(f"{'Beta de cartera':<40}: {results['beta']:.4f}")
        logger.info(f"{'Volatilidad diaria del Portafolio':<40}: {vol_pct:.4f}%")
        logger.info(f"{'Overall Deviation (Desv. Anualizada)':<40}: {overall_dev:.4f}%")
        logger.info(f"{'Sharpe Ratio':<40}: {shar:.4f}")
        logger.info("")  # línea en blanco para separar

        # 4.4) Mostrar fórmula CAPM (opcional, se deja igual)
        logger.info("FÓRMULA CAPM UTILIZADA:")
        logger.info("  E(Ri) = Rf + βi × (E(Rm) − Rf)")
        logger.info(f"  Rf (trimestral): {rf*100:.2f}%")
        logger.info(f"  E(Rm) IBC    : {rm_pct:.2f}%")

        # ============================================================
        # 5) Ajustes realizados respecto al trimestre anterior (igual que antes)
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

        # ============================================================
        # 6) Exportar a Excel (sin cambios)
        df_comp = pd.DataFrame(tabla, columns=[
            'Ticker','Peso','Cantidad','Precio_Inicial','Inversion_Inicial',
            'Precio_Final','Retorno (%)','Beta'
        ])
        df_res = pd.DataFrame([{
            'Capital_Disponible_Bs': capital0,
            'Comision_%': comm_pct,
            'Comision_Bs': comm_bs,
            'Capital_Invertido_Bs': inv,
            'Remanente_Bs': remanente,
            'Valor_Final_Bs': final_v,
            'Ganancia_Bs': ganancia,
            'ROI_%': roi_pct,
            'Retorno_CAPM_%': retcapm,
            'Rendimiento_IBC_%': rm_pct,
            'Diferencia_ROI_IBC_%': roi_pct - rm_pct,
            'Volatilidad_diaria_%': vol_pct,
            'Overall_Deviation_%': overall_dev,
            'Sharpe_Ratio': shar,
            'Rf_trimestral_%': rf * 100,
            'E_Rm_IBC_%': rm_pct,
            'Formula_CAPM': 'Rf + β*(E(Rm)-Rf)'
        }])

        os.makedirs("resultados", exist_ok=True)
        fn = f"resultados/optimizacion_{results['trimestre']}.xlsx"
        with pd.ExcelWriter(fn, engine='openpyxl') as writer:
            df_comp.to_excel(writer, sheet_name="Composicion", index=False)
            df_res.to_excel(writer, sheet_name="Resumen", index=False)

        logger.info(f"Excel generado en → {fn}\n")


if __name__ == "__main__":
    optimizer = PortfolioOptimizer(data_folder='data_acciones')
    print("\nOptimizador Trimestral - Mercado Venezolano\n")  # <-- lo cambiamos a logger.info
    # Secuencia de trimestres a analizar
    quarters = [
        (2021, 'Q1'), (2021, 'Q2'), (2021, 'Q3'), (2021, 'Q4'),
        (2022, 'Q1'), (2022, 'Q2'), (2022, 'Q3'), (2022, 'Q4'),
        (2023, 'Q1'), (2023, 'Q2'), (2023, 'Q3'), (2023, 'Q4'),
        (2024, 'Q1'), (2024, 'Q2'), (2024, 'Q3'), (2024, 'Q4')
    ]

    prev_weights, prev_tickers = None, None
    all_results = []

    for year, quarter in quarters:
        # Aquí vamos a:
        # 1) Eliminar cualquier FileHandler antiguo
        # 2) Crear uno nuevo apuntando a "log_<year>-<quarter>.txt"
        # 3) Añadirlo a logger
        #
        # De este modo cada trimestre escribe en su propio archivo.
        #
        for handler in list(logger.handlers):
            # Si el handler es FileHandler, lo quitamos
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)

        log_filename = f"log_{year}-{quarter}.txt"
        file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Mensaje de encabezado (se verá en pantalla y en el archivo correspondiente)
        logger.info("\n" + "="*50)
        logger.info(f"PROCESANDO TRIMESTRE: {year}-{quarter}")
        logger.info("="*50)

        # Ahora reemplazamos los prints iniciales por logger.info
        try:
            results = optimizer.run_quarterly_analysis(
                year, quarter, prev_weights, prev_tickers
            )
            optimizer.print_results(results)
            all_results.append(results)

            # Actualizar para el próximo trimestre
            prev_weights = results.get('weights', None)
            prev_tickers = results.get('acciones', None)

        except Exception as e:
            logger.error(f"❌ Error procesando {year}-{quarter}: {str(e)}")
            # Creamos un resultado de error para mantener secuencia
            results = {
                'trimestre': f"{year}-{quarter}",
                'error': str(e)
            }
            optimizer.print_results(results)
            all_results.append(results)

            # Resetear si es un error crítico de datos
            if "Datos insuficientes" in str(e):
                prev_weights, prev_tickers = None, None

