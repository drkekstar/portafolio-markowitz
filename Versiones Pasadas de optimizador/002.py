import numpy as np
import pandas as pd
import os
import logging
import sys
from datetime import datetime, timedelta
from scipy.optimize import minimize
from tabulate import tabulate
from decimal import Decimal, getcontext
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
            'Rf_VEN': 0.1092,      # 10.92% trimestral
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

    def get_historical_prices(self, tickers, year, quarter):
        """Carga precios de cierre del año anterior al trimestre dado."""
        # Determinar fechas como objetos datetime
        end_dt = self.get_quarter_end_date(year, quarter)
        start_dt = end_dt - timedelta(days=365)

        # Conservar objetos datetime para el procesamiento
        all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')
        
        df_total = pd.DataFrame(index=all_dates)
        
        # Convertir a formato string y timestamp
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)

        # Crear índice de fechas comunes
        all_dates = pd.date_range(start=start_ts, end=end_ts, freq='B')  # Días hábiles
        df_total = pd.DataFrame(index=all_dates)
        
        # Diccionarios para almacenar betas y mkt caps
        self.current_betas = {}
        self.current_mkt_caps = {}

        for t in tickers:
            path = os.path.join(self.data_folder, f"{t}.csv")
            try:
                df = pd.read_csv(path, parse_dates=['time'])
                df = df.set_index('time').sort_index()
                # Verificar explícitamente el orden
                if not df.index.is_monotonic_increasing:
                    df = df.sort_index() 
                
                # Obtener el último registro válido
                last_row = df.loc[df.index <= end_dt].iloc[-1]
                self.current_betas[t] = last_row['Beta']
                self.current_mkt_caps[t] = last_row['Mkt cap']
                
                # Filtrar y reindexar
                dfq = df.reindex(all_dates, method='ffill')
                df_total[t] = dfq['close']
                
                if dfq.empty:
                    print(f"⚠️ Advertencia: no hay datos para {t} en {start_date} a {end_date}")
                    continue
                
                # Reindexar para asegurar continuidad
                dfq = dfq.reindex(all_dates, method='ffill')
                df_total[t] = dfq['close']
                
            except Exception as e:
                print(f"⚠️ Error procesando {t}: {str(e)}")
        
        if df_total.empty:
            raise ValueError("No se pudieron cargar datos históricos para ningún ticker.")
        
            # Flexibilizar limpieza de datos
        threshold = len(tickers) * 0.3  # Permitir hasta 70% de datos faltantes por fila
        df_clean = df_total.dropna(thresh=len(tickers)//2)
        
        # Verificar si tenemos suficientes datos
        if len(df_clean) < 50:  # Mínimo absoluto de 50 días de datos
            raise ValueError(f"Datos insuficientes. Solo {len(df_clean)} días completos")
        
            # Convertir a strings solo al final
        return df_clean, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

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
            init_prices = prices.iloc[0].values
            Rf = self.market_params['Rf_VEN']
            tc = self.market_params['transaction_cost']

            # 5. Configuración de pesos iniciales con validación
            w0 = np.ones(len(tickers)) / len(tickers)  # Default equal weights
            if initial_weights is not None:
                if len(initial_weights) == len(tickers):
                    w0 = initial_weights
                else:
                    print(f"⚠️ initial_weights ignorado: dimensión {len(initial_weights)} != {len(tickers)}")

            # 6. Función objetivo con protección contra divisiones por cero
            def neg_sharpe(w):
                port_return = np.dot(w, exp_ret)
                port_vol = np.sqrt(w @ cov @ w)
                return -(port_return - Rf) / (port_vol + 1e-10)  # Evita división por cero

            # 7. Configuración de restricciones
            bnds = [(0.005, 0.30) for _ in tickers]
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            # 8. Optimización con manejo de fallos
            res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bnds, constraints=cons,
                        options={'ftol': 1e-4, 'maxiter': 1000, 'disp': False})

            if not res.success:
                print(f"⚠️ Optimización subóptima: {res.message}. Usando pesos iniciales como fallback.")
                w = w0
            else:
                w = res.x

            # 9. Cálculo preciso de cantidades con Decimal
            getcontext().prec = 10
            getcontext().rounding = 'ROUND_HALF_UP'
            
            capital_ajustado = Decimal(str(capital)) / (Decimal('1') + Decimal(str(tc)))
            qty = []
            for i in range(len(tickers)):
                try:
                    q = (Decimal(str(w[i])) * capital_ajustado / 
                        Decimal(str(init_prices[i]))).quantize(Decimal('0.0001'))
                    qty.append(q)
                except:
                    print(f"⚠️ Error cálculo cantidad para {tickers[i]}. Usando 0.")
                    qty.append(Decimal('0'))

            # 10. Cálculo de montos invertidos
            init_prices_dec = [Decimal(str(p)).quantize(Decimal('0.01')) for p in init_prices]
            invested = sum(q * p for q, p in zip(qty, init_prices_dec))
            commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
            total_gastado = invested + commission

            # 11. Ajuste final de capital si es necesario
            if total_gastado > Decimal(str(capital)):
                adjustment = Decimal(str(capital)) / total_gastado
                qty = [q * adjustment for q in qty]
                qty = [q.quantize(Decimal('0.0001')) for q in qty]
                invested = sum(q * p for q, p in zip(qty, init_prices_dec))
                commission = (invested * Decimal(str(tc))).quantize(Decimal('0.01'))
                total_gastado = invested + commission

            # 12. Validación final
            if float(total_gastado) > capital + 0.01:
                print(f"⚠️ Ajuste de capital insuficiente: {float(total_gastado):.2f} > {capital}")

            # 13. Cálculo de métricas de desempeño
            final_prices = prices.iloc[-1].values
            final_val = sum(float(q) * p for q, p in zip(qty, final_prices))
            roi = (final_val - float(total_gastado)) / capital
            port_returns = returns.dot(w)
            overall_deviation = port_returns.std() * np.sqrt(252)  # Anualizada

            return {
                'weights': w,
                'quantities': [float(q) for q in qty],
                'initial_prices': [float(p) for p in init_prices],
                'final_prices': final_prices,
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
                'beta': beta_p,
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
        """Imprime en consola y guarda el reporte en el log del trimestre."""
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
        rm = results['rm_trimestral']
        comm_pct = opt['commission'] * 100 / capital0

        # Composición del portafolio
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

        # Resumen numérico
        inv = opt['invested']
        comm_bs = opt['commission']
        remanente = capital0 - inv - comm_bs
        final_v = opt['final_value']
        ganancia = final_v - inv - comm_bs
        roi_pct = opt['roi'] * 100
        retcapm = results['retorno_esperado'] * 100
        vol_pct = opt['risk'] * 100
        overall_dev = opt['overall_deviation'] * 100
        shar = opt['sharpe']

        # --------------------------------------------------------
        # Ahora usamos logger en lugar de print:
        logger.info("\n" + "="*80)
        logger.info(f"RESULTADOS {results['trimestre']}  {results['fecha_inicio']} → {results['fecha_fin']}")
        logger.info("="*80)
        logger.info("COMPOSICIÓN ÓPTIMA:")
        logger.info(tabulate(tabla,
                            headers=['Ticker','Peso','Cant','Precio Ini','Inv Ini','Precio Fin','Ret%','Beta'],
                            tablefmt='grid'))

        logger.info("\nRESUMEN DEL PORTAFOLIO:")
        logger.info(f"{'Capital disponible (Bs)':<40}: {capital0:,.2f}")
        logger.info(f"{'Comisión aplicada (%)':<40}: {comm_pct:.2f}%")
        logger.info(f"{'Comisión aplicada (Bs)':<40}: {comm_bs:,.2f}")
        logger.info(f"{'Capital invertido (Bs)':<40}: {inv:,.2f}")
        logger.info(f"{'Dinero no invertido (remanente Bs)':<40}: {remanente:,.2f}")
        logger.info(f"{'Valor Final del Portafolio (Bs)':<40}: {final_v:,.2f}")
        logger.info(f"{'Ganancia neta del periodo (Bs)':<40}: {ganancia:,.2f}")
        logger.info(f"{'Rendimiento total (ROI)':<40}: {roi_pct:.4f}%")
        logger.info(f"{'Retorno Esperado (CAPM)':<40}: {retcapm:.4f}%")
        logger.info(f"{'Volatilidad diaria del Portafolio':<40}: {vol_pct:.4f}%")
        logger.info(f"{'Overall Deviation (Desv. Anualizada)':<40}: {overall_dev:.4f}%")
        logger.info(f"{'Sharpe Ratio':<40}: {shar:.4f}")

        # Fórmula CAPM
        logger.info("\nFÓRMULA CAPM UTILIZADA:")
        logger.info("  E(Ri) = Rf + βi × (E(Rm) − Rf)")
        logger.info(f"  Rf (trimestral): {rf*100:.2f}%")
        logger.info(f"  E(Rm) IBC    : {rm*100:.2f}%")

        # Ajustes recomendados (si existen)
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

        # Exportar a Excel (sin cambios)
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
            'Volatilidad_diaria_%': vol_pct,
            'Overall_Deviation_%': overall_dev,
            'Sharpe_Ratio': shar,
            'Rf_trimestral_%': rf * 100,
            'E_Rm_IBC_%': rm * 100,
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
        (2024, 'Q1')
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
