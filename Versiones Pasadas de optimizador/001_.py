import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from scipy.optimize import minimize
from tabulate import tabulate


class PortfolioOptimizer:
   def __init__(self, data_folder='data_acciones'):
       # Betas de las acciones
       self.betas = {
           'BPV': 1.15, 'MVZ.A': 0.86, 'ABC.A': 0.66, 'ENV': 0.82,
           'EFE': 0.21, 'CGQ': 0.96, 'CRM.A': 0.96, 'DOM': 0.62,
           'MPA': 0.76, 'CCR': 0.40, 'GZL': 0.61, 'FNC': 0.26,
           'BNC': -0.77, 'RST.B': 0.76
       }
       # Parámetros del mercado
       self.market_params = {
           'Rf_VEN': 0.1092,      # 10.92% trimestral
           'transaction_cost': 0.0275  # 2.75%
       }
       self.priority_tickers = list(self.betas.keys())
       self.data_folder = data_folder
       self.ibc_data = self.load_ibc_data()

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
   
   def load_market_caps(self, year, quarter):
       """Carga y ordena capitalizaciones de mercado del trimestre."""
       fn = f"Mkt_Cap_Tri_{year}.csv"
       path = os.path.join(self.data_folder, fn)
       df = pd.read_csv(path)
       for col in ['Trimestre', 'Mkt cap', 'Ticker']:
           if col not in df.columns:
               raise ValueError(f"{fn} debe contener columna '{col}'")
       qstr = f"{year} {quarter}"
       dfq = df[df['Trimestre'] == qstr]
       if dfq.empty:
           raise ValueError(f"No hay datos de capitalización para {qstr}")
       return dfq.sort_values('Mkt cap', ascending=False)
   
   def get_historical_prices(self, tickers, year, quarter):
    """Versión robusta a datos faltantes"""
    # 1. Determinar rango de fechas (año anterior)
    q_map = {'Q1': (1, 3), 'Q2': (4, 6), 'Q3': (7, 9), 'Q4': (10, 12)}
    m1, m2 = q_map[quarter]
    end_date = (datetime(year, m2, 1) + timedelta(days=31)).replace(day=1) - timedelta(days=1)
    start_date = end_date.replace(year=end_date.year - 1)
     # 2. Crear índice de fechas comunes
    all_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Días hábiles
    # 3. Cargar y alinear datos
    df_total = pd.DataFrame(index=all_dates)
    for t in tickers:
        try:
            df = pd.read_csv(os.path.join(self.data_folder, f"{t}.csv"), 
                            parse_dates=['time'], 
                            index_col='time')
            # Rellenar huecos: forward fill + backward fill
            df_reindexed = df.reindex(all_dates).ffill().bfill()
            df_total[t] = df_reindexed['close']
        except Exception as e:
            print(f"⚠️ Error cargando {t}: {str(e)}")
            df_total[t] = np.nan  # Marcar como faltante
     # 4. Validación final
    if df_total.isna().all().any():
        raise ValueError("Al menos una acción no tiene datos históricos")
    
    # 5. Eliminar filas con muchos faltantes (opcional)
    df_clean = df_total.dropna(thresh=len(tickers)//2)  # Ej: conservar si hay al menos 50% de datos
    
    return df_clean, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
   
   def calculate_expected_returns(self, tickers, year, quarter):
       """Calcula E(Ri) por CAPM local."""
       Rf = self.market_params['Rf_VEN']
       Rm = self.get_quarterly_rm(year, quarter)
       return np.array([Rf + self.betas[t] * (Rm - Rf) for t in tickers])
   def optimize_portfolio(self, tickers, prices, year, quarter, capital=10000):
        """Versión mejorada con validaciones adicionales"""
         # 1. Verificar datos completos
        if prices.isna().any().any():
            raise ValueError("Hay precios faltantes en el DataFrame")
        
        returns = prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("No hay suficientes datos para calcular retornos")
        
        cov = returns.cov().values
        exp_ret = self.calculate_expected_returns(tickers, year, quarter)
        init_prices = prices.iloc[0].values
        if not all(p > 0 for p in init_prices):
            raise ValueError("Precios deben ser positivos")
        def objective(w):
            volatility = np.sqrt(w @ cov @ w)
            quantities = (w * capital_ajustado) / init_prices
            total_inv = np.sum(quantities * init_prices)
            penalty = max(0, (total_inv * (1 + tc)) - capital) * 1000
            return volatility + penalty
        
        # 1. Calcular capital ajustado para incluir comisiones
        tc = self.market_params['transaction_cost']
        capital_ajustado = capital / (1 + tc)
        bounds = [(0.005, 0.30) for _ in tickers]
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # 2. Optimización con restricciones de pesos
        bnds = [(0.005, 0.30)] * len(tickers)
        cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
        def obj(w): return np.sqrt(w @ cov @ w)
        w0 = np.ones(len(tickers)) / len(tickers)
        res = minimize(obj, w0, method='SLSQP', bounds=bnds, constraints=cons)
        if not res.success:
            raise ValueError("Optimización fallida")
        w = res.x
        result = minimize(
            objective,
            x0=np.ones(len(tickers))/len(tickers),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            raise ValueError(f"Optimización fallida: {result.message}")
        
        # 3. Calcular cantidades con capital ajustado
        qty = (w * capital_ajustado) / init_prices
        
        # 4. Ajuste de precisión: reducir ligeramente las cantidades
        # para garantizar que no se exceda el capital
        reduction_factor = 0.9999  # Reducción del 0.01%
        qty = qty * reduction_factor
        
        # 5. Calcular montos con precisión decimal
        from decimal import Decimal, getcontext
        getcontext().prec = 10
        getcontext().rounding = 'ROUND_HALF_UP'

        w_opt = result.x
        quantities = [
            (Decimal(str(w)) * Decimal(str(capital_ajustado)) / Decimal(str(p))
             ).quantize(Decimal('0.0001')) for w, p in zip(w_opt, init_prices)
             ]
        quantities = [q.quantize(Decimal('0.0001')) for q in quantities]

        
        # Convertir a Decimal para cálculos precisos
        qty = [Decimal(str(q)).quantize(Decimal('0.0001')) for q in qty]
        init_prices = [Decimal(str(p)).quantize(Decimal('0.01')) for p in init_prices]
        
        # Calcular montos exactos
        invested = sum(q * p for q, p in zip(qty, init_prices))
        commission = invested * Decimal(str(tc))
        total_gastado = invested + commission
        
        # 6. Ajuste final de comisión para garantizar <= capital
        if total_gastado > Decimal(str(capital)):
            # Reducir comisión para cumplir con el capital
            commission = Decimal(str(capital)) - invested
            total_gastado = invested + commission
        
        # 7. Convertir de vuelta a float para el resto de cálculos
        qty = [float(q) for q in qty]
        init_prices = [float(p) for p in init_prices]
        invested = float(invested)
        commission = float(commission)
        total_gastado = float(total_gastado)
        
        # 8. Calcular métricas de desempeño
        final_prices = prices.iloc[-1].values
        final_val = sum(q * p for q, p in zip(qty, final_prices))
        roi = (final_val - total_gastado) / capital
        
        #9. Validacion final estricta
        if abs (float(total_gastado) - capital) > 0.01:
            raise ValueError(f"Difererencia de capital significativa: {float(total_gastado):.2f} vs {capital}")

        return {
            'weights': w,
            'quantities': qty,
            'initial_prices': init_prices,
            'final_prices': final_prices,
            'expected_return': w @ exp_ret,
            'risk': res.fun,
            'sharpe': (w @ exp_ret - self.market_params['Rf_VEN']) / res.fun,
            'invested': invested,
            'commission': commission,
            'final_value': final_val,
            'roi': roi,
            'total_gastado': total_gastado
        }


   def run_quarterly_analysis(self, year, quarter):
       caps     = self.load_market_caps(year, quarter)
       top10    = caps[caps['Ticker'].isin(self.priority_tickers)].head(10)['Ticker'].tolist()
       prices, sd, ed = self.get_historical_prices(top10, year, quarter)
       opt      = self.optimize_portfolio(top10, prices, year, quarter)
       beta_p   = sum(opt['weights'][i] * self.betas[t] for i, t in enumerate(top10))
       return {
           'trimestre':        f"{year}-{quarter}",
           'acciones':         top10,
           'beta':             beta_p,
           'retorno_esperado': opt['expected_return'],
           'riesgo':           opt['risk'],
           'sharpe':           opt['sharpe'],
           'roi':              opt['roi'],
           'optimizacion':     opt,
           'fecha_inicio':     sd,
           'fecha_fin':        ed,
           'rm_trimestral':    self.get_quarterly_rm(year, quarter)
       }


   def print_results(self, results):
       """Imprime en consola y exporta el reporte a Excel."""
       if not results:
           return
       opt       = results['optimizacion']
       capital0  = 10000
       rf        = self.market_params['Rf_VEN']
       rm        = results['rm_trimestral']
       # Comisión en % sobre capital inicial
       comm_pct  = opt['commission'] * 100 / capital0


       # Composición
       tabla = []
       for i, t in enumerate(results['acciones']):
           tabla.append([
               t,
               f"{opt['weights'][i]*100:.2f}%",
               f"{opt['quantities'][i]:.4f}",
               f"{opt['initial_prices'][i]:.4f}",
               f"{(opt['quantities'][i]*opt['initial_prices'][i]):,.4f}",
               f"{opt['final_prices'][i]:.4f}",
               f"{(opt['final_prices'][i]/opt['initial_prices'][i]-1)*100:.4f}%",
               f"{self.betas[t]:.2f}"
           ])

       # Resumen CORREGIDO:
       inv = opt['invested']
       comm_bs   = opt['commission']
       remanente = capital0 - inv - comm_bs
       final_v   = opt['final_value']
       ganancia  = final_v - inv - comm_bs
       roi_pct   = opt['roi'] * 100  
       retcapm   = results['retorno_esperado'] * 100  
       vol_pct   = opt['risk'] * 100  
       shar      = opt['sharpe']

       # Impresión consola
       print(f"\n{'='*80}")
       print(f"RESULTADOS {results['trimestre']}  {results['fecha_inicio']} → {results['fecha_fin']}")
       print(f"{'='*80}")
       print("COMPOSICIÓN ÓPTIMA:")
       print(tabulate(tabla,
           headers=['Ticker','Peso','Cant','Precio Ini','Inv Ini','Precio Fin','Ret%','Beta'],
           tablefmt='grid'))
       print("\nRESUMEN DEL PORTAFOLIO:")
       print(f"{'Capital disponible (Bs)':<40}: {capital0:,.2f}")
       print(f"{'Comisión aplicada (%)':<40}: {comm_pct:.2f}%")
       print(f"{'Comisión aplicada (Bs)':<40}: {comm_bs:,.2f}")
       print(f"{'Capital invertido (Bs)':<40}: {inv:,.2f}")
       print(f"{'Dinero no invertido (remanente Bs)':<40}: {remanente:,.2f}")
       print(f"{'Valor Final del Portafolio (Bs)':<40}: {final_v:,.2f}")
       print(f"{'Ganancia neta del periodo (Bs)':<40}: {ganancia:,.2f}")
       print(f"{'Rendimiento total (ROI)':<40}: {roi_pct:.4f}%")
       print(f"{'Retorno Esperado (CAPM)':<40}: {retcapm:.4f}%")
       print(f"{'Volatilidad del Portafolio':<40}: {vol_pct:.4f}%")
       print(f"{'Sharpe Ratio':<40}: {shar:.2f}")
       print("\nFÓRMULA CAPM UTILIZADA:")
       print("  E(Ri) = Rf + βi × (E(Rm) − Rf)")
       print(f"  Rf (trimestral): {rf*100:.2f}%")
       print(f"  E(Rm) IBC    : {rm*100:.2f}%")
       print(f"{'='*80}\n")


       # Exportar a Excel
       df_comp = pd.DataFrame(tabla, columns=[
           'Ticker','Peso','Cantidad','Precio_Inicial','Inversion_Inicial',
           'Precio_Final','Retorno (%)','Beta'
       ])
       df_res  = pd.DataFrame([{
           'Capital_Disponible_Bs':   capital0,
           'Comision_%':              comm_pct,
           'Comision_Bs':             comm_bs,
           'Capital_Invertido_Bs':    inv,
           'Remanente_Bs':            remanente,
           'Valor_Final_Bs':          final_v,
           'Ganancia_Bs':             ganancia,
           'ROI_%':                   roi_pct,
           'Retorno_CAPM_%':          retcapm,
           'Volatilidad_%':           vol_pct,
           'Sharpe_Ratio':            shar,
           'Rf_trimestral_%':         rf*100,
           'E_Rm_IBC_%':              rm*100,
           'Formula_CAPM':            'Rf + β*(E(Rm)-Rf)'
       }])
       os.makedirs("resultados", exist_ok=True)
       fn = f"resultados/optimizacion_{results['trimestre']}.xlsx"
       with pd.ExcelWriter(fn, engine='openpyxl') as writer:
           df_comp.to_excel(writer, sheet_name="Composicion", index=False)
           df_res.to_excel(writer, sheet_name="Resumen",   index=False)
       print(f"Excel generado en → {fn}\n")

if __name__ == "__main__":
   optimizer = PortfolioOptimizer(data_folder='data_acciones')
   print("\nOptimizador Trimestral - Mercado Venezolano\n")
   results = optimizer.run_quarterly_analysis(2024, 'Q1')
   optimizer.print_results(results)