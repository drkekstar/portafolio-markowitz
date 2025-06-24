## Optimizador de Portafolios Trimestrales – Mercado Venezolano

Este proyecto contiene un *optimizador de portafolios* basado en el modelo de Markowitz con retornos estimados vía **CAPM local (Venezuela)** y datos reales del índice IBC. Está diseñado para:

- **Seleccionar** automáticamente las 10 acciones con mayor capitalización del trimestre.
- **Cargar** precios históricos trimestrales desde archivos CSV individuales.
- **Optimizar** la combinación de activos minimizando la volatilidad y maximizando el Ratio de Sharpe.
- Permitir **cantidades fraccionarias** de acciones.
- **Respetar** un presupuesto inicial (Bs. 10 000) incluyendo comisiones.
- **Calcular** métricas clave:
  - ROI real a partir de precios finales.
  - Retorno esperado (CAPM local).
  - Volatilidad, Sharpe Ratio, beta de portafolio.
- **Exportar** un informe en Excel con hojas “Composición” y “Resumen”.

### Requisitos

- Python 3.8+  
- Paquetes:
  ```bash
  pip install numpy pandas scipy tabulate openpyxl



Estructura de datos

En la carpeta data_acciones/ debes colocar:

IBC_Tri_2020_2024.csv - Este archivo contiene los datos trimestrales del índice IBC.

Columnas: TIME (e.g. 2023Q2), RM Trimestral (e.g. 25.34%).

Mkt_Cap_Tri_<AÑO>.csv - Estos archivos contienen la capitalización de mercado trimestral de las acciones.

Columnas: Trimestre (e.g. 2023 Q2), Ticker, Mkt cap.

<TICKER>.csv (por cada acción) - Estos archivos contienen los precios históricos trimestrales de cada acción.

Columnas: time (YYYY-MM-DD), close.


Al correrlo:

Elige el trimestre (por defecto 2023-Q2 en el bloque __main__).

Muestra en consola:

Tabla de composición (peso, cantidad, precio, retorno, beta).

Resumen financiero (capital, comisiones, ROI, CAPM, Sharpe, etc.).

Crea un archivo Excel en resultados/optimizacion_<trimestre>.xlsx con:

Hoja “Composicion”: detalle de cada acción.

Hoja “Resumen”: todas las métricas clave y la fórmula CAPM utilizada.

