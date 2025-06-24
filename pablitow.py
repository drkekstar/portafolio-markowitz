import pandas as pd
import matplotlib.pyplot as plt

# 1) Cargar el CSV que contiene: "inicio" (trimestre), 
#    "Portafolio Optimizado" (ROI), "Indice IBC" (retorno IBC) y "Alpha"
df = pd.read_csv('resumen_trimestres.csv')

# 2) Definir las series a graficar
periodos = df['inicio']                    # p.ej. ["2021-Q1", "2021-Q2", ...]
roi = df['Portafolio Optimizado']          # ROI (%) del portafolio
ibc = df['Indice IBC']                     # Retorno (%) del IBC
alpha = df['Alpha']                        # Alpha (%) = ROI - Retorno IBC

# 3) Configurar los valores de X como posiciones numéricas (0,1,2,...)
x = range(len(periodos))

# 4) Crear la figura y dibujar cada curva
plt.figure(figsize=(12, 6))

plt.plot(x, roi,   marker='o', linestyle='-', label='ROI Portafolio')
plt.plot(x, ibc,   marker='s', linestyle='--', label='Retorno IBC')
plt.plot(x, alpha, marker='^', linestyle='-.', label='Alpha')

# 5) Ajustar ejes, etiquetas y leyenda
plt.xticks(x, periodos, rotation=45, ha='right')
plt.xlabel('Trimestre')
plt.ylabel('Porcentaje (%)')
plt.title('Comparación de ROI, Retorno IBC y Alpha por Período')
plt.grid(alpha=0.3)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
# 6) Guardar la figura como imagen 
plt.savefig('comparacion_roi_ibc_alpha.png', dpi=300, bbox_inches='tight')  