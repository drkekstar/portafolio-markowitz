import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Cargar el CSV
df = pd.read_csv('resumen_trimestres.csv')

# 2) Seleccionar solo las columnas que realmente existen en tu archivo
#    (según df.columns: 'inicio', 'Fecha', 'CAPM', 'Beta',
#     'Sharpe', 'Desviacion', 'Portafolio Optimizado', 'Indice IBC', 'Alpha')
cols = [
    'Portafolio Optimizado',  # ROI real del portafolio (%)
    'Indice IBC',             # Retorno (%) del IBC
    'Alpha',                  # ROI minus IBC (%)
    'CAPM',                   # CAPM ex‐ante (%)
    'Beta',                   # Beta de carta (%)
    'Desviacion',             # Desviación anualizada del portafolio (%)
    'Sharpe'                  # Sharpe ratio (unit-less)
]

# 3) Calcular matriz de correlación
corr_matrix = df[cols].corr()

# 4) Graficar heatmap con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    cbar_kws={'shrink': 0.8}
)
plt.title('Correlación entre métricas clave')
plt.tight_layout()
plt.show()
# 5) Guardar la figura como imagen
plt.savefig('correlacion_metrica_clave.png', dpi=300, bbox_inches='tight')
