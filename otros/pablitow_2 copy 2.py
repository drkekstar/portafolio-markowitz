import pandas as pd
import matplotlib.pyplot as plt

# 1) Definir la prima de riesgo país anual (en decimal)
crp_anual = {
    2021: 0.1918,
    2022: 0.2034,
    2023: 0.2469,
    2024: 0.2358
}

# 2) Función para convertir CRP anual a CRP trimestral
def crp_trim_from_anual(crp_anual_val):
    return (1 + crp_anual_val) ** (1/4) - 1

# 3) Cargar el CSV
df = pd.read_csv('resumen_trimestres.csv')

# 4) Extraer año de la columna 'inicio' (que tiene formato "YYYY-Qx")
df['Año'] = df['inicio'].str.split('-').str[0].astype(int)

# 5) Calcular CRP trimestral para cada fila
df['CRP_trim'] = df['Año'].map(lambda y: crp_trim_from_anual(crp_anual[y]))

# 6) Calcular Rm_net = Rm(IBC) (%) − (CRP_trim × 100)
#    (ambos en % para que estén en la misma escala)
df['Rm_net'] = df['Rm(ibc)'] - df['CRP_trim'] * 100

# 7) Extraer los vectores que utilizaremos:
#    X = Desviación anualizada (%) del portafolio
#    Y = Rm_net (%), ya neto de CRP trimestral
#    labels = etiqueta de trimestre ("2021-Q1", etc.)
x = df['Desviacion']
y = df['Rm_net']
labels = df['inicio']

# 8) Calcular Sharpe = (Rm_net) / (Desviacion) para cada trimestre
#    (recordar que Rm_net y Desviacion ya están en %, por eso ambas magnitudes son comparables)
df['Sharpe'] = df['Rm_net'] / df['Desviacion']

# 9) Dibujar el scatter principal
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='tab:green', s=60, alpha=0.7, label='Trimestres')

# 10) Dibujar, para cada punto, la recta desde el origen con pendiente = Sharpe
#     (esto hará que la línea que une (0,0)→(x[i],y[i]) tenga pendiente exacta Sharpe[i]).
#     Con un poco de transparencia (alpha), se verá el “ángulo” de cada Sharpe.
for i in range(len(df)):
    xi = x.iloc[i]
    yi = y.iloc[i]
    si = df['Sharpe'].iloc[i]
    # Generar dos puntos: (0,0) → (xi, yi). Los unimos con una línea fina
    plt.plot([0, xi], [0, yi],
             color='tab:gray',
             linestyle='--',
             linewidth=0.7,
             alpha=0.5)

# 11) Anotar cada punto con su trimestre y su valor de Sharpe (opcional)
for i, label in enumerate(labels):
    xi = x.iloc[i]
    yi = y.iloc[i]
    si = df['Sharpe'].iloc[i]
    plt.annotate(
        f"{label}\nS={si:.2f}",
        (xi, yi),
        textcoords="offset points",
        xytext=(5, 5),
        ha='left',
        fontsize=8
    )

# 12) Ajustar ejes, etiquetas y título
plt.axhline(0, color='black', linewidth=0.5, linestyle=':')
plt.axvline(0, color='black', linewidth=0.5, linestyle=':')
plt.xlabel('Desviación Estándar Anualizada del Portafolio [%]')
plt.ylabel('Rm(IBC) neto de CRP trimestral [%]')
plt.title('Scatter: (Rm(IBC) − CRP_trim) vs. Desviación Estándar\ncada línea punteada = Sharpe para ese trimestre')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
# 13) Guardar la figura como imagen
plt.savefig('scatter_rm_net_vs_desviacion.png', dpi=300, bbox_inches='tight')
