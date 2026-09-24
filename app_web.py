import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

# Configuración de página
st.set_page_config(page_title="Optimización de Canales - Carhuaz", layout="wide")

st.title("🌊 Optimización Hidráulica y Pérdidas por Infiltración")
st.markdown("**Proyecto:** Canal Trapezoidal no Revestido en Carhuaz, Áncash")

# Función auxiliar para formatear números eliminando ceros a la derecha sin perder precisión
def fmt(val):
    return f"{val:.5f}".rstrip('0').rstrip('.')

# ---------------------------------------------------------
# BARRA LATERAL (DATOS DE ENTRADA INTERACTIVOS EN L/s)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Canal")

# Entradas sin forzar ceros sobrantes mediante format="%g"
Q_ls = st.sidebar.number_input("Caudal de diseño Q (L/s)", value=1200.0, step=10.0, format="%g")
# Conversión explícita a m³/s para la hidráulica de Manning
Q_diseno = Q_ls / 1000.0 

n_manning = st.sidebar.number_input("Rugosidad de Manning n", value=0.025, step=0.001, format="%g")
S_pendiente = st.sidebar.number_input("Pendiente longitudinal S (m/m)", value=0.001, step=0.0005, format="%g")
z_talud = st.sidebar.number_input("Talud lateral z (H:1V)", value=1.5, step=0.1, format="%g")
C_ingham = st.sidebar.number_input("Coeficiente de Ingham C", value=1.0, step=0.1, format="%g")

# ---------------------------------------------------------
# CÁLCULOS HIDRÁULICOS
# ---------------------------------------------------------
def area_hidraulica(b, y, z=z_talud):
    return (b + z * y) * y

def perimetro_mojado(b, y, z=z_talud):
    return b + 2 * y * np.sqrt(1 + z**2)

def radio_hidraulico(b, y, z=z_talud):
    return area_hidraulica(b, y, z) / perimetro_mojado(b, y, z)

def caudal_manning(b, y, n=n_manning, S=S_pendiente, z=z_talud):
    A = area_hidraulica(b, y, z)
    R = radio_hidraulico(b, y, z)
    return (1.0 / n) * A * (R**(2.0/3.0)) * np.sqrt(S)

def tasa_infiltracion_ingham(b, y, C=C_ingham, z=z_talud):
    P = perimetro_mojado(b, y, z)
    return 0.0025 * C * P * np.sqrt(y) * 1000

# MEH (Máxima Eficiencia Hidráulica)
factor_meh = 2 * (np.sqrt(1 + z_talud**2) - z_talud)
res_meh = minimize_scalar(lambda y: (caudal_manning(factor_meh*y, y) - Q_diseno)**2, bounds=(0.01, 5.0), method='bounded')
y_meh = res_meh.x
b_meh = factor_meh * y_meh
bl_meh = 0.35
H_meh = y_meh + bl_meh

# SMI (Sección de Mínima Infiltración)
factor_smi = 4 * (np.sqrt(1 + z_talud**2) - z_talud)
res_smi = minimize_scalar(lambda y: (caudal_manning(factor_smi*y, y) - Q_diseno)**2, bounds=(0.01, 5.0), method='bounded')
y_smi = res_smi.x
b_smi = factor_smi * y_smi
bl_smi = 0.35
H_smi = y_smi + bl_smi

# Construcción de la curva del Frente de Pareto
b_min = max(0.1, min(b_meh, b_smi) * 0.4)
b_max = max(b_meh, b_smi) * 2.5
b_rango = np.linspace(b_min, b_max, 100)
pareto_A, pareto_I = [], []

for b_i in b_rango:
    res = minimize_scalar(lambda y: (caudal_manning(b_i, y) - Q_diseno)**2, bounds=(0.01, 5.0), method='bounded')
    y_i = res.x
    pareto_A.append(area_hidraulica(b_i, y_i))
    pareto_I.append(tasa_infiltracion_ingham(b_i, y_i))

# ---------------------------------------------------------
# TABLA DE RESULTADOS FORMATO DINÁMICO
# ---------------------------------------------------------
st.subheader(f"📊 Tabla Comparativa de Resultados (Q = {fmt(Q_ls)} L/s = {fmt(Q_diseno)} m³/s)")

table_data = {
    "Parámetro / Variable": [
        "Ancho de Solera b (m)", "Tirante de Agua y (m)", "Área Hidráulica A (m²)",
        "Perímetro Mojado P (m)", "Velocidad del Flujo V (m/s)", "Borde Libre BL (m)",
        "Altura Total H (m)", "Infiltración (L/s/km)"
    ],
    "Criterio MEH": [
        fmt(b_meh), fmt(y_meh), fmt(area_hidraulica(b_meh, y_meh)),
        fmt(perimetro_mojado(b_meh, y_meh)), fmt(Q_diseno/area_hidraulica(b_meh, y_meh)),
        fmt(bl_meh), fmt(H_meh), fmt(tasa_infiltracion_ingham(b_meh, y_meh))
    ],
    "Criterio SMI": [
        fmt(b_smi), fmt(y_smi), fmt(area_hidraulica(b_smi, y_smi)),
        fmt(perimetro_mojado(b_smi, y_smi)), fmt(Q_diseno/area_hidraulica(b_smi, y_smi)),
        fmt(bl_smi), fmt(H_smi), fmt(tasa_infiltracion_ingham(b_smi, y_smi))
    ]
}

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# GRÁFICOS
# ---------------------------------------------------------
st.subheader("📈 Análisis Gráfico")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

# Frente de Pareto
ax1.plot(pareto_A, pareto_I, color='#1f77b4', linewidth=2, label='Frente de Pareto')
ax1.scatter(area_hidraulica(b_meh, y_meh), tasa_infiltracion_ingham(b_meh, y_meh), color='red', s=60, zorder=5, label='MEH')
ax1.scatter(area_hidraulica(b_smi, y_smi), tasa_infiltracion_ingham(b_smi, y_smi), color='green', s=60, zorder=5, label='SMI')
ax1.set_title('Curva Frente de Pareto', fontweight='bold')
ax1.set_xlabel('Área Hidráulica A (m²)')
ax1.set_ylabel('Infiltración Estimada (L/s/km)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Sección Transversal
x_meh = [-b_meh/2 - z_talud*y_meh, -b_meh/2, b_meh/2, b_meh/2 + z_talud*y_meh]
y_geom_meh = [y_meh, 0, 0, y_meh]
x_smi = [-b_smi/2 - z_talud*y_smi, -b_smi/2, b_smi/2, b_smi/2 + z_talud*y_smi]
y_geom_smi = [y_smi, 0, 0, y_smi]

ax2.plot(x_meh, y_geom_meh, color='red', linestyle='--', linewidth=2, label=f'MEH (b={fmt(b_meh)}m)')
ax2.plot(x_smi, y_geom_smi, color='green', linestyle='-', linewidth=2, label=f'SMI (b={fmt(b_smi)}m)')
ax2.set_title('Geometría Transversal Trapezoidal', fontweight='bold')
ax2.set_xlabel('Ancho de Sección (m)')
ax2.set_ylabel('Tirante y (m)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)
