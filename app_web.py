import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar

# Configuración de página
st.set_page_config(page_title="Optimización de Canales - Carhuaz", layout="wide")

st.title("🌊 Optimización Hidráulica y Pérdidas por Infiltración")
st.markdown("**Proyecto:** Canal Trapezoidal no Revestido en Carhuaz, Áncash")

# ---------------------------------------------------------
# BARRA LATERAL (DATOS DE ENTRADA INTERACTIVOS - 5 DECIMALES Y L/s)
# ---------------------------------------------------------
st.sidebar.header("⚙️ Parámetros del Canal")

# Ingreso en L/s con 5 decimales
Q_ls = st.sidebar.number_input("Caudal de diseño Q (L/s)", value=1200.00000, step=10.00000, format="%.5f")
# Conversión automática a m³/s para el cálculo interno de Manning
Q_diseno = Q_ls / 1000.0 

n_manning = st.sidebar.number_input("Rugosidad de Manning n", value=0.02500, step=0.00100, format="%.5f")
S_pendiente = st.sidebar.number_input("Pendiente longitudinal S (m/m)", value=0.00100, step=0.00050, format="%.5f")
z_talud = st.sidebar.number_input("Talud lateral z (H:1V)", value=1.50000, step=0.10000, format="%.5f")
C_ingham = st.sidebar.number_input("Coeficiente de Ingham C", value=1.00000, step=0.10000, format="%.5f")

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

# MEH
factor_meh = 2 * (np.sqrt(1 + z_talud**2) - z_talud)
res_meh = minimize_scalar(lambda y: (caudal_manning(factor_meh*y, y) - Q_diseno)**2, bounds=(0.01, 5.0), method='bounded')
y_meh = res_meh.x
b_meh = factor_meh * y_meh
bl_meh = 0.35
H_meh = y_meh + bl_meh

# SMI
factor_smi = 4 * (np.sqrt(1 + z_talud**2) - z_talud)
res_smi = minimize_scalar(lambda y: (caudal_manning(factor_smi*y, y) - Q_diseno)**2, bounds=(0.01, 5.0), method='bounded')
y_smi = res_smi.x
b_smi = factor_smi * y_smi
bl_smi = 0.35
H_smi = y_smi + bl_smi

# Pareto
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
# MOSTRAR RESULTADOS EN LA WEB
# ---------------------------------------------------------
st.subheader(f"📊 Tabla Comparativa de Resultados (Q = {Q_ls:.2f} L/s = {Q_diseno:.3f} m³/s)")

table_data = {
    "Parámetro / Variable": [
        "Ancho de Solera b (m)", "Tirante de Agua y (m)", "Área Hidráulica A (m²)",
        "Perímetro Mojado P (m)", "Velocidad del Flujo V (m/s)", "Borde Libre BL (m)",
        "Altura Total H (m)", "Infiltración (L/s/km)"
    ],
    "Criterio MEH": [
        f"{b_meh:.5f}", f"{y_meh:.5f}", f"{area_hidraulica(b_meh, y_meh):.5f}",
        f"{perimetro_mojado(b_meh, y_meh):.5f}", f"{Q_diseno/area_hidraulica(b_meh, y_meh):.5f}",
        f"{bl_meh:.5f}", f"{H_meh:.5f}", f"{tasa_infiltracion_ingham(b_meh, y_meh):.5f}"
    ],
    "Criterio SMI": [
        f"{b_smi:.5f}", f"{y_smi:.5f}", f"{area_hidraulica(b_smi, y_smi):.5f}",
        f"{perimetro_mojado(b_smi, y_smi):.5f}", f"{Q_diseno/area_hidraulica(b_smi, y_smi):.5f}",
        f"{bl_smi:.5f}", f"{H_smi:.5f}", f"{tasa_infiltracion_ingham(b_smi, y_smi):.5f}"
    ]
}

df = pd.DataFrame(table_data)
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# GRÁFICOS INTERACTIVOS
# ---------------------------------------------------------
st.subheader("📈 Análisis Gráfico")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100)

# Pareto
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

ax2.plot(x_meh, y_geom_meh, color='red', linestyle='--', linewidth=2, label=f'MEH (b={b_meh:.3f}m)')
ax2.plot(x_smi, y_geom_smi, color='green', linestyle='-', linewidth=2, label=f'SMI (b={b_smi:.3f}m)')
ax2.set_title('Geometría Transversal Trapezoidal', fontweight='bold')
ax2.set_xlabel('Ancho de Sección (m)')
ax2.set_ylabel('Tirante y (m)')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
st.pyplot(fig)
