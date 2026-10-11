"""
Laboratorio Final — Starbucks Rewards
el uso es que una persona pueda visualizar el porcentaje de usuarios que deberia contactar para maximizar las ganancias
"""

#importamos librerias
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#agregmos icono y titulo para que se visualice en la pestaña del naegador
st.set_page_config(page_title="Laboratorio Final — Starbucks Rewards", page_icon="☕", layout="wide")

#Coloes par ala grafica
GREEN = "#00704A"
ORANGE = "#C4512F"
GRAY = "#9A988E"

#columnas que tuilizaremos para el modelo
FEATURES = ["recency_days", "frequency", "monetary", "email_open_rate", "tenure_days"]

#carpeta donde estanran los archivos csv
CARPETA = Path(__file__).parent  


#cargamos el csv
@st.cache_data
def load_data():
    return pd.read_csv(CARPETA / "data" / "uplift_campaign.csv")

# modelo 
@st.cache_data
def entrenar_t_learner(df: pd.DataFrame, test_size: float, seed: int):
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["treatment_group"])

    # t-learner: un modelo con los tratados y otro con el control
    # separamos a los que SÍ recibieron el cupón y a los que NO
    train_t = train[train["treatment_group"] == "treatment"]
    train_c = train[train["treatment_group"] == "control"]

    # Modelo 1: aprende cómo compra la gente CUANDO recibe el cupón
    modelo_tratado = LogisticRegression(max_iter=1000).fit(train_t[FEATURES], train_t["responded_60d"])
     # Modelo 2: aprende cómo compra la gente CUANDO NO recibe nada
    modelo_control = LogisticRegression(max_iter=1000).fit(train_c[FEATURES], train_c["responded_60d"])

    test = test.copy()
    X_test = test[FEATURES]
    
    test["p_tratado"] = modelo_tratado.predict_proba(X_test)[:, 1]
    test["p_control"] = modelo_control.predict_proba(X_test)[:, 1]
    test["uplift_estimado"] = test["p_tratado"] - test["p_control"]
    return train, test

#politicas
def valor_de_la_politica(test_df, seleccionados):
    #solo con los clientes que queremos contactar5
    sel = test_df.loc[seleccionados]
    #comparamos lo que ganamos con los que se les envio el cupon
    valor_tratado = sel.loc[sel["treatment_group"] == "treatment", "utilidad_neta_60d"].mean()
    valor_control = sel.loc[sel["treatment_group"] == "control", "margin_60d"].mean()
    return valor_tratado - valor_control, len(sel)

#cuva qini
def curva_qini(test_df, score_col, steps=40):
    # Misma función del notebook
    orden = test_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    es_tratado = (orden["treatment_group"] == "treatment").values
    respondio = orden["responded_60d"].values
    acumulado_tratados = np.cumsum(np.where(es_tratado, respondio, 0))
    acumulado_control = np.cumsum(np.where(~es_tratado, respondio, 0))
    n_tratados_acum = np.cumsum(es_tratado)
    n_control_acum = np.cumsum(~es_tratado)
    n = len(orden)
    xs, ys = [0.0], [0.0]
    for i in np.linspace(1, n, steps).astype(int):
        nt, nc = n_tratados_acum[i - 1], n_control_acum[i - 1]
        tasa_t = acumulado_tratados[i - 1] / nt if nt > 0 else 0
        tasa_c = acumulado_control[i - 1] / nc if nc > 0 else 0
        xs.append(i / n)
        ys.append((tasa_t - tasa_c) * n)
    return np.array(xs), np.array(ys)


def coeficiente_qini(xs, ys, n):
    area_modelo = np.trapezoid(ys, xs)
    area_azar = np.trapezoid(np.linspace(0, ys[-1], len(xs)), xs)
    return (area_modelo - area_azar) / n

#DISEÑO DE L APAGINA
# --- Título y carga de datos ---
st.title("Laboratorio Final — Starbucks Rewards - Diego Alvarado")
st.caption("Puedes decidir a qué porcentaje de clientes puedes enviar el cupón"
           "para obtener la mayor ganancia, usando el uplift estimado por un modelo T-learner.")

df = load_data()

# Control para que el usuario pueda mover el porcentaje de usuarios a econtactar
with st.sidebar:
    st.header("Parámetros")
    pct_contactar = st.slider("% de usuarios a contactar para enviar cupon", min_value=10, max_value=100, value=20, step=10)

#se entrena el modelo
train, test = entrenar_t_learner(df, test_size=0.30, seed=42)

#  Curva Qini 
st.subheader("Curva Qini del T-learner")
xs, ys = curva_qini(test, "uplift_estimado")
qini = coeficiente_qini(xs, ys, len(test))
st.write(f"Coeficiente Qini: **{qini:+.3f}**")

# Se dibuja la grafica
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(xs, ys, color=GREEN, linewidth=3, label="T-learner")
ax.plot([0, 1], [0, ys[-1]], color=GRAY, linestyle="--", label="Al azar")
ax.axvline(pct_contactar / 100, color=ORANGE, linestyle=":", linewidth=2, label=f"{pct_contactar}% elegido")
ax.set_ylim(0, 5000)  # se omite el primer punto, calculado con un solo cliente
ax.set_xlabel("% de clientes contactado para enviar cupon")
ax.set_ylabel("Respuestas incrementales")
ax.legend()
st.pyplot(fig)

#  Tabla de políticas
# ordenado 
st.subheader("Tabla de políticas")
resultados = []
for pct in range(10, 101, 10):
    n_c = int(len(test) * pct / 100)
    idx = test.sort_values("uplift_estimado", ascending=False).head(n_c).index
    valor_cliente, n = valor_de_la_politica(test, idx)
    resultados.append({"pct_contactado": pct, "valor_por_cliente": round(valor_cliente, 2),
                       "n_contactados": n, "valor_total": round(valor_cliente * n, 2)})
tabla_politicas = pd.DataFrame(resultados)

fila = tabla_politicas[tabla_politicas["pct_contactado"] == pct_contactar].iloc[0]
optimo = tabla_politicas.loc[tabla_politicas["valor_total"].idxmax()]

col1, col2, col3 = st.columns(3)
col1.metric("Clientes contactados", f"{int(fila['n_contactados']):,}")
col2.metric("Valor por cliente", f"${fila['valor_por_cliente']:,.2f}")
col3.metric("Valor total", f"${fila['valor_total']:,.2f}")

st.dataframe(tabla_politicas, hide_index=True, width="stretch")
st.success(f"El valor total máximo se obtiene contactando al {int(optimo['pct_contactado'])}% "
           f"de la base: ${optimo['valor_total']:,.2f}.")
