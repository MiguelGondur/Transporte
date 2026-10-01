import pandas as pd
import pulp
import streamlit as st

st.set_page_config(page_title="Distribución de medicamentos", layout="wide")

CENTROS = ["Medellín", "Rionegro", "Bello"]
HOSPITALES = ["Hospital Norte", "Hospital Central", "Hospital Oriente", "Hospital Sur"]

CAP_DEF = {"Medellín": 140, "Rionegro": 120, "Bello": 130}
DEM_DEF = {"Hospital Norte": 90, "Hospital Central": 100, "Hospital Oriente": 80, "Hospital Sur": 100}
COSTOS_DEF = pd.DataFrame(
    [[8, 12, 15, 11], [14, 10, 9, 16], [11, 13, 14, 8]],
    index=CENTROS, columns=HOSPITALES, dtype=float,
)
LIM_RC_DEF = 30


def resolver(capacidad, demanda, costos, limite_rc):
    """Resuelve el problema de transporte con PuLP usando solo los datos recibidos."""
    prob = pulp.LpProblem("Distribucion_Medicamentos", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{a}_{b}", lowBound=0)
        for a, i in enumerate(CENTROS)
        for b, j in enumerate(HOSPITALES)
    }
    prob += pulp.lpSum(costos.loc[i, j] * x[i, j] for i in CENTROS for j in HOSPITALES)
    for i in CENTROS:
        prob += pulp.lpSum(x[i, j] for j in HOSPITALES) <= capacidad[i], f"Cap_{i}"
    for j in HOSPITALES:
        prob += pulp.lpSum(x[i, j] for i in CENTROS) == demanda[j], f"Dem_{j}"
    prob += x["Rionegro", "Hospital Central"] <= limite_rc, "Limite_Rionegro_Central"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    estado = pulp.LpStatus[prob.status]

    envios = pd.DataFrame(
        [[x[i, j].value() or 0 for j in HOSPITALES] for i in CENTROS],
        index=CENTROS, columns=HOSPITALES,
    ).round(2)
    costo = pulp.value(prob.objective) if estado == "Optimal" else None
    return estado, costo, envios


st.title("🚚 Optimización de distribución de medicamentos")
st.caption("Modifique los datos y presione **Optimizar distribución**.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Capacidad máxima por centro")
    cap_df = st.data_editor(
        pd.DataFrame({"Capacidad": CAP_DEF}), key="cap", use_container_width=True
    )
with col2:
    st.subheader("Demanda por hospital")
    dem_df = st.data_editor(
        pd.DataFrame({"Demanda": DEM_DEF}), key="dem", use_container_width=True
    )

st.subheader("Costos unitarios de transporte")
costos_df = st.data_editor(COSTOS_DEF, key="costos", use_container_width=True)

limite_rc = st.number_input(
    "Capacidad máxima de la ruta Rionegro → Hospital Central",
    min_value=0.0, value=float(LIM_RC_DEF), step=1.0,
)

if st.button("Optimizar distribución", type="primary"):
    capacidad = cap_df["Capacidad"].astype(float).to_dict()
    demanda = dem_df["Demanda"].astype(float).to_dict()
    costos = costos_df.astype(float)

    cap_total = sum(capacidad.values())
    dem_total = sum(demanda.values())

    if dem_total > cap_total:
        st.warning(
            f"⚠️ La demanda total ({dem_total:,.0f}) es superior a la capacidad total "
            f"disponible ({cap_total:,.0f}). Con las condiciones ingresadas no existe "
            "capacidad suficiente para satisfacer completamente la demanda."
        )

    estado, costo, envios = resolver(capacidad, demanda, costos, limite_rc)

    if estado == "Optimal":
        st.success("Estado de la solución: Óptima")
    else:
        st.error(f"Estado de la solución: {estado} (no se encontró una distribución factible)")

    utilizacion = (dem_total / cap_total * 100) if cap_total > 0 else 0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Costo mínimo total", f"${costo:,.2f}" if costo is not None else "N/D")
    m2.metric("Capacidad total", f"{cap_total:,.0f}")
    m3.metric("Demanda total", f"{dem_total:,.0f}")
    m4.metric("Utilización de capacidad", f"{utilizacion:.1f}%")
    m5.metric("Estado", estado)

    if estado == "Optimal":
        st.subheader("Cantidades óptimas enviadas (centro → hospital)")
        st.dataframe(envios, use_container_width=True)

        usado = envios.sum(axis=1)
        resumen = pd.DataFrame({
            "Capacidad": pd.Series(capacidad),
            "Utilizada": usado,
            "No utilizada": pd.Series(capacidad) - usado,
            "% utilización": (usado / pd.Series(capacidad) * 100).round(1),
        })
        st.subheader("Uso de capacidad por centro")
        st.dataframe(resumen, use_container_width=True)
