# ============ Celda 1: instalación ============
# !pip install "pulp<4" -q

# ============ Celda 2: datos ingresados por el usuario ============
# (Cambie estos valores; el modelo NO tiene datos fijos internos)
import pandas as pd
import pulp

centros = ["Medellín", "Rionegro", "Bello"]
hospitales = ["Hospital Norte", "Hospital Central", "Hospital Oriente", "Hospital Sur"]

capacidad = {"Medellín": 140, "Rionegro": 120, "Bello": 130}
demanda = {"Hospital Norte": 90, "Hospital Central": 100, "Hospital Oriente": 80, "Hospital Sur": 100}
costos = pd.DataFrame(
    [[8, 12, 15, 11], [14, 10, 9, 16], [11, 13, 14, 8]],
    index=centros, columns=hospitales,
)
limite_rionegro_central = 30

# ============ Celda 3: modelo ============
def resolver(centros, hospitales, capacidad, demanda, costos, limite_rc):
    prob = pulp.LpProblem("Distribucion_Medicamentos", pulp.LpMinimize)
    x = {(i, j): pulp.LpVariable(f"x_{a}_{b}", lowBound=0)
         for a, i in enumerate(centros) for b, j in enumerate(hospitales)}

    prob += pulp.lpSum(costos.loc[i, j] * x[i, j] for i in centros for j in hospitales)

    for i in centros:
        prob += pulp.lpSum(x[i, j] for j in hospitales) <= capacidad[i]
    for j in hospitales:
        prob += pulp.lpSum(x[i, j] for i in centros) == demanda[j]
    prob += x["Rionegro", "Hospital Central"] <= limite_rc

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    estado = pulp.LpStatus[prob.status]
    envios = pd.DataFrame(
        [[x[i, j].value() or 0 for j in hospitales] for i in centros],
        index=centros, columns=hospitales).round(2)
    costo = pulp.value(prob.objective) if estado == "Optimal" else None
    return estado, costo, envios

# ============ Celda 4: resultados ============
estado, costo, envios = resolver(centros, hospitales, capacidad, demanda, costos, limite_rionegro_central)

print("Estado de la solución:", estado)
print("Costo mínimo total:", costo)

if estado == "Optimal":
    usado = envios.sum(axis=1)
    resumen = pd.DataFrame({
        "Capacidad": pd.Series(capacidad),
        "Utilizada": usado,
        "No utilizada": pd.Series(capacidad) - usado,
    })
    print("\nEnvíos óptimos (centro → hospital):")
    display(envios)
    print("\nCapacidad utilizada y disponible:")
    display(resumen)
else:
    print("No hay solución factible con los datos ingresados.")
