import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

st.set_page_config(layout="wide", page_title="Machine Learning App", page_icon="🤖")

st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
    }
    .stButton>button {
        background-color: #0066CC;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 24px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #0052A3;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: white !important;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_excel('C:\\Users\\Dennis\\PycharmProjects\\Lucifer\\pythonProject8\\Pymes y variables_ING.xlsx')

    sector_names = {
        "Sector 11": "Agricultura, cría y explotación de animales, aprovechamiento forestal, pesca y caza",
        "Sector 21": "Minería",
        "Sector 22": "Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final",
        "Sector 23": "Construcción",
        "Sector 31-33": "Industrias manufactureras",
        "Sector 43": "Comercio al por mayor",
        "Sector 46": "Comercio al por menor",
        "Sector 48-49": "Transportes, correos y almacenamiento",
        "Sector 51": "Información en medios masivos",
        "Sector 52": "Servicios financieros y de seguros",
        "Sector 53": "Servicios inmobiliarios y de alquiler de bienes muebles e intangibles",
        "Sector 54": "Servicios profesionales, científicos y técnicos",
        "Sector 55": "Dirección y administración de grupos empresariales o corporativos",
        "Sector 56": "Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación",
        "Sector 61": "Servicios educativos",
        "Sector 62": "Servicios de salud y de asistencia social",
        "Sector 71": "Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos",
        "Sector 72": "Servicios de alojamiento temporal y de preparación de alimentos y bebidas",
        "Sector 81": "Otros servicios excepto actividades gubernamentales",
        "Sector 93": "Actividades legislativas, gubernamentales, de impartición de justicia y de organismos internacionales y extraterritoriales"
    }

    df['Sector'] = df['Sector'].replace(sector_names)
    return df


df = load_data()

if 'resultados_kmeans' not in st.session_state:
    st.session_state.resultados_kmeans = None
if 'resultados_regresion' not in st.session_state:
    st.session_state.resultados_regresion = None

st.title("Aplicación de Machine Learning")
st.markdown("### Análisis con K-Means Clustering y Regresión Lineal")
st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.header("K-MEANS CLUSTERING")
    st.markdown("**Tipo:** Aprendizaje No Supervisado")
    st.markdown("---")

    variables_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    st.subheader("Configuración del Modelo")

    kmeans_vars = st.multiselect(
        "Selecciona las columnas para clustering:",
        variables_numericas,
        default=[variables_numericas[0], variables_numericas[1]] if len(
            variables_numericas) >= 2 else variables_numericas,
        key='kmeans_vars'
    )

    if kmeans_vars:
        st.markdown("**Variables seleccionadas:**")
        for var in kmeans_vars:
            st.markdown(f"- {var}")

    n_clusters = st.slider("Número de Clusters:", min_value=2, max_value=10, value=3, key='n_clusters')

    ejecutar_kmeans = st.button("Ejecutar K-Means", type="primary", key='btn_kmeans')

    if ejecutar_kmeans:
        if len(kmeans_vars) < 2:
            st.error("Selecciona al menos 2 variables")
        else:
            df_kmeans = df[kmeans_vars].dropna()

            if len(df_kmeans) < n_clusters:
                st.error("No hay suficientes datos para el número de clusters seleccionado")
            else:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_kmeans)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)

                df_kmeans['Cluster'] = clusters
                st.session_state.resultados_kmeans = {
                    'df': df_kmeans,
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'vars': kmeans_vars,
                    'n_clusters': n_clusters
                }

                st.success(f"Modelo K-Means entrenado con {n_clusters} clusters")

with col_right:
    st.header("REGRESIÓN LINEAL")
    st.markdown("**Tipo:** Aprendizaje Supervisado")
    st.markdown("---")

    st.subheader("Configuración del Modelo")

    variable_objetivo = st.selectbox(
        "Variable Dependiente (Y) a predecir:",
        variables_numericas,
        key='var_y'
    )

    if variable_objetivo:
        st.markdown(f"**Variable a predecir:** {variable_objetivo}")

    variables_predictoras_disponibles = [v for v in variables_numericas if v != variable_objetivo]
    variables_predictoras = st.multiselect(
        "Variables Independientes (X) predictoras:",
        variables_predictoras_disponibles,
        default=variables_predictoras_disponibles[:3] if len(
            variables_predictoras_disponibles) >= 3 else variables_predictoras_disponibles,
        key='vars_x'
    )

    if variables_predictoras:
        st.markdown("**Variables predictoras:**")
        for var in variables_predictoras:
            st.markdown(f"- {var}")

    ejecutar_regresion = st.button("Ejecutar Regresión Lineal", type="primary", key='btn_regression')

    if ejecutar_regresion:
        if not variables_predictoras:
            st.error("Selecciona al menos una variable predictora")
        else:
            columnas_modelo = [variable_objetivo] + variables_predictoras
            df_modelo = df[columnas_modelo].dropna()

            if len(df_modelo) < 10:
                st.error("No hay suficientes datos para entrenar el modelo")
            else:
                X = df_modelo[variables_predictoras]
                y = df_modelo[variable_objetivo]

                modelo = LinearRegression()
                modelo.fit(X, y)
                y_pred = modelo.predict(X)

                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                rmse = np.sqrt(mean_squared_error(y, y_pred))

                st.session_state.resultados_regresion = {
                    'modelo': modelo,
                    'X': X,
                    'y': y,
                    'y_pred': y_pred,
                    'r2': r2,
                    'mae': mae,
                    'rmse': rmse,
                    'var_objetivo': variable_objetivo,
                    'vars_predictoras': variables_predictoras
                }

                st.success("Modelo de Regresión Lineal entrenado exitosamente")

st.markdown("---")

if st.session_state.resultados_kmeans is not None:
    resultados = st.session_state.resultados_kmeans
    df_kmeans = resultados['df']
    kmeans = resultados['kmeans']
    scaler = resultados['scaler']
    kmeans_vars = resultados['vars']
    n_clusters = resultados['n_clusters']

    st.header("Resultados K-Means Clustering")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Visualización del Clustering")

        if len(kmeans_vars) >= 2:
            fig_kmeans = px.scatter(
                df_kmeans,
                x=kmeans_vars[0],
                y=kmeans_vars[1],
                color='Cluster',
                title=f'Clustering: {kmeans_vars[0]} vs {kmeans_vars[1]}',
                color_continuous_scale='Viridis',
                template='plotly_dark'
            )

            centroides_originales = scaler.inverse_transform(kmeans.cluster_centers_)
            fig_kmeans.add_trace(go.Scatter(
                x=centroides_originales[:, 0],
                y=centroides_originales[:, 1],
                mode='markers',
                marker=dict(color='red', size=15, symbol='x', line=dict(width=2, color='white')),
                name='Centroides'
            ))

            st.plotly_chart(fig_kmeans, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - Cada punto representa una observación de los datos
        - Los colores indican a qué cluster pertenece cada punto
        - Las X rojas marcan los centroides (centro de cada cluster)
        - Observaciones similares se agrupan en el mismo cluster
        """)

    with col2:
        st.subheader("Distribución de Clusters")

        fig_dist = px.bar(
            df_kmeans['Cluster'].value_counts().reset_index(),
            x='Cluster',
            y='count',
            title='Número de Observaciones por Cluster',
            color='count',
            color_continuous_scale='Blues',
            template='plotly_dark',
            labels={'count': 'Cantidad'}
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - Muestra cuántas observaciones pertenecen a cada cluster
        - Permite identificar si los clusters están balanceados
        - Clusters con pocas observaciones pueden ser outliers o casos especiales
        """)

    st.subheader("Estadísticas por Cluster")
    cluster_stats = df_kmeans.groupby('Cluster')[kmeans_vars].mean()
    st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'), use_container_width=True)

    st.markdown("""
    **Interpretación de la tabla:**
    - Muestra el valor promedio de cada variable en cada cluster
    - Permite identificar las características distintivas de cada grupo
    - Valores más altos en azul más intenso, valores más bajos en azul más claro
    """)

    st.subheader("Análisis Detallado de los Resultados")

    cluster_counts = df_kmeans['Cluster'].value_counts()
    cluster_mas_grande = cluster_counts.idxmax()
    cluster_mas_pequeno = cluster_counts.idxmin()

    st.markdown(f"""
    ### Resumen del Clustering:

    **Número total de clusters creados:** {n_clusters}

    **Distribución de los datos:**
    - **Cluster más poblado:** Cluster {cluster_mas_grande} con {cluster_counts[cluster_mas_grande]} observaciones
    - **Cluster menos poblado:** Cluster {cluster_mas_pequeno} con {cluster_counts[cluster_mas_pequeno]} observaciones

    **¿Qué significa cada cluster?**
    """)

    for cluster_id in sorted(df_kmeans['Cluster'].unique()):
        cluster_data = df_kmeans[df_kmeans['Cluster'] == cluster_id]
        st.markdown(f"""
        **Cluster {cluster_id}:**
        - Contiene {len(cluster_data)} observaciones ({len(cluster_data) / len(df_kmeans) * 100:.1f}% del total)
        - Características promedio:
        """)
        for var in kmeans_vars:
            promedio = cluster_data[var].mean()
            st.markdown(f"  - {var}: {promedio:,.2f}")

    st.markdown("""
    ### ¿Cómo interpretar estos resultados?

    **El algoritmo K-Means ha agrupado tus datos en grupos con características similares:**

    1. **Clusters grandes** (muchas observaciones): Representan patrones comunes en tus datos
    2. **Clusters pequeños** (pocas observaciones): Pueden ser casos atípicos o situaciones excepcionales
    3. **Centroides** (X rojas en la gráfica): Son el "centro" de cada grupo, representan el punto más característico

    **Aplicaciones prácticas:**
    - Identificar segmentos de mercado con características similares
    - Detectar patrones o comportamientos inusuales
    - Agrupar datos para análisis más específicos
    - Tomar decisiones estratégicas basadas en grupos homogéneos
    """)

st.markdown("---")

if st.session_state.resultados_regresion is not None:
    resultados = st.session_state.resultados_regresion
    modelo = resultados['modelo']
    y = resultados['y']
    y_pred = resultados['y_pred']
    r2 = resultados['r2']
    mae = resultados['mae']
    rmse = resultados['rmse']
    variable_objetivo = resultados['var_objetivo']
    variables_predictoras = resultados['vars_predictoras']

    st.header("Resultados Regresión Lineal")

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("R² (Coeficiente de Determinación)", f"{r2:.3f}")
    col_m2.metric("MAE (Error Absoluto Medio)", f"{mae:,.2f}")
    col_m3.metric("RMSE (Raíz del Error Cuadrático Medio)", f"{rmse:,.2f}")

    st.markdown("""
    **Significado de las métricas:**
    - **R²:** Indica qué porcentaje de la variabilidad de Y es explicada por las variables X. Valores cercanos a 1 indican mejor ajuste.
    - **MAE:** Promedio de los errores absolutos entre valores reales y predichos. Menor valor es mejor.
    - **RMSE:** Similar al MAE pero penaliza más los errores grandes. Menor valor es mejor.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Valores Reales vs Predichos")

        fig_pred = go.Figure()

        fig_pred.add_trace(go.Scatter(
            x=y,
            y=y_pred,
            mode='markers',
            name='Predicciones',
            marker=dict(color='#0066CC', size=8, opacity=0.6)
        ))

        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Predicción Perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))

        fig_pred.update_layout(
            title='Comparación: Valores Reales vs Predichos',
            xaxis_title=f'{variable_objetivo} (Real)',
            yaxis_title=f'{variable_objetivo} (Predicho)',
            template='plotly_dark'
        )

        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - Cada punto azul representa una observación
        - La línea roja muestra dónde deberían estar los puntos si la predicción fuera perfecta
        - Puntos cercanos a la línea roja indican buenas predicciones
        - Puntos alejados indican mayor error de predicción
        """)

    with col2:
        st.subheader("Importancia de Variables")

        coef_df = pd.DataFrame({
            'Variable': variables_predictoras,
            'Coeficiente': modelo.coef_
        })
        coef_df = coef_df.sort_values('Coeficiente', key=abs, ascending=False)

        fig_coef = px.bar(
            coef_df,
            x='Coeficiente',
            y='Variable',
            orientation='h',
            title='Coeficientes del Modelo',
            color='Coeficiente',
            color_continuous_scale='RdBu_r',
            template='plotly_dark'
        )
        st.plotly_chart(fig_coef, use_container_width=True)

        st.markdown("""
        **Interpretación:**
        - **Coeficiente positivo (azul):** Cuando esta variable aumenta, Y aumenta
        - **Coeficiente negativo (rojo):** Cuando esta variable aumenta, Y disminuye
        - **Magnitud del coeficiente:** Indica qué tan fuerte es el efecto de cada variable
        - Barras más largas = mayor impacto en la predicción
        """)

    st.subheader("Ecuación del Modelo")
    ecuacion = f"**{variable_objetivo} = {modelo.intercept_:.2f}**"
    for var, coef in zip(variables_predictoras, modelo.coef_):
        signo = "+" if coef >= 0 else ""
        ecuacion += f" **{signo} {coef:.2f} × {var}**"

    st.markdown(ecuacion)

    st.markdown("""
    **Interpretación de la ecuación:**
    - Esta fórmula permite calcular el valor predicho de la variable objetivo
    - Cada término muestra cómo cada variable contribuye al resultado final
    - El primer número es el intercepto (valor base cuando todas las X son 0)
    """)

    st.subheader("Análisis Detallado de los Resultados")

    if r2 >= 0.9:
        calidad_modelo = "Excelente"
        color_calidad = "green"
    elif r2 >= 0.7:
        calidad_modelo = "Bueno"
        color_calidad = "blue"
    elif r2 >= 0.5:
        calidad_modelo = "Moderado"
        color_calidad = "orange"
    else:
        calidad_modelo = "Bajo"
        color_calidad = "red"

    st.markdown(f"""
    ### Evaluación del Modelo:

    **Calidad del ajuste:** :{color_calidad}[{calidad_modelo}] (R² = {r2:.3f})

    **¿Qué tan bien predice el modelo?**
    - El modelo explica el **{r2 * 100:.1f}%** de la variabilidad de {variable_objetivo}
    - El {100 - r2 * 100:.1f}% restante se debe a otros factores no incluidos en el modelo

    **Error promedio de predicción:**
    - MAE: {mae:,.2f} - En promedio, el modelo se equivoca por ±{mae:,.2f} unidades
    - RMSE: {rmse:,.2f} - Penaliza más los errores grandes
    """)

    coef_abs = [(var, abs(coef)) for var, coef in zip(variables_predictoras, modelo.coef_)]
    coef_abs.sort(key=lambda x: x[1], reverse=True)
    var_mas_importante = coef_abs[0][0]
    var_menos_importante = coef_abs[-1][0]

    st.markdown(f"""
    ### Variables más influyentes:

    **Variable con mayor impacto:** {var_mas_importante}
    - Es la que más afecta el valor de {variable_objetivo}

    **Variable con menor impacto:** {var_menos_importante}
    - Tiene el efecto más pequeño en la predicción

    ### Dirección de las relaciones:
    """)

    for var, coef in zip(variables_predictoras, modelo.coef_):
        if coef > 0:
            direccion = "positiva"
            explicacion = f"aumentar {var} en 1 unidad incrementa {variable_objetivo} en {coef:.2f}"
        else:
            direccion = "negativa"
            explicacion = f"aumentar {var} en 1 unidad disminuye {variable_objetivo} en {abs(coef):.2f}"

        st.markdown(f"- **{var}**: Relación {direccion} - {explicacion}")

    st.markdown(f"""
    ### ¿Cómo usar este modelo?

    **Para hacer una predicción nueva:**
    1. Toma los valores de las variables predictoras
    2. Multiplica cada una por su coeficiente
    3. Suma todos los resultados más el intercepto ({modelo.intercept_:.2f})

    **Ejemplo práctico:**
    Si quieres predecir {variable_objetivo}, necesitas conocer los valores de:
    {', '.join(variables_predictoras)}

    **Limitaciones del modelo:**
    - Solo funciona dentro del rango de datos con el que fue entrenado
    - Asume relaciones lineales entre las variables
    - No considera interacciones complejas entre variables
    - R² de {r2:.3f} indica que hay factores adicionales no capturados

    **Recomendaciones:**
    """)

    if r2 < 0.5:
        st.markdown("- Considera agregar más variables predictoras para mejorar el modelo")
        st.markdown("- El modelo actual tiene limitaciones, úsalo con precaución")
    elif r2 < 0.7:
        st.markdown("- El modelo es aceptable pero podría mejorarse")
        st.markdown("- Considera revisar si hay otras variables relevantes")
    else:
        st.markdown("- El modelo tiene un buen desempeño predictivo")
        st.markdown("- Las variables seleccionadas explican bien el comportamiento de la variable objetivo")

st.markdown("---")
st.caption("Aplicación de Machine Learning | K-Means Clustering & Regresión Lineal")