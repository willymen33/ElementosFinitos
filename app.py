import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fem import resolver


# Configuración

st.set_page_config(
    page_title="Barra Cónica FEM",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Estilos

st.markdown("""
<style>

    .stApp {
        background: linear-gradient(
            135deg,
            #f6f9fc 0%,
            #eef4f8 100%
        );
    }

    .hero {
        padding: 2rem;
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            #0f4c75,
            #3282b8
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
    }

    .hero p {
        margin-top: 0.5rem;
        font-size: 1.05rem;
        opacity: 0.9;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.07);
        border-left: 5px solid #3282b8;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #102a43,
            #243b53
        );
    }

    div[data-testid="stSidebar"] * {
        color: white;
    }

    .info-box {
        padding: 1rem;
        border-radius: 14px;
        background: white;
        border-left: 5px solid #38a169;
        box-shadow: 0 3px 12px rgba(0,0,0,0.06);
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding-top: 2rem;
        font-size: 0.85rem;
    }

</style>
""", unsafe_allow_html=True)


# Funciones analíticas

def u_exacta(x, d1, d2, L, E, P):

    if np.isclose(d1, d2):

        A = np.pi * d1**2 / 4

        return P * x / (E * A)

    d = d1 + (d2 - d1) * x / L

    return (
        4 * P * L /
        (np.pi * E * (d1 - d2))
    ) * (
        1 / d - 1 / d1
    )


def sigma_exacta(x, d1, d2, L, P):

    d = d1 + (d2 - d1) * x / L

    return 4 * P / (np.pi * d**2)


# Validación de datos

def validar_datos(d1, d2, L, E, P):

    errores = []

    if d1 <= 0:
        errores.append(
            "El diámetro d1 debe ser mayor que cero."
        )

    if d2 <= 0:
        errores.append(
            "El diámetro d2 debe ser mayor que cero."
        )

    if L <= 0:
        errores.append(
            "La longitud debe ser mayor que cero."
        )

    if E <= 0:
        errores.append(
            "El módulo de elasticidad debe ser positivo."
        )

    if P < 0:
        errores.append(
            "La carga debe ser no negativa para este caso de tracción."
        )

    return errores


# Encabezado

st.markdown("""
<div class="hero">
    <h1>📐 Barra Cónica · Elementos Finitos</h1>
    <p>
    Simulación interactiva de una barra de sección circular variable
    sometida a carga axial.
    </p>
</div>
""", unsafe_allow_html=True)


# Barra lateral

st.sidebar.title("⚙️ Parámetros del modelo")

st.sidebar.caption(
    "Modifica los valores para analizar diferentes configuraciones."
)

d1_mm = st.sidebar.number_input(
    "🔵 Diámetro inicial d₁ [mm]",
    min_value=0.0,
    value=50.0,
    step=1.0
)

d2_mm = st.sidebar.number_input(
    "🟢 Diámetro final d₂ [mm]",
    min_value=0.0,
    value=25.0,
    step=1.0
)

L_mm = st.sidebar.number_input(
    "📏 Longitud L [mm]",
    min_value=0.0,
    value=500.0,
    step=10.0
)

E_GPa = st.sidebar.number_input(
    "🔩 Módulo de elasticidad E [GPa]",
    min_value=0.0,
    value=200.0,
    step=10.0
)

P_kN = st.sidebar.number_input(
    "➡️ Carga axial P [kN]",
    min_value=0.0,
    value=40.0,
    step=5.0
)


# Conversión a SI

d1 = d1_mm * 1e-3
d2 = d2_mm * 1e-3
L = L_mm * 1e-3
E = E_GPa * 1e9
P = P_kN * 1e3


# Validación

errores = validar_datos(
    d1,
    d2,
    L,
    E,
    P
)

if errores:

    st.error(
        "⚠️ Se encontraron datos físicamente inválidos."
    )

    for mensaje in errores:
        st.warning(mensaje)

    st.stop()


# Solución exacta

if np.isclose(d1, d2):

    A = np.pi * d1**2 / 4

    u_exact_L = P * L / (E * A)

else:

    u_exact_L = (
        4 * P * L /
        (
            np.pi *
            E *
            d1 *
            d2
        )
    )


sigma_exact_L = (
    4 * P /
    (
        np.pi *
        d2**2
    )
)


# Convergencia

ns = np.array([
    1, 2, 4, 8,
    16, 32, 64, 128
])

u_lista = []
eu_lista = []

sigma_lista = []
es_lista = []


for n in ns:

    x, u, sigma = resolver(
        d1,
        d2,
        L,
        E,
        P,
        int(n)
    )

    u_lista.append(u[-1])

    sigma_lista.append(
        sigma[-1]
    )

    if np.isclose(u_exact_L, 0):

        eu = 0

    else:

        eu = (
            abs(
                u_exact_L -
                u[-1]
            )
            /
            abs(u_exact_L)
        )

    eu_lista.append(eu)


    if np.isclose(
        sigma_exact_L,
        0
    ):

        es = 0

    else:

        es = (
            abs(
                sigma_exact_L -
                sigma[-1]
            )
            /
            abs(sigma_exact_L)
        )

    es_lista.append(es)


u_lista = np.array(
    u_lista
)

eu_lista = np.array(
    eu_lista
)

sigma_lista = np.array(
    sigma_lista
)

es_lista = np.array(
    es_lista
)


# Cocientes

ru = np.full(
    len(ns),
    np.nan
)

rs = np.full(
    len(ns),
    np.nan
)


for i in range(
    len(ns) - 1
):

    if eu_lista[i + 1] != 0:

        ru[i] = (
            eu_lista[i]
            /
            eu_lista[i + 1]
        )

    if es_lista[i + 1] != 0:

        rs[i] = (
            es_lista[i]
            /
            es_lista[i + 1]
        )


# Número mínimo de elementos

n_u = None
n_s = None


for n, error in zip(
    ns,
    eu_lista
):

    if error * 100 < 0.1:

        n_u = n
        break


for n, error in zip(
    ns,
    es_lista
):

    if error * 100 < 1:

        n_s = n
        break


# Métricas principales

m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "📏 u(L) exacto",
        f"{u_exact_L * 1000:.6f} mm"
    )


with m2:

    st.metric(
        "🔩 σmáx exacto",
        f"{sigma_exact_L / 1e6:.4f} MPa"
    )


with m3:

    st.metric(
        "🎯 n para error u < 0.1 %",
        n_u if n_u else "—"
    )


with m4:

    st.metric(
        "🎯 n para error σ < 1 %",
        n_s if n_s else "—"
    )


st.divider()


# Pestañas

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Resumen",
    "📉 Convergencia",
    "📏 Desplazamiento",
    "🔩 Esfuerzo",
    "✅ Validaciones"
])


# Resumen

with tab1:

    st.subheader(
        "Resultados generales"
    )

    tabla = pd.DataFrame({

        "n":
            ns,

        "u(L) FEM [mm]":
            u_lista * 1000,

        "Error u [%]":
            eu_lista * 100,

        "r u":
            ru,

        "σ último [MPa]":
            sigma_lista / 1e6,

        "Error σ [%]":
            es_lista * 100,

        "r σ":
            rs
    })


    st.dataframe(
        tabla.style.format({

            "u(L) FEM [mm]":
                "{:.6f}",

            "Error u [%]":
                "{:.6f}",

            "r u":
                "{:.4f}",

            "σ último [MPa]":
                "{:.4f}",

            "Error σ [%]":
                "{:.6f}",

            "r σ":
                "{:.4f}"

        }),

        use_container_width=True
    )


    csv = tabla.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(

        "⬇️ Descargar tabla en CSV",

        data=csv,

        file_name=
        "convergencia_barra_conica.csv",

        mime="text/csv"
    )


    st.markdown("""
    <div class="info-box">
    <b>Interpretación:</b><br>
    Al aumentar el número de elementos,
    los resultados FEM se aproximan
    progresivamente a la solución analítica.
    </div>
    """,
    unsafe_allow_html=True
    )


# Convergencia de errores

with tab2:

    st.subheader(
        "Convergencia de los errores"
    )

    if P > 0:

        fig = go.Figure()


        fig.add_trace(
            go.Scatter(
                x=ns,
                y=eu_lista * 100,
                mode="lines+markers",
                name="Desplazamiento"
            )
        )


        fig.add_trace(
            go.Scatter(
                x=ns,
                y=es_lista * 100,
                mode="lines+markers",
                name="Esfuerzo máximo"
            )
        )


        fig.update_xaxes(
            type="log",
            title="Número de elementos n"
        )


        fig.update_yaxes(
            type="log",
            title="Error relativo [%]"
        )


        fig.update_layout(

            title=
            "Convergencia FEM",

            hovermode=
            "x unified",

            height=550
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        c1, c2 = st.columns(2)


        with c1:

            st.success(
                f"Desplazamiento: "
                f"n = {n_u} para error < 0.1 %"
            )


        with c2:

            st.info(
                f"Esfuerzo: "
                f"n = {n_s} para error < 1 %"
            )


# Desplazamiento

with tab3:

    n_sel_u = st.select_slider(

        "Número de elementos",

        options=[
            1, 2, 4, 8,
            16, 32, 64, 128
        ],

        value=8,

        key="n_u"
    )


    x, u, sigma = resolver(
        d1,
        d2,
        L,
        E,
        P,
        n_sel_u
    )


    x_cont = np.linspace(
        0,
        L,
        500
    )


    u_cont = u_exacta(
        x_cont,
        d1,
        d2,
        L,
        E,
        P
    )


    fig_u = go.Figure()


    fig_u.add_trace(
        go.Scatter(

            x=x_cont * 1000,

            y=u_cont * 1000,

            mode="lines",

            name="Solución exacta"
        )
    )


    fig_u.add_trace(
        go.Scatter(

            x=x * 1000,

            y=u * 1000,

            mode="lines+markers",

            name=f"FEM n={n_sel_u}"
        )
    )


    fig_u.update_layout(

        title=
        "Desplazamiento a lo largo de la barra",

        xaxis_title=
        "Posición x [mm]",

        yaxis_title=
        "Desplazamiento u(x) [mm]",

        hovermode=
        "x unified",

        height=550
    )


    st.plotly_chart(
        fig_u,
        use_container_width=True
    )


# Esfuerzo

with tab4:

    n_sel_s = st.select_slider(

        "Número de elementos",

        options=[
            1, 2, 4, 8,
            16, 32, 64, 128
        ],

        value=8,

        key="n_s"
    )


    x, u, sigma = resolver(
        d1,
        d2,
        L,
        E,
        P,
        n_sel_s
    )


    x_cont = np.linspace(
        0,
        L,
        500
    )


    sigma_cont = sigma_exacta(
        x_cont,
        d1,
        d2,
        L,
        P
    )


    fig_s = go.Figure()


    fig_s.add_trace(
        go.Scatter(

            x=x_cont * 1000,

            y=sigma_cont / 1e6,

            mode="lines",

            name="Solución exacta"
        )
    )


    x_step = np.repeat(
        x * 1000,
        2
    )[1:-1]


    y_step = np.repeat(
        sigma / 1e6,
        2
    )


    fig_s.add_trace(
        go.Scatter(

            x=x_step,

            y=y_step,

            mode="lines",

            name=f"FEM n={n_sel_s}"
        )
    )


    fig_s.update_layout(

        title=
        "Distribución del esfuerzo",

        xaxis_title=
        "Posición x [mm]",

        yaxis_title=
        "Esfuerzo σ(x) [MPa]",

        hovermode=
        "x unified",

        height=550
    )


    st.plotly_chart(
        fig_s,
        use_container_width=True
    )


# Validaciones

with tab5:

    st.subheader(
        "Validaciones físicas del modelo"
    )


    st.success(
        "Los datos actuales son físicamente válidos."
    )


    st.markdown("""
    La aplicación verifica automáticamente:

    - **d₁ > 0**
    - **d₂ > 0**
    - **L > 0**
    - **E > 0**
    - **P ≥ 0**

    Estas condiciones evitan configuraciones
    que no representan el problema físico
    planteado.
    """)


    with st.expander(
        "ℹ️ ¿Cómo funciona el método FEM?"
    ):

        st.write(
            """
            La barra se divide en elementos
            unidimensionales de dos nodos.

            Para cada elemento se evalúa
            el área transversal en su punto
            medio y se construye su matriz
            de rigidez.

            Posteriormente se ensamblan las
            matrices elementales, se aplica
            la condición u(0)=0 y se resuelve
            el sistema global.
            """
        )


# Pie de página

st.markdown("""
<div class="footer">
Método de los Elementos Finitos · Barra Cónica ·
Doctorado en Ingeniería
</div>
""",
unsafe_allow_html=True
)
