import os
import streamlit as st
import pandas as pd
import numpy as np
import pyreadstat
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Path relativo al directorio del script — funciona sin importar desde dónde se ejecute
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "GEIH2025_Barranquilla.dta")

# ─── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEIH 2025 · Barranquilla · Jóvenes",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; max-width: 1340px; }

/* ── Tarjeta KPI ─────────────────────────────────────────── */
.kpi-box {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px 22px 16px 22px;
    border-left: 5px solid var(--c);
    box-shadow: 0 2px 8px rgba(15,45,94,.07), 0 1px 2px rgba(0,0,0,.04);
    margin-bottom: 10px;
    transition: box-shadow .15s;
}
.kpi-box:hover { box-shadow: 0 4px 14px rgba(15,45,94,.13); }
.kpi-label {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .6px; color: #64748B; margin-bottom: 6px;
}
.kpi-value {
    font-size: 30px; font-weight: 800; color: #0F172A;
    line-height: 1.1; letter-spacing: -.5px;
}
.kpi-sub { font-size: 11px; color: #94A3B8; margin-top: 5px; font-weight: 500; }

/* ── Encabezado de sección ───────────────────────────────── */
.section-header {
    font-size: 15px; font-weight: 700; padding: 6px 0 14px 0;
    border-bottom: 2px solid var(--c); margin-bottom: 18px;
    color: var(--c); letter-spacing: .1px;
}

/* ── Recuadro de cada gráfica ────────────────────────────── */
.chart-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 18px 16px 10px 16px;
    box-shadow: 0 2px 10px rgba(15,45,94,.08), 0 1px 3px rgba(0,0,0,.04);
    margin-bottom: 4px;
}

/* ── Pestañas ────────────────────────────────────────────── */
div[data-testid="stTab"] button {
    font-weight: 600 !important; font-size: 13px !important;
}

/* ── Fondo general ligeramente cálido ────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section[data-testid="stSidebar"],
.stApp {
    background-color: #ffffff !important;
}
[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ─── Paleta profesional ───────────────────────────────────────────────────────
NAVY   = "#0F2D5E"   # azul marino - encabezados
BLUE   = "#1D4ED8"   # azul real   - hombres / primario
VIOLET = "#7C3AED"   # violeta     - mujeres
GREEN  = "#059669"   # esmeralda   - positivo / ocupación
AMBER  = "#B45309"   # ámbar       - alerta / informalidad
RED    = "#DC2626"   # rojo        - negativo
TEAL   = "#0D9488"   # teal        - calidad de vida
ORANGE = "#C2410C"   # naranja     - IPM
SLATE  = "#475569"   # gris pizarra - neutro

# compatibilidad con código que usa PURPLE
PURPLE = VIOLET

SEX_MAP    = {BLUE: "Hombre", VIOLET: "Mujer"}
SEX_COLORS = {"Hombre": BLUE, "Mujer": VIOLET}
MIXED = [BLUE, VIOLET, TEAL, ORANGE, GREEN, AMBER, "#0284C7", "#6D28D9", "#BE185D", SLATE]

# ─── Cargar datos ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando base de datos GEIH 2025...")
def load_data():
    df, meta = pyreadstat.read_dta(DATA_PATH)
    df.columns = [c.lower() for c in df.columns]

    # Sexo
    df["sexo"] = df["p3271"].map({1: "Hombre", 2: "Mujer"})

    # Grupos de edad
    df["grupo_edad"] = pd.cut(
        df["p6040"],
        bins=[13, 17, 19, 22, 25, 28],
        labels=["14-17", "18-19", "20-22", "23-25", "26-28"],
        right=True,
    )

    # Condición laboral derivada
    df["_ocu"] = df["oci"] == 1
    df["_des"] = df["dsi"] == 1
    df["_ina"] = df["fft"] == 1
    df["_pet"] = df["pet"] == 1
    df["_est"] = df["p6170"] == 1

    df["condicion"] = np.where(
        df["_ocu"], "Ocupado",
        np.where(df["_des"], "Desocupado",
        np.where(df["_ina"], "Inactivo", pd.NA))
    )

    # Posición ocupacional: p6430 (1=emp.particular, 2=gobierno, 3=doméstico,
    #                               4=cuenta propia, 5=trab.familiar, 6=patrón)
    # p6430 NO es pensión — era un error conceptual del código original.

    # Informal: ocupado que NO cotiza a pensión
    # p6590: cotización para asalariados (1=cotiza, 2=no cotiza, 9=NS)
    # p6600: cotización para independientes (1=cotiza, 2=no cotiza, 9=NS)
    inf_asal  = df["_ocu"] & (df["p6590"] == 2)
    inf_indep = df["_ocu"] & (df["p6600"] == 2)
    df["_inf"] = inf_asal | inf_indep

    # No estudia
    no_est = ~df["_est"]

    # Jóvenes con Potencial — traducción exacta del código Stata:
    # jp = 1 si No Estudia & (dsi==1 | fft==1 | informal==1)
    jp_des  = no_est & (df["dsi"] == 1)          # no estudia + desocupado
    jp_ina  = no_est & (df["fft"] == 1)           # no estudia + inactivo
    jp_inf  = no_est & df["_inf"]                 # no estudia + informal
    df["_jp"]   = jp_des | jp_ina | jp_inf

    # NINI: no estudia y no trabaja (des + ina) — subcomponente de JP
    df["_nini"] = df["_pet"] & (jp_des | jp_ina)

    # Factor de expansión mensualizado: fex_c18 / 12
    df["w"] = df["fex_c18"] / 12

    return df

df = load_data()
W  = "w"   # peso mensualizado

# ─── Cargar ECV 2026 ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando ECV 2026...")
def load_ecv():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_ECV26.csv")
    ecv = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    ecv["fex"] = ecv["FEX_C"].astype(str).str.replace(",", ".").astype(float)
    ecv = ecv[ecv["TERRITORIO"] == "Barranquilla"].copy()
    ecv = ecv[ecv["edad"].isin(["14-17", "18-23", "24-28"])].copy()
    sat_map = {"Totalmente satisfecho": 10, "Totalmente insatisfecho": 1}
    sat_map.update({str(i): i for i in range(1, 10)})
    for col in ["satisfecho_salud", "satisfecho_vida", "satisfecho_tiempolibre"]:
        ecv[col + "_n"] = pd.to_numeric(ecv[col].map(sat_map), errors="coerce")
    ecv["Sexo"] = ecv["sexo"].map({"Hombres": "Hombre", "Mujeres": "Mujer"})
    return ecv

@st.cache_data(show_spinner="Cargando IPM...")
def load_ipm():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datos_IPM.csv")
    ipm = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    ipm = ipm[(ipm["DEPARTAMENTO"] == 13) & (ipm["clase"] == "Cabecera")].copy()
    ipm = ipm[ipm["edad"].isin(["14-17", "18-23", "24-28"])].copy()
    ipm["Sexo"] = ipm["sexo"].map({"Hombres": "Hombre", "Mujeres": "Mujer"})
    return ipm

ecv = load_ecv()
ipm = load_ipm()

# ─── Etiquetas de categorías ──────────────────────────────────────────────────
LAB = {
    "estado_civil": {1:"Unión libre <2 años",2:"Unión libre ≥2 años",3:"Casado(a)",
                     4:"Separado/Divorciado",5:"Viudo(a)",6:"Soltero(a)"},
    "etnia":        {1:"Indígena",2:"Gitano/Rom",3:"Raizal",4:"Palenquero",
                     5:"Negro/Afrocolombiano",6:"Ninguno"},
    "posicion":     {1:"Emp. particular",2:"Emp. gobierno",3:"Emp. doméstico",
                     4:"Cuenta propia",5:"Trab. familiar",6:"Patrón/Empleador"},
    "pension":      {1:"Cotiza",2:"No cotiza",9:"No sabe"},
    "afiliacion":   {1:"Afiliado",2:"No afiliado"},
    "regimen":      {1:"Contributivo",2:"Subsidiado",3:"Especial/excepción",9:"No sabe"},
    "asistencia":   {1:"Sí asiste",2:"No asiste"},
    "alfabetismo":  {1:"Sí",2:"No"},
    "nivel_edu":    {1:"Ninguno",2:"Preescolar",3:"Primaria",4:"Secundaria",5:"Media",
                     6:"Normalista",7:"Técnico/Tec. incompleto",8:"Técnico/Tecnológico",
                     9:"Universitario incompleto",10:"Universitario",
                     11:"Especialización",12:"Maestría/Doctorado"},
    "busqueda": {
        "p3362s1":"Ayuda familia/amigos",
        "p3362s2":"Llevó hojas de vida",
        "p3362s3":"SENA/bolsa de empleo/SPE",
        "p3362s4":"Avisos clasificados",
        "p3362s5":"Redes sociales/internet",
        "p3362s6":"Esperó resultados anteriores",
        "p3362s7":"Gestiones para negocio propio",
        "p3362s8":"Otra gestión",
    },
    "sector": {
        47:"Comercio minorista",56:"Restaurantes/bares",82:"Apoyo empresarial",
        86:"Salud humana",96:"Otros serv. personales",49:"Transporte terrestre",
        41:"Construcción edif.",85:"Educación",97:"Hogares empleadores",
        10:"Elaboración alimentos",43:"Act. especializadas construc.",
        47:"Comercio minorista",78:"Empleo temporal",62:"Informática/software",
        84:"Admón. pública",52:"Almacenamiento/transporte",
    },
}

# ─── Helpers ─────────────────────────────────────────────────────────────────
def wfreq(col, cat_labels=None, mask=None, by_sex=True):
    """Frecuencia ponderada, opcionalmente por sexo."""
    d = df if mask is None else df[mask]
    if by_sex:
        g = d.groupby(["sexo", col])[W].sum().reset_index()
        g.columns = ["Sexo", "cat", "n"]
    else:
        g = d.groupby(col)[W].sum().reset_index()
        g.columns = ["cat", "n"]
    if cat_labels:
        g["cat"] = g["cat"].map(cat_labels)
    g = g.dropna(subset=["cat"])
    g["n"] = g["n"].round(0).astype(int)
    if by_sex:
        g["pct"] = g.groupby("Sexo")["n"].transform(lambda x: x / x.sum() * 100).round(1)
    else:
        g["pct"] = (g["n"] / g["n"].sum() * 100).round(1)
    return g

def kpis(mask=None):
    d = df if mask is None else df[mask]
    pet   = d.loc[d["_pet"], W].sum()
    ocu   = d.loc[d["_pet"] & d["_ocu"], W].sum()
    des   = d.loc[d["_pet"] & d["_des"], W].sum()
    ft    = d.loc[d["ft"] == 1, W].sum()
    inf_m = d["_pet"] & d["_ocu"] & ((d["p6590"] == 2) | (d["p6600"] == 2))
    inf   = d.loc[inf_m, W].sum()
    ing   = d.loc[d["_pet"] & d["_ocu"] & d["inglabo"].notna() & (d["inglabo"] > 0), "inglabo"]
    edu   = d.loc[d["_pet"] & d["p3042"].isin([8,9,10,11,12]), W].sum()
    jp   = d.loc[d["_jp"]  & d["_pet"], W].sum()
    nini = d.loc[d["_nini"]& d["_pet"], W].sum()
    return {
        "TO":   round(ocu / pet * 100, 1) if pet else 0,
        "TD":   round(des / ft  * 100, 1) if ft  else 0,
        "INF":  round(inf / ocu * 100, 1) if ocu else 0,
        "ING":  int(round(ing.median())) if len(ing) else 0,
        "EDU":  round(edu / pet * 100, 1) if pet else 0,
        "JP":   round(jp  / pet * 100, 1) if pet else 0,
        "OC":   int(round(ocu)), "DES": int(round(des)),
        "PET":  int(round(pet)),
        "JP_N": int(round(jp)),
        "NINI_N": int(round(nini)),
        "INF_N":  int(round(inf)),
    }

def layout_chart():
    return dict(
        # fondo blanco puro, sin cuadrícula
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter",
        font_color="#334155",
        title_font_size=14,
        title_font_color="#0F172A",
        title_font_family="Inter",
        # título arriba con su espacio, leyenda debajo del eje X sin tocarlo
        margin=dict(t=52, b=95, l=16, r=20),
        legend=dict(
            orientation="h",
            yanchor="top",    y=-0.32,
            xanchor="center", x=0.5,
            font_size=12, font_color="#475569",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
        ),
        # eje X: solo línea base, sin grid
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            linecolor="#E2E8F0",
            linewidth=1,
            tickfont=dict(size=11, color="#64748B"),
        ),
        # eje Y: líneas horizontales muy tenues, sin línea de eje
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            gridwidth=1,
            zeroline=False,
            linecolor="rgba(0,0,0,0)",
            tickfont=dict(size=11, color="#64748B"),
        ),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#CBD5E1",
            font_size=12,
            font_family="Inter",
            font_color="#0F172A",
        ),
    )

def bar_sex(data, x="cat", y="n", title="", h=False, pct=False, xlabel="", ylabel=""):
    yval = "pct" if pct else y
    text_vals = data[yval].apply(lambda v: f"{v:.1f}%" if pct else f"{int(v):,}")
    fig = px.bar(
        data,
        x=x if not h else yval,
        y=yval if not h else x,
        color="Sexo", barmode="group",
        color_discrete_map=SEX_COLORS,
        title=title,
        orientation="h" if h else "v",
        text=text_vals,
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=1,
        textposition="outside",
        textfont=dict(size=10, color="#334155"),
    )
    fig.update_layout(**layout_chart(), height=330,
                      xaxis_title=xlabel, yaxis_title=ylabel)
    if pct:
        if h:
            fig.update_xaxes(ticksuffix="%")
        else:
            fig.update_yaxes(ticksuffix="%")
    return fig

def chart(fig, key=None):
    """Envuelve una figura en un contenedor con borde nativo de Streamlit."""
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, key=key)

def layout_pie(title, height=310):
    """Layout limpio para gráficas de torta/donut — sin ejes, fondo blanco."""
    return dict(
        title_text=title,
        title_font_size=14,
        title_font_color="#0F172A",
        title_font_family="Inter",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter",
        font_color="#334155",
        height=height,
        margin=dict(t=56, b=16, l=8, r=8),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="white", bordercolor="#CBD5E1",
            font_size=12, font_family="Inter", font_color="#0F172A",
        ),
    )

def donut_pair(col, cat_labels, title, colors=MIXED):
    """Dos donuts lado a lado: Hombre / Mujer."""
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "pie"}]],
        subplot_titles=["Hombres", "Mujeres"],
    )
    for i, sex in enumerate(["Hombre", "Mujer"], 1):
        d = df[df["sexo"] == sex].copy()
        g = d.groupby(col)[W].sum().reset_index()
        g.columns = ["cat", "n"]
        g["cat"] = g["cat"].map(cat_labels)
        g = g.dropna(subset=["cat"])
        fig.add_trace(go.Pie(
            labels=g["cat"].tolist(),
            values=g["n"].tolist(),
            name=sex,
            hole=0.62,
            marker=dict(colors=colors[:len(g)], line=dict(color="white", width=2)),
            textinfo="percent",
            textfont=dict(size=11, color="#0F172A"),
            hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:,.0f} personas<extra></extra>",
            showlegend=(i == 1),
        ), row=1, col=i)
    fig.update_layout(
        title_text=title,
        title_font_size=14,
        title_font_color="#0F172A",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter",
        font_color="#334155",
        height=300,
        margin=dict(t=56, b=16, l=8, r=8),
        showlegend=False,
        hoverlabel=dict(bgcolor="white", bordercolor="#CBD5E1",
                        font_size=12, font_family="Inter", font_color="#0F172A"),
    )
    fig.update_annotations(font_size=12, font_color="#64748B")
    return fig

# ─── CABECERA ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(120deg,#0F2D5E 0%,#1D4ED8 60%,#2563EB 100%);
            color:#fff;border-radius:14px;padding:26px 32px;margin-bottom:22px;
            box-shadow:0 4px 20px rgba(15,45,94,.18)">
  <div style="display:flex;align-items:center;gap:14px">
    <div>
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                  letter-spacing:1.2px;opacity:.65;margin-bottom:6px">
      </div>
      <h1 style="font-size:22px;margin:0;font-weight:800;letter-spacing:-.3px">
        Mercado Laboral de Jóvenes 14–28 años
      </h1>
      <p style="margin:6px 0 0 0;opacity:.75;font-size:12.5px;font-weight:400">
        Gran Encuesta Integrada de Hogares (GEIH) 2025 · DANE ·
        Área Metropolitana de Barranquilla
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI CARDS ────────────────────────────────────────────────────────────────
k = kpis()
col1,col2,col3,col4,col5,col6 = st.columns(6)

def kpi_card(col, label, value, sub, color):
    col.markdown(f"""
    <div class="kpi-box" style="--c:{color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(col1, "Tasa de Ocupación",      f"{k['TO']}%",           "de la PET tiene empleo",    BLUE)
kpi_card(col2, "Tasa de Desempleo",      f"{k['TD']}%",           "de la PEA busca trabajo",   RED)
kpi_card(col3, "Trabaja sin contrato",   f"{k['INF']}%",          "de los ocupados es informal", AMBER)
kpi_card(col4, "Ingreso Mediano",        f"${k['ING']/1e3:.0f}k", "COP al mes",                GREEN)
kpi_card(col5, "Educación Superior",     f"{k['EDU']}%",          "tiene técnico o más",       VIOLET)
kpi_card(col6, "Jóvenes con Potencial",  f"{k['JP']}%",           "NINI o trabajando informal", TEAL)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_socio, tab_ml, tab_edu, tab_salud, tab_busq, tab_jp, tab_calidad = st.tabs([
    "Sociodemográfico",
    "Mercado Laboral",
    "Educación",
    "Salud",
    "Búsqueda de Empleo",
    "Jóvenes con Potencial",
    "Calidad de Vida & IPM",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · SOCIODEMOGRÁFICO
# ══════════════════════════════════════════════════════════════════════════════
with tab_socio:
    st.markdown(f"<p class='section-header' style='--c:{VIOLET}'>Perfil Sociodemográfico</p>",
                unsafe_allow_html=True)

    # Sexo + Edad
    c1, c2 = st.columns(2)
    with c1:
        sx = wfreq("p3271", {1:"Hombre", 2:"Mujer"}, by_sex=False)
        fig = go.Figure(go.Pie(
            labels=sx["cat"], values=sx["n"], hole=0.62,
            marker=dict(colors=[BLUE, VIOLET], line=dict(color="white", width=3)),
            textinfo="percent+label", textfont_size=13,
            hovertemplate="<b>%{label}</b><br>%{percent}<br>%{value:,.0f} personas<extra></extra>",
        ))
        fig.update_layout(**layout_pie("Distribución por Sexo", height=300))
        chart(fig)

    with c2:
        edad = wfreq("grupo_edad", by_sex=True)
        fig = bar_sex(edad, x="cat", y="n", title="Grupos de Edad",
                      xlabel="Grupo de edad", ylabel="Número de personas")
        fig.update_xaxes(categoryorder="array",
                         categoryarray=["14-17","18-19","20-22","23-25","26-28"])
        chart(fig)

    # Estado civil + Etnia
    c3, c4 = st.columns(2)
    with c3:
        ec = wfreq("p6070", LAB["estado_civil"])
        fig = bar_sex(ec, h=True, title="Estado Civil",
                      xlabel="Número de personas", ylabel="")
        fig.update_layout(height=340)
        chart(fig)

    with c4:
        et = wfreq("p6080", LAB["etnia"], by_sex=True)
        fig = bar_sex(et, h=True, title="Pertenencia Étnica",
                      xlabel="Número de personas", ylabel="")
        fig.update_layout(height=340)
        chart(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · MERCADO LABORAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.markdown(f"<p class='section-header' style='--c:{BLUE}'>Mercado Laboral</p>",
                unsafe_allow_html=True)

    # Condición actividad (donuts) + Posición ocupacional
    c1, c2 = st.columns(2)
    with c1:
        fig = donut_pair("condicion",
                         {"Ocupado":"Ocupado","Desocupado":"Desocupado","Inactivo":"Inactivo"},
                         "Condición de Actividad",
                         [GREEN, AMBER, "#9CA3AF"])
        chart(fig)

    with c2:
        pos = wfreq("p6240", LAB["posicion"])
        fig = bar_sex(pos, h=True, pct=True, title="Posición Ocupacional",
                      xlabel="Porcentaje (%)", ylabel="")
        fig.update_layout(height=320)
        chart(fig)

    # Top 10 sectores
    st.markdown("---")
    ocu_mask = df["_ocu"]
    sec_g = (df[ocu_mask]
             .groupby("rama2d_r4")[W].sum()
             .round(0).astype(int)
             .reset_index()
             .sort_values(W, ascending=False)
             .head(10))
    sec_g["sector"] = sec_g["rama2d_r4"].map(LAB["sector"]).fillna(
        sec_g["rama2d_r4"].astype(str))
    fig = px.bar(sec_g, x=W, y="sector", orientation="h",
                 title="¿En qué sectores trabajan los jóvenes? · Top 10",
                 color=W,
                 color_continuous_scale=[[0, "#BFDBFE"], [0.5, "#3B82F6"], [1, "#1D4ED8"]],
                 labels={W: "Número de personas", "sector": ""},
                 text=sec_g[W].apply(lambda v: f"{v:,}"))
    fig.update_layout(**layout_chart(), height=360,
                      coloraxis_showscale=False, showlegend=False,
                      xaxis_title="Número de personas (promedio mensual)", yaxis_title="")
    fig.update_traces(marker_line_width=0, textposition="outside",
                      textfont=dict(size=10, color="#334155"))
    fig.update_yaxes(categoryorder="total ascending", tickfont=dict(size=11))
    fig.update_xaxes(showgrid=True, gridcolor="#E2E8F0")
    chart(fig)

    # Ingresos + Pensión
    c3, c4 = st.columns(2)
    with c3:
        ing_d = df[ocu_mask & (df["inglabo"] > 0) & df["inglabo"].notna()].copy()
        bins = [0,500000,800000,1160000,1423500,1800000,2500000,4000000,1e10]
        labs = ["<500k","500k-800k","800k-1.16M","1.16M-$1.42M","$1.42M-1.8M","1.8M-2.5M","2.5M-4M",">4M"]
        ing_d["rango"] = pd.cut(ing_d["inglabo"], bins=bins, labels=labs)
        ing_g = (ing_d.groupby(["sexo","rango"], observed=True)[W].sum()
                 .round(0).astype(int)
                 .reset_index().rename(columns={"rango":"Rango","sexo":"Sexo",W:"n"}))
        fig = px.bar(ing_g, x="Rango", y="n", color="Sexo",
                     barmode="group",
                     title="¿Cuánto ganan los jóvenes? · Ingreso laboral mensual",
                     color_discrete_map=SEX_COLORS)
        fig.update_layout(**layout_chart(), height=330,
                          xaxis_title="Rango salarial (COP) · Salario mínimo 2025 = $1.42M",
                          yaxis_title="Número de personas")
        fig.update_traces(marker_line_width=0, opacity=1)
        chart(fig)

    with c4:
        fig = donut_pair("p6590", LAB["pension"],
                         "Cotización a Pensión (asalariados)",
                         [GREEN, RED, AMBER])
        chart(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · EDUCACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with tab_edu:
    st.markdown(f"<p class='section-header' style='--c:{GREEN}'>Educación</p>",
                unsafe_allow_html=True)

    # Nivel educativo (horizontal, comparativo)
    niv = wfreq("p3042", LAB["nivel_edu"])
    orden_niv = list(LAB["nivel_edu"].values())
    fig = px.bar(niv, x="pct", y="cat", color="Sexo",
                 barmode="group", orientation="h",
                 title="Nivel Educativo Alcanzado",
                 color_discrete_map=SEX_COLORS,
                 category_orders={"cat": orden_niv})
    fig.update_layout(**layout_chart(), height=380,
                      xaxis_title="Porcentaje (%)", yaxis_title="Nivel educativo")
    fig.update_traces(marker_line_width=0, opacity=0.88)
    fig.update_xaxes(ticksuffix="%")
    chart(fig)

    # Asistencia + Alfabetismo
    c1, c2 = st.columns(2)
    with c1:
        fig = donut_pair("p6170", LAB["asistencia"],
                         "Asistencia Escolar Actual",
                         [GREEN, AMBER])
        chart(fig)

    with c2:
        alf = wfreq("p6160", LAB["alfabetismo"], by_sex=True)
        fig = bar_sex(alf, title="Alfabetismo (¿Sabe leer y escribir?)", pct=True,
                      xlabel="Condición", ylabel="Porcentaje (%)")
        chart(fig)

    # Nivel educativo por condición laboral
    st.markdown("---")
    niv_cond = (df[df["condicion"].notna()]
                .groupby(["condicion","p3042"])[W].sum()
                .reset_index())
    niv_cond["nivel"] = niv_cond["p3042"].map(LAB["nivel_edu"])
    niv_cond = niv_cond.dropna(subset=["nivel"])
    niv_cond["pct"] = niv_cond.groupby("condicion")[W].transform(lambda x: x/x.sum()*100).round(1)

    fig = px.bar(niv_cond, y="nivel", x="pct", color="condicion",
                 barmode="group", orientation="h",
                 title="Nivel Educativo por Condición Laboral",
                 color_discrete_map={"Ocupado":GREEN,"Desocupado":AMBER,"Inactivo":"#9CA3AF"},
                 category_orders={"nivel": list(reversed(orden_niv))},
                 labels={"condicion": "Condición"})
    fig.update_layout(**layout_chart(), height=420,
                      xaxis_title="Porcentaje (%)", yaxis_title="")
    fig.update_traces(marker_line_width=0, opacity=0.88)
    fig.update_xaxes(ticksuffix="%")
    chart(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · SALUD
# ══════════════════════════════════════════════════════════════════════════════
with tab_salud:
    st.markdown(f"<p class='section-header' style='--c:{PURPLE}'>Salud y Protección Social</p>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = donut_pair("p6090", LAB["afiliacion"],
                         "Afiliación al Sistema de Salud",
                         [PURPLE, RED])
        chart(fig)

    with c2:
        reg = wfreq("p6100", LAB["regimen"])
        fig = bar_sex(reg, h=True, pct=True, title="Régimen de Salud",
                      xlabel="Porcentaje (%)", ylabel="")
        fig.update_layout(height=260)
        chart(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · BÚSQUEDA DE EMPLEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_busq:
    st.markdown(f"<p class='section-header' style='--c:{AMBER}'>Búsqueda de Empleo</p>",
                unsafe_allow_html=True)

    des_n = int(round(df.loc[df["_des"], W].sum()))
    st.markdown(
        f"Gestiones realizadas en las últimas 4 semanas por los **{des_n:,}** "
        f"jóvenes desocupados para conseguir trabajo (promedio mensual)."
    )

    bq_rows = []
    for var, label in LAB["busqueda"].items():
        if var in df.columns:
            n = int(round(df.loc[df[var] == 1, W].sum()))
            bq_rows.append({"Método": label, "Personas": n})
    bq_df = pd.DataFrame(bq_rows).sort_values("Personas", ascending=True)

    bq_df["pct"] = (bq_df["Personas"] / des_n * 100).round(1)
    fig = px.bar(bq_df, x="Personas", y="Método", orientation="h",
                 title="¿Cómo buscan trabajo los jóvenes desocupados?",
                 color="Personas",
                 color_continuous_scale=[[0, "#FEF3C7"], [0.5, "#F59E0B"], [1, "#B45309"]],
                 text=bq_df["pct"].apply(lambda v: f"{v:.0f}%"))
    fig.update_layout(**layout_chart(), height=370,
                      coloraxis_showscale=False, showlegend=False,
                      xaxis_title="Número de jóvenes desocupados", yaxis_title="")
    fig.update_traces(marker_line_width=0, textposition="outside",
                      textfont=dict(size=11, color="#334155"))
    fig.update_yaxes(tickfont=dict(size=11))
    chart(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · JÓVENES CON POTENCIAL
# ══════════════════════════════════════════════════════════════════════════════
with tab_jp:
    st.markdown(f"<p class='section-header' style='--c:{TEAL}'>Jóvenes con Potencial</p>",
                unsafe_allow_html=True)

    st.markdown(
        f"Se consideran **Jóvenes con Potencial** quienes no estudian y no trabajan (NINI) "
        f"o se desempeñan en condición de informalidad. En total representan el "
        f"**{k['JP']}%** de la población en edad de trabajar - "
        f"**{k['JP_N']:,}** jóvenes en promedio mensual."
    )

    # Desglose: NINI / Informal / Resto — comparativo por sexo
    c1, c2 = st.columns(2)
    with c1:
        jp_rows = []
        for sex in ["Hombre", "Mujer"]:
            d = df[df["sexo"] == sex]
            pet_s = d.loc[d["_pet"], W].sum()
            nini_s = d.loc[d["_nini"] & d["_pet"], W].sum()
            inf_s  = d.loc[d["_inf"]  & d["_pet"], W].sum()
            resto  = pet_s - nini_s - inf_s
            for cat, val in [("NINI", nini_s), ("Informal", inf_s), ("Formal / Estudia", resto)]:
                jp_rows.append({"Sexo": sex, "Categoría": cat,
                                 "Número de personas": int(round(val)),
                                 "Porcentaje (%)": round(val / pet_s * 100, 1) if pet_s else 0})
        jp_df = pd.DataFrame(jp_rows)
        fig = px.bar(jp_df, x="Categoría", y="Porcentaje (%)", color="Sexo",
                     barmode="group", title="Distribución por categoría",
                     color_discrete_map=SEX_COLORS,
                     category_orders={"Categoría": ["NINI","Informal","Formal / Estudia"]})
        fig.update_layout(**layout_chart(), height=320,
                          xaxis_title="Categoría", yaxis_title="Porcentaje (%)")
        fig.update_traces(marker_line_width=0, opacity=0.88)
        fig.update_yaxes(ticksuffix="%")
        chart(fig)

    with c2:
        # % JP por grupo de edad y sexo
        jp_edad = []
        for g in ["14-17","18-19","20-22","23-25","26-28"]:
            for sex in ["Hombre","Mujer"]:
                d = df[(df["grupo_edad"] == g) & (df["sexo"] == sex)]
                pet_s = d.loc[d["_pet"], W].sum()
                jp_s  = d.loc[d["_jp"] & d["_pet"], W].sum()
                jp_edad.append({"Grupo de edad": g, "Sexo": sex,
                                 "Porcentaje (%)": round(jp_s / pet_s * 100, 1) if pet_s else 0})
        jp_edad_df = pd.DataFrame(jp_edad)
        fig = px.bar(jp_edad_df, x="Grupo de edad", y="Porcentaje (%)", color="Sexo",
                     barmode="group", title="% Jóvenes con Potencial por grupo de edad",
                     color_discrete_map=SEX_COLORS)
        fig.update_layout(**layout_chart(), height=320,
                          xaxis_title="Grupo de edad", yaxis_title="Porcentaje (%)")
        fig.update_traces(marker_line_width=0, opacity=0.88)
        fig.update_yaxes(ticksuffix="%")
        chart(fig)

    # Resumen cifras absolutas
    st.markdown("---")
    r1, r2, r3 = st.columns(3)
    r1.metric("NINI", f"{k['NINI_N']:,}", "No trabaja y no estudia")
    r2.metric("Informal", f"{k['INF_N']:,}", "Trabaja sin protección social")
    r3.metric("Total Jóvenes con Potencial", f"{k['JP_N']:,}", f"{k['JP']}% de la PET")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Fuente: DANE · Gran Encuesta Integrada de Hogares (GEIH) 2025 · "
    "Área Metropolitana de Barranquilla · Población: jóvenes 14–28 años · "
    "Indicadores expandidos con factor *fex_c18 ÷ 12* (promedio mensual)"
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 · CALIDAD DE VIDA & IPM
# ══════════════════════════════════════════════════════════════════════════════
TEAL2 = "#0D9488"
ORG   = "#EA580C"

with tab_calidad:

    st.markdown(f"""
    <div style="display:flex;gap:12px;margin-bottom:18px">
      <div style="background:linear-gradient(135deg,{TEAL2},#14B8A6);
                  border-radius:10px;padding:14px 20px;flex:1;color:#fff">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;opacity:.8">
          ECV 2026 · Barranquilla</div>
        <div style="font-size:13px;margin-top:4px">Bienestar subjetivo, acceso a servicios
          y conectividad · Jóvenes 14-28 años</div>
      </div>
      <div style="background:linear-gradient(135deg,{ORG},#F97316);
                  border-radius:10px;padding:14px 20px;flex:1;color:#fff">
        <div style="font-size:11px;font-weight:700;text-transform:uppercase;opacity:.8">
          IPM · Atlántico urbano</div>
        <div style="font-size:13px;margin-top:4px">Índice de Pobreza Multidimensional
          · 14 dimensiones de privación · Jóvenes 14-28 años</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # BLOQUE ECV
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"<p class='section-header' style='--c:{TEAL2}'>Encuesta de Calidad de Vida 2026 · Barranquilla</p>",
                unsafe_allow_html=True)

    def sat_kpi(col_n):
        vals = ecv.dropna(subset=[col_n])
        return np.average(vals[col_n], weights=vals["fex"]) if len(vals) else 0

    kp1 = sat_kpi("satisfecho_salud_n")
    kp2 = sat_kpi("satisfecho_vida_n")
    kp3 = sat_kpi("satisfecho_tiempolibre_n")

    ck1, ck2, ck3 = st.columns(3)
    for col_w, (avg, label, color) in zip(
        [ck1, ck2, ck3],
        [(kp1, "Satisfacción con la Salud", TEAL2),
         (kp2, "Satisfacción con la Vida",  BLUE),
         (kp3, "Satisfacción con el Tiempo Libre", PURPLE)]
    ):
        stars = "★" * round(avg / 2) + "☆" * (5 - round(avg / 2))
        col_w.markdown(f"""
        <div class="kpi-box" style="--c:{color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{avg:.1f}<span style="font-size:16px;color:#9CA3AF"> / 10</span></div>
          <div class="kpi-sub" style="font-size:14px;color:{color}">{stars}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Bienestar subjetivo por sexo + Suficiencia de ingresos ───────────────
    c1, c2 = st.columns([3, 2])

    with c1:
        sat_rows = []
        for dim, lab in [("satisfecho_salud_n", "Salud"),
                         ("satisfecho_vida_n",   "Vida"),
                         ("satisfecho_tiempolibre_n", "Tiempo libre")]:
            for sex in ["Hombre", "Mujer"]:
                d = ecv[ecv["Sexo"] == sex].dropna(subset=[dim])
                if len(d):
                    avg = np.average(d[dim], weights=d["fex"])
                    sat_rows.append({"Dimensión": lab, "Sexo": sex, "Promedio": round(avg, 2)})
        sat_df = pd.DataFrame(sat_rows)
        fig = px.bar(sat_df, x="Dimensión", y="Promedio", color="Sexo",
                     barmode="group",
                     title="Bienestar Subjetivo Promedio (escala 1-10)",
                     color_discrete_map=SEX_COLORS, text="Promedio")
        fig.update_traces(marker_line_width=0, opacity=0.9,
                          texttemplate="%{text:.1f}", textposition="outside")
        fig.update_layout(**layout_chart(), height=310,
                          xaxis_title="", yaxis_title="Puntuación promedio")
        fig.update_yaxes(range=[0, 11])
        chart(fig)

    with c2:
        ing_col = "ingresos suficientes"
        ing_g = (ecv.dropna(subset=[ing_col])
                    .groupby(ing_col)["fex"].sum().reset_index())

        def label_ing(s):
            sl = s.lower()
            if "m" in sl and "que" in sl: return "Cubre mas que lo minimo"
            if "no alcanzan" in sl:       return "No alcanza"
            return "Apenas alcanza"

        ing_g["label"] = ing_g[ing_col].apply(label_ing)
        ing_g["pct"]   = (ing_g["fex"] / ing_g["fex"].sum() * 100).round(1)
        ing_g["color"] = ing_g["label"].map({
            "Cubre mas que lo minimo": GREEN,
            "Apenas alcanza":          AMBER,
            "No alcanza":              RED,
        })
        ing_g = ing_g.sort_values("pct", ascending=True)

        fig = go.Figure()
        for _, row in ing_g.iterrows():
            fig.add_trace(go.Bar(
                x=[row["pct"]], y=[""],
                orientation="h",
                name=row["label"],
                marker_color=row["color"],
                text=f"  {row['label']}  {row['pct']:.0f}%",
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(size=11, color="white", family="Inter"),
                hovertemplate=f"<b>{row['label']}</b><br>{row['pct']:.1f}%<extra></extra>",
            ))
        fig.update_layout(
            barmode="stack",
            title_text="Suficiencia de ingresos del hogar",
            title_font_size=14, title_font_color="#0F172A",
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Inter",
            height=180,
            margin=dict(t=48, b=8, l=8, r=8),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, 100]),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            hoverlabel=dict(bgcolor="white", bordercolor="#CBD5E1",
                            font_size=12, font_family="Inter"),
        )
        chart(fig)

    # ── Acceso a servicios + Conectividad digital ─────────────────────────────
    c3, c4 = st.columns([3, 2])

    with c3:
        tiempo_cols = {
            "tiempo_paradero":  "Paradero",
            "tiempo_hospital":  "Hospital",
            "tiempo_parque":    "Parque",
            "tiempo_cai":       "CAI",
            "tiempo_cultural":  "Centro cultural",
            "tiempo_deportivo": "Instalación deportiva",
        }
        tiem_rows = []
        for col, label in tiempo_cols.items():
            d = ecv.dropna(subset=[col])
            if len(d):
                med = np.average(d[col], weights=d["fex"])
                tiem_rows.append({"Servicio": label, "Minutos": round(med, 1)})
        tiem_df = pd.DataFrame(tiem_rows).sort_values("Minutos", ascending=True)
        fig = px.bar(tiem_df, x="Minutos", y="Servicio", orientation="h",
                     title="Tiempo promedio de acceso a servicios (minutos)",
                     color="Minutos",
                     color_continuous_scale=[[0, "#CCFBF1"], [0.5, "#14B8A6"], [1, "#0F766E"]],
                     text="Minutos")
        fig.update_traces(marker_line_width=0, texttemplate="%{text:.0f} min",
                          textposition="outside")
        fig.update_layout(**layout_chart(), height=310,
                          coloraxis_showscale=False, showlegend=False,
                          xaxis_title="Minutos", yaxis_title="")
        chart(fig)

    with c4:
        def label_inet(s):
            sl = s.lower()
            if "todos" in sl:       return "Todos los dias"
            if "semana" in sl:      return "Varias veces / semana"
            if "mes" in sl:         return "Varias veces / mes"
            return "No usa internet"

        inet_g = (ecv.dropna(subset=["frecuencia_internet"])
                     .groupby("frecuencia_internet")["fex"].sum().reset_index())
        inet_g["label"] = inet_g["frecuencia_internet"].apply(label_inet)
        inet_g["pct"]   = (inet_g["fex"] / inet_g["fex"].sum() * 100).round(1)
        inet_colors = [BLUE, TEAL, AMBER, SLATE]
        inet_order  = ["Todos los dias", "Varias veces / semana",
                       "Varias veces / mes", "No usa internet"]
        inet_g["_ord"] = inet_g["label"].map({v: i for i, v in enumerate(inet_order)})
        inet_g = inet_g.sort_values("_ord").reset_index(drop=True)

        fig = px.bar(
            inet_g, x="pct", y="label", orientation="h",
            title="Uso de internet",
            color="label",
            color_discrete_sequence=inet_colors[:len(inet_g)],
            text=inet_g["pct"].apply(lambda v: f"{v:.0f}%"),
        )
        fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(size=11, color="#334155"))
        fig.update_layout(
            **layout_chart(), height=310,
            showlegend=False,
            xaxis_title="", yaxis_title="",
        )
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False,
                         range=[0, inet_g["pct"].max() * 1.25])
        chart(fig)

    # ── Estrato socioeconómico ─────────────────────────────────────────────────
    est_g = (ecv.dropna(subset=["estrato_vivienda"])
                .groupby(["estrato_vivienda", "Sexo"])["fex"].sum()
                .reset_index()
                .rename(columns={"estrato_vivienda": "Estrato", "fex": "n"}))
    est_g["pct"] = (est_g.groupby("Sexo")["n"]
                        .transform(lambda x: x / x.sum() * 100).round(1))
    fig = px.bar(est_g, x="Estrato", y="pct", color="Sexo", barmode="group",
                 title="Estrato Socioeconómico de la Vivienda",
                 color_discrete_map=SEX_COLORS,
                 text=est_g["pct"].apply(lambda v: f"{v:.1f}%"))
    fig.update_traces(marker_line_width=0, opacity=0.88, textposition="outside")
    fig.update_layout(**layout_chart(), height=280,
                      xaxis_title="Estrato", yaxis_title="Porcentaje (%)")
    fig.update_yaxes(ticksuffix="%")
    chart(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # BLOQUE IPM
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(f"<p class='section-header' style='--c:{ORG}'>Índice de Pobreza Multidimensional (IPM) · Atlántico urbano</p>",
                unsafe_allow_html=True)

    PRIV_COLS = {
        "empleo_formal":           "Sin empleo formal",
        "bajo logro educativo":    "Bajo logro educativo",
        "rezago escolar":          "Rezago escolar",
        "hacinamiento":            "Hacinamiento",
        "alcantarillado":          "Sin alcantarillado",
        "acueducto":               "Sin acueducto",
        "aseguramiento salud":     "Sin aseguramiento salud",
        "inasistencia escolar":    "Inasistencia escolar",
        "privacion_desempleo":     "Desempleo de larga duración",
        "pisos":                   "Pisos inadecuados",
        "privacion_analfabetismo": "Analfabetismo",
        "trabajo infantil":        "Trabajo infantil",
        "barreras salud":          "Barreras acceso a salud",
        "paredes":                 "Paredes inadecuadas",
    }

    total_w  = ipm["FEX_C"].sum()
    pobre_w  = ipm.loc[ipm["pobre"] == "Pobre", "FEX_C"].sum()
    ipm_avg  = np.average(ipm["IPM"], weights=ipm["FEX_C"]) if total_w else 0
    pobre_pct = round(pobre_w / total_w * 100, 1) if total_w else 0

    ip1, ip2, ip3 = st.columns(3)
    ip1.markdown(f"""
    <div class="kpi-box" style="--c:{ORG}">
      <div class="kpi-label">Pobreza Multidimensional</div>
      <div class="kpi-value">{pobre_pct}%</div>
      <div class="kpi-sub">jóvenes clasificados como pobres</div>
    </div>""", unsafe_allow_html=True)
    ip2.markdown(f"""
    <div class="kpi-box" style="--c:{AMBER}">
      <div class="kpi-label">IPM Promedio</div>
      <div class="kpi-value">{ipm_avg:.3f}</div>
      <div class="kpi-sub">0 = no pobre · 1 = máxima privación</div>
    </div>""", unsafe_allow_html=True)
    ip3.markdown(f"""
    <div class="kpi-box" style="--c:{BLUE}">
      <div class="kpi-label">Muestra analizada</div>
      <div class="kpi-value">{len(ipm):,}</div>
      <div class="kpi-sub">obs. · Atlántico cabecera · 14-28 años</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Incidencia de todas las privaciones ───────────────────────────────────
    priv_rows = []
    for col, label in PRIV_COLS.items():
        d = ipm.dropna(subset=[col])
        priv_w = d.loc[d[col] == "Privación", "FEX_C"].sum()
        tot_w  = d["FEX_C"].sum()
        priv_rows.append({"Privación": label,
                          "Pct": round(priv_w / tot_w * 100, 1) if tot_w else 0})
    priv_df = pd.DataFrame(priv_rows).sort_values("Pct", ascending=True)

    fig = px.bar(priv_df, x="Pct", y="Privación", orientation="h",
                 title="Incidencia de Privaciones IPM - Jóvenes 14-28 años (Atlántico urbano)",
                 color="Pct",
                 color_continuous_scale=[[0, "#FEF3C7"], [0.4, "#F59E0B"], [1, "#B45309"]],
                 text=priv_df["Pct"].apply(lambda v: f"{v:.1f}%"))
    fig.update_traces(marker_line_width=0, textposition="outside")
    fig.update_layout(**layout_chart(), height=420,
                      coloraxis_showscale=False, showlegend=False,
                      xaxis_title="% con privación", yaxis_title="")
    fig.update_xaxes(ticksuffix="%", range=[0, priv_df["Pct"].max() * 1.18])
    chart(fig)

    # ── Top 6 privaciones por sexo + Pobreza por edad ─────────────────────────
    c5, c6 = st.columns(2)

    with c5:
        top6 = priv_df.sort_values("Pct", ascending=False).head(6)["Privación"].tolist()
        top6_cols = {k: v for k, v in PRIV_COLS.items() if v in top6}
        priv_sex_rows = []
        for col, label in top6_cols.items():
            for sex in ["Hombre", "Mujer"]:
                d = ipm[ipm["Sexo"] == sex].dropna(subset=[col])
                priv_w = d.loc[d[col] == "Privación", "FEX_C"].sum()
                tot_w  = d["FEX_C"].sum()
                priv_sex_rows.append({"Privación": label, "Sexo": sex,
                                      "Pct": round(priv_w / tot_w * 100, 1) if tot_w else 0})
        psex_df = pd.DataFrame(priv_sex_rows)
        fig = px.bar(psex_df, x="Pct", y="Privación", color="Sexo",
                     barmode="group", orientation="h",
                     title="Top 6 Privaciones por Sexo",
                     color_discrete_map=SEX_COLORS,
                     text=psex_df["Pct"].apply(lambda v: f"{v:.0f}%"))
        fig.update_traces(marker_line_width=0, opacity=0.88, textposition="outside")
        fig.update_layout(**layout_chart(), height=340,
                          xaxis_title="% con privación", yaxis_title="")
        fig.update_xaxes(ticksuffix="%", range=[0, psex_df["Pct"].max() * 1.22])
        chart(fig)

    with c6:
        ipm_edad_rows = []
        for g in ["14-17", "18-23", "24-28"]:
            for sex in ["Hombre", "Mujer"]:
                d = ipm[(ipm["edad"] == g) & (ipm["Sexo"] == sex)]
                if len(d):
                    pw = d.loc[d["pobre"] == "Pobre", "FEX_C"].sum()
                    tw = d["FEX_C"].sum()
                    ipm_edad_rows.append({"Grupo de edad": g, "Sexo": sex,
                                          "% Pobre": round(pw / tw * 100, 1) if tw else 0})
        ipm_edad_df = pd.DataFrame(ipm_edad_rows)
        fig = px.bar(ipm_edad_df, x="Grupo de edad", y="% Pobre", color="Sexo",
                     barmode="group",
                     title="Pobreza Multidimensional por Grupo de Edad",
                     color_discrete_map=SEX_COLORS,
                     text=ipm_edad_df["% Pobre"].apply(lambda v: f"{v:.1f}%"))
        fig.update_traces(marker_line_width=0, opacity=0.88, textposition="outside")
        fig.update_layout(**layout_chart(), height=340,
                          xaxis_title="Grupo de edad", yaxis_title="% pobres")
        fig.update_yaxes(ticksuffix="%", range=[0, ipm_edad_df["% Pobre"].max() * 1.25])
        chart(fig)

    # ── Distribución del índice IPM ────────────────────────────────────────────
    fig = go.Figure()
    for sex, color in [("Hombre", BLUE), ("Mujer", VIOLET)]:
        d = ipm[ipm["Sexo"] == sex]["IPM"].dropna()
        fig.add_trace(go.Histogram(
            x=d, name=sex, marker_color=color, opacity=0.72,
            xbins=dict(start=0, end=1, size=0.05), histnorm="percent",
        ))
    fig.update_layout(
        **layout_chart(), height=280, barmode="overlay",
        title_text="Distribución del Índice IPM por Sexo (0 = no pobre · 1 = máx. privación)",
        xaxis_title="Valor del IPM", yaxis_title="% de personas",
    )
    fig.update_yaxes(ticksuffix="%")
    chart(fig)

    st.caption(
        "ECV 2026: Encuesta de Calidad de Vida · DANE · Barranquilla · Jóvenes 14-28 años · "
        "Ponderado con FEX_C. | "
        "IPM: Índice de Pobreza Multidimensional · DANE · Atlántico cabecera · "
        "Jóvenes 14-28 años · Ponderado con FEX_C."
    )
