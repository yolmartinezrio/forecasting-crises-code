"""
=============================================================================
config/settings.py
=============================================================================
Configuración global del proyecto.
Centraliza todas las rutas, parámetros y constantes compartidas.
Cualquier otro módulo importa desde aquí; no hay "magic numbers" dispersos.
=============================================================================
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# RUTAS DEL PROYECTO
# ---------------------------------------------------------------------------
ROOT_DIR    = Path(__file__).resolve().parents[1]   # financial_crises/
DATA_DIR    = ROOT_DIR / "data_input"
OUTPUT_DIR  = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_OUT    = OUTPUT_DIR / "data"

# Crear carpetas de salida si no existen
for d in [OUTPUT_DIR, FIGURES_DIR, DATA_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ARCHIVOS DE DATOS FUENTE
# ---------------------------------------------------------------------------
JST_FILE = DATA_DIR / "JSTdatasetR6.xlsx"
LV_FILE  = DATA_DIR / "SYSTEMIC_BANKING_CRISES_DATABASE_2018.xlsx"
LV_SHEET = "Crisis Years"

# ---------------------------------------------------------------------------
# PARÁMETROS DEL PANEL
# ---------------------------------------------------------------------------
YEAR_MIN         = 1870
YEAR_MAX         = 2020
CRISIS_DURATION  = 3          # años que se considera activa una crisis LV
FORECAST_HORIZONS = [1, 2, 3] # horizontes de predicción (años adelante)
HP_LAMBDA        = 400        # parámetro suavizado filtro Hodrick-Prescott (anual)

# ---------------------------------------------------------------------------
# MAPEO DE NOMBRES: Laeven-Valencia → JST
# ---------------------------------------------------------------------------
LV_TO_JST: dict[str, str] = {
    "United Kingdom": "UK",
    "United States":  "USA",
    "Australia":      "Australia",
    "Belgium":        "Belgium",
    "Canada":         "Canada",
    "Denmark":        "Denmark",
    "Finland":        "Finland",
    "France":         "France",
    "Germany":        "Germany",
    "Ireland":        "Ireland",
    "Italy":          "Italy",
    "Japan":          "Japan",
    "Netherlands":    "Netherlands",
    "Norway":         "Norway",
    "Portugal":       "Portugal",
    "Spain":          "Spain",
    "Sweden":         "Sweden",
    "Switzerland":    "Switzerland",
}

JST_COUNTRIES: list[str] = list(LV_TO_JST.values())

# ---------------------------------------------------------------------------
# VARIABLES PREDICTORAS — agrupadas por bloque temático
# ---------------------------------------------------------------------------
# Cada grupo: lista de columnas JST base (antes de derivar nuevas variables).
# La clave es el nombre del bloque; el valor es un dict con:
#   "vars"  : columnas originales del JST que entran en el grupo
#   "doc"   : justificación económica (usada en el diccionario de variables)
# ---------------------------------------------------------------------------
FEATURE_GROUPS: dict[str, dict] = {

    "CRÉDITO_BANCARIO": {
        "vars": ["tloans", "tmort"],
        "doc": (
            "Préstamos totales y préstamos hipotecarios del sector bancario, "
            "ambos normalizados por PIB. El crecimiento del crédito bancario es, "
            "según Schularick & Taylor (2012), el mejor predictor individual de "
            "crisis bancarias sistémicas en una muestra de 150 años. Un auge "
            "crediticio sostenido por encima de la tendencia histórica aumenta "
            "la probabilidad de crisis con horizonte 3-5 años. La brecha "
            "crédito/PIB respecto a su tendencia HP es el indicador "
            "macroprudencial de referencia del BIS (Drehmann & Juselius, 2014)."
        ),
    },

    "ESTRUCTURA_BANCARIA": {
        "vars": ["lev", "ltd", "noncore"],
        "doc": (
            "Indicadores de la fragilidad estructural del balance bancario. "
            "El apalancamiento (lev) captura el riesgo de insolvencia ante "
            "shocks de valoración de activos. La ratio préstamos/depósitos (ltd) "
            "señala la dependencia de financiación mayorista: cuando supera 1 el "
            "banco financia crédito con deuda de mercado, más volátil que los "
            "depósitos minoristas. Los pasivos no-depósito (noncore) miden la "
            "exposición a mercados de financiación que se cierran abruptamente "
            "en episodios de estrés sistémico."
        ),
    },

    "PRECIOS_ACTIVOS": {
        "vars": ["hpnom", "housing_capgain", "eq_capgain"],
        "doc": (
            "Precios nominales de vivienda y plusvalías anuales del mercado "
            "inmobiliario y bursátil. Borio & Lowe (2002) documentan que la "
            "combinación de auge crediticio y auge de precios de activos "
            "(especialmente inmobiliarios) es el predictor más robusto de "
            "crisis bancarias. La caída brusca de estos precios puede "
            "desencadenar el episodio de crisis tras el auge."
        ),
    },

    "TIPOS_INTERÉS": {
        "vars": ["stir", "ltrate", "bill_rate"],
        "doc": (
            "Tipos de interés a corto plazo (mercado monetario), largo plazo "
            "(bono soberano) y letras del Tesoro. La pendiente de la curva de "
            "tipos (diferencial largo-corto) es un indicador adelantado del "
            "ciclo económico y financiero. Tipos reales muy bajos prolongados "
            "alimentan la toma de riesgo excesiva. El diferencial de tipos y "
            "el coste del servicio de deuda son señales clave de vulnerabilidad "
            "(Drehmann & Juselius, 2014)."
        ),
    },

    "MACROECONOMÍA": {
        "vars": ["gdp", "iy", "ca", "cpi", "debtgdp", "money"],
        "doc": (
            "Variables macroeconómicas fundamentales. La ratio inversión/PIB "
            "(iy) captura booms de acumulación de capital a menudo financiados "
            "con crédito. El saldo de cuenta corriente (ca) señala "
            "vulnerabilidades externas: déficits persistentes financiados con "
            "deuda exterior preceden muchas crisis. La inflación (cpi) entra "
            "como proxy del entorno de política monetaria. La deuda pública/PIB "
            "(debtgdp) y el agregado M2 (money) completan el cuadro macro."
        ),
    },

    "RETORNOS_FINANCIEROS": {
        "vars": ["bond_rate", "housing_tr", "eq_tr"],
        "doc": (
            "Retornos realizados de bonos soberanos, activos inmobiliarios y "
            "renta variable. Retornos inmobiliarios muy elevados sostenidos son "
            "señal de burbuja; su colapso desencadena crisis bancarias a través "
            "del canal colateral. El yield del bono soberano incorpora la "
            "percepción de riesgo país."
        ),
    },
}

# Variables que se imputan por forward/backward-fill dentro de cada país
IMPUTE_FFILL: list[str] = [
    "hpnom", "housing_capgain", "eq_capgain",
    "housing_tr", "eq_tr", "bond_tr",
    "unemp", "lev", "ltd", "noncore", "tmort", "thh", "tbus",
]

# ---------------------------------------------------------------------------
# DICCIONARIO COMPLETO DE VARIABLES JST ORIGINALES
# Formato: columna → (descripción, unidad, relevancia)
# relevancia: "ALTA" | "MEDIA" | "BAJA" | "IDENTIF"
# ---------------------------------------------------------------------------
JST_VARIABLE_DICT: dict[str, tuple[str, str, str]] = {
    "year"        : ("Año calendario",                                  "año",          "IDENTIF"),
    "country"     : ("Nombre del país",                                 "texto",        "IDENTIF"),
    "iso"         : ("Código ISO-3 del país",                           "código",       "IDENTIF"),
    "ifs"         : ("Código IFS del FMI",                              "código",       "IDENTIF"),
    "pop"         : ("Población total (miles)",                         "miles hab.",   "BAJA"),
    "rgdpmad"     : ("PIB real per cápita, base Maddison",              "USD 1990",     "MEDIA"),
    "rgdpbarro"   : ("PIB real per cápita, base Barro",                 "USD",          "MEDIA"),
    "rconsbarro"  : ("Consumo real per cápita, base Barro",             "USD",          "MEDIA"),
    "gdp"         : ("PIB nominal total",                               "moneda local", "ALTA"),
    "iy"          : ("Inversión / PIB (ratio)",                         "fracción",     "ALTA"),
    "cpi"         : ("Índice de precios al consumo",                    "índice",       "MEDIA"),
    "ca"          : ("Saldo cuenta corriente / PIB",                    "fracción",     "ALTA"),
    "imports"     : ("Importaciones / PIB",                             "fracción",     "MEDIA"),
    "exports"     : ("Exportaciones / PIB",                             "fracción",     "MEDIA"),
    "narrowm"     : ("M1 — dinero estrecho / PIB",                      "fracción",     "ALTA"),
    "money"       : ("M2/M3 — dinero amplio / PIB",                     "fracción",     "ALTA"),
    "stir"        : ("Tipo de interés a corto plazo (nominal)",         "porcentaje",   "ALTA"),
    "ltrate"      : ("Tipo de interés a largo plazo (nominal)",         "porcentaje",   "ALTA"),
    "hpnom"       : ("Índice de precios nominales de vivienda",         "índice",       "ALTA"),
    "unemp"       : ("Tasa de desempleo",                               "porcentaje",   "MEDIA"),
    "wage"        : ("Salario real per cápita (índice)",                "índice",       "MEDIA"),
    "debtgdp"     : ("Deuda pública bruta / PIB",                       "fracción",     "ALTA"),
    "revenue"     : ("Ingresos fiscales / PIB",                         "fracción",     "MEDIA"),
    "expenditure" : ("Gasto público / PIB",                             "fracción",     "MEDIA"),
    "xrusd"       : ("Tipo de cambio frente al USD",                    "USD/moneda",   "MEDIA"),
    "tloans"      : ("Préstamos totales del sector bancario / PIB",     "fracción",     "ALTA"),
    "tmort"       : ("Préstamos hipotecarios totales / PIB",            "fracción",     "ALTA"),
    "thh"         : ("Préstamos a hogares / PIB",                       "fracción",     "ALTA"),
    "tbus"        : ("Préstamos a empresas / PIB",                      "fracción",     "ALTA"),
    "bdebt"       : ("Deuda exterior del sector bancario / PIB",        "fracción",     "ALTA"),
    "lev"         : ("Apalancamiento bancario (activos/fondos propios)","ratio",        "ALTA"),
    "ltd"         : ("Ratio préstamos/depósitos",                       "ratio",        "ALTA"),
    "noncore"     : ("Pasivos no-depósito del banco / activos",         "fracción",     "ALTA"),
    "crisisJST"   : ("Indicador de crisis bancaria (cronología JST)",   "binario",      "IDENTIF"),
    "peg"         : ("Régimen de tipo de cambio fijo",                  "binario",      "MEDIA"),
    "peg_strict"  : ("Tipo de cambio fijo estricto",                    "binario",      "BAJA"),
    "eq_tr"       : ("Retorno total de renta variable",                 "fracción",     "MEDIA"),
    "housing_tr"  : ("Retorno total del mercado inmobiliario",          "fracción",     "ALTA"),
    "bond_tr"     : ("Retorno total de bonos soberanos",                "fracción",     "MEDIA"),
    "bill_rate"   : ("Tipo de interés de letras del tesoro",            "fracción",     "ALTA"),
    "housing_capgain": ("Plusvalía inmobiliaria anual",                 "fracción",     "ALTA"),
    "housing_rent_rtn":("Rentabilidad por alquiler inmobiliario",       "fracción",     "MEDIA"),
    "eq_capgain"  : ("Plusvalía bursátil anual",                        "fracción",     "MEDIA"),
    "eq_dp"       : ("Dividend yield del mercado de acciones",          "fracción",     "MEDIA"),
    "bond_rate"   : ("Tipo de interés (yield) del bono soberano",       "porcentaje",   "ALTA"),
}

# Variables derivadas generadas durante el preprocesamiento
DERIVED_VARIABLE_DICT: dict[str, tuple[str, str, str]] = {
    "tloans_gap"   : ("Brecha crédito/PIB respecto a tendencia HP (λ=400) — "
                      "indicador macroprudencial BIS",               "pp. de PIB",   "ALTA"),
    "hpnom_gap"    : ("Brecha precios nominales vivienda respecto a tendencia HP",
                                                                     "pp.",          "ALTA"),
    "term_spread"  : ("Diferencial tipos largo plazo – corto plazo", "pp.",          "ALTA"),
    "tloans_growth": ("Tasa de crecimiento anual del crédito bancario total/PIB",
                                                                     "fracción",     "ALTA"),
}

# ---------------------------------------------------------------------------
# ESTILO VISUAL GLOBAL
# ---------------------------------------------------------------------------
PLOT_STYLE   = "whitegrid"
PLOT_PALETTE = "muted"
PLOT_FONT_SCALE = 1.05
COLORS = {
    "crisis"    : "#C0392B",
    "no_crisis" : "#2980B9",
    "accent"    : "#27AE60",
    "neutral"   : "#7F8C8D",
}
FIGURE_DPI = 140
