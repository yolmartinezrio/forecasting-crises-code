"""
=============================================================================
models/logit_panel.py
=============================================================================
Clase LogitPanel — modelo logit de panel con efectos fijos de país.

Este modelo constituye la línea base econométrica (baseline) del proyecto.
Representa el estado del arte en la práctica supervisora institucional
(Borio & Lowe, 2002; Lo Duca et al., 2017) y sirve como referencia
contra la que se miden las mejoras aportadas por los modelos generativos.

Arquitectura
------------
  Logit(crisis_{i,t+h}) = α_i + β · X_{i,t}

donde:
  α_i   = efectos fijos de país (dummies, absorbidos por el intercept
           cuando se usa one-hot encoding o eliminados con within-transform)
  X_{i,t} = vector de predictores estandarizados
  h     = horizonte de predicción ∈ {1, 2, 3}

Implementación
--------------
Los efectos fijos de país se modelan mediante variables dummy (within
estimator), eliminando la necesidad de librerías de econometría de panel
y haciendo el modelo compatible con el pipeline scikit-learn estándar que
se usa en la evaluación walk-forward.

La regularización L2 (ridge) controla el sobreajuste en el conjunto de
datos pequeño y desequilibrado. El parámetro C (inverso de λ) se
selecciona por validación cruzada temporal dentro del conjunto de
entrenamiento de cada ventana walk-forward.

Referencia
----------
Bluwstein et al. (2020). Credit growth, the yield curve and financial
crisis prediction. Bank of England Working Paper, 848.

Lo Duca et al. (2017). A new database for financial crises in European
countries. ECB Occasional Paper, 194.
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config.settings import FORECAST_HORIZONS, JST_COUNTRIES


# ---------------------------------------------------------------------------
# Configuración del modelo
# ---------------------------------------------------------------------------

# Features seleccionadas para el logit de referencia.
# Criterio: variables con < 15 % de NaN, representativas de cada bloque
# temático y con poder predictivo documentado en la literatura.
LOGIT_FEATURES: list[str] = [
    # Crédito bancario (★ predictores principales ★)
    "tloans_gap",      # brecha crédito/PIB vs tendencia HP — indicador BIS
    "tloans_growth",   # crecimiento anual del crédito bancario
    "tloans",          # nivel crédito/PIB — Schularick & Taylor (2012)
    # Estructura bancaria
    "lev",             # apalancamiento bancario
    "ltd",             # ratio préstamos/depósitos
    "noncore",         # dependencia de financiación mayorista
    # Precios de activos
    "hpnom_gap",       # brecha precios vivienda vs tendencia HP
    "housing_capgain", # plusvalía inmobiliaria anual
    "eq_capgain",      # plusvalía bursátil anual
    # Tipos de interés
    "term_spread",     # pendiente de la curva de tipos (LP - CP)
    "stir",            # tipo de interés a corto plazo
    "ltrate",          # tipo de interés a largo plazo
    # Macroeconomía
    "ca",              # saldo cuenta corriente / PIB
    "debtgdp",         # deuda pública / PIB
    "iy",              # inversión / PIB
    "money",           # M2/M3 / PIB
    # Retornos financieros
    "housing_tr",      # retorno total del mercado inmobiliario
    "eq_tr",           # retorno total renta variable
]

# Valores de C (inverso de la regularización L2) a explorar en CV
C_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0]

# Peso de clase positiva relativo a la negativa en la función de pérdida.
# Equivale aproximadamente al ratio de desbalance 1:47, lo que penaliza
# mucho más los falsos negativos (crisis no detectadas) que los falsos
# positivos (falsas alarmas), en línea con las preferencias del supervisor
# macroprudencial (Alessi & Detken, 2011).
CLASS_WEIGHT: str = "balanced"


class LogitPanel:
    """
    Modelo logit de panel con efectos fijos de país para la predicción
    binaria de crisis bancarias sistémicas.

    Parámetros
    ----------
    horizon : int
        Horizonte de predicción h ∈ {1, 2, 3}. Determina qué columna
        target se usa (crisis_h1, crisis_h2 o crisis_h3).
    features : list[str] | None
        Lista de nombres de features a usar. Si es None, se usa
        LOGIT_FEATURES definido en este módulo.
    C : float
        Parámetro de regularización L2 (inverso de λ). Valores menores
        implican mayor regularización. Se selecciona por CV temporal
        en WalkForwardEvaluator.
    country_fe : bool
        Si True, añade dummies de país (efectos fijos). Default True.
    random_state : int
        Semilla para reproducibilidad.

    Atributos (tras llamar a fit)
    -----------------------------
    pipeline_ : sklearn.pipeline.Pipeline
        Pipeline ajustado: StandardScaler → LogisticRegression.
    feature_names_in_ : list[str]
        Lista de features (incluyendo dummies de país si country_fe=True)
        en el orden en que el modelo las recibe.
    coef_df_ : pd.DataFrame
        DataFrame con los coeficientes estimados y sus odds-ratios.
    n_train_ : int
        Número de observaciones de entrenamiento usadas en el último fit.

    Uso típico
    ----------
    >>> model = LogitPanel(horizon=1)
    >>> model.fit(X_train, y_train, countries_train)
    >>> probs = model.predict_proba(X_test, countries_test)
    """

    def __init__(
        self,
        horizon:      int        = 1,
        features:     list[str] | None = None,
        C:            float      = 0.1,
        country_fe:   bool       = True,
        random_state: int        = 42,
    ) -> None:
        if horizon not in FORECAST_HORIZONS:
            raise ValueError(
                f"horizon debe ser uno de {FORECAST_HORIZONS}, "
                f"recibido: {horizon}"
            )
        self.horizon      = horizon
        self.features     = features if features is not None else LOGIT_FEATURES
        self.C            = C
        self.country_fe   = country_fe
        self.random_state = random_state

        # Atributos que se pueblan tras fit()
        self.pipeline_:         Pipeline | None     = None
        self.feature_names_in_: list[str]           = []
        self.coef_df_:          pd.DataFrame | None = None
        self.n_train_:          int                 = 0
        self._countries_train:  list[str]           = []

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    @property
    def target_col(self) -> str:
        """Nombre de la columna objetivo según el horizonte."""
        return f"crisis_h{self.horizon}"

    def fit(
        self,
        X:         pd.DataFrame,
        y:         pd.Series,
        countries: pd.Series | None = None,
    ) -> "LogitPanel":
        """
        Ajusta el modelo sobre el conjunto de entrenamiento.

        Parámetros
        ----------
        X : pd.DataFrame
            Features (sin dummies de país). Se esperan las columnas
            listadas en self.features; las ausentes se ignoran.
        y : pd.Series
            Variable dependiente binaria (0/1).
        countries : pd.Series | None
            Códigos de país para construir los efectos fijos. Necesario
            si country_fe=True.

        Retorna
        -------
        self
        """
        X_model, feat_names = self._prepare_X(X, countries, is_train=True)
        self.feature_names_in_ = feat_names
        self.n_train_          = len(X_model)

        self.pipeline_ = Pipeline([
            ("scaler", StandardScaler()),
            ("logit",  LogisticRegression(
                C=self.C,
                solver="lbfgs",
                max_iter=2000,
                class_weight=CLASS_WEIGHT,
                random_state=self.random_state,
            )),
        ])
        self.pipeline_.fit(X_model, y)
        self._build_coef_df(feat_names)
        return self

    def predict_proba(
        self,
        X:         pd.DataFrame,
        countries: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Devuelve la probabilidad estimada de crisis para cada observación.

        Retorna
        -------
        np.ndarray de shape (n,) con P(crisis=1 | X).
        """
        self._check_fitted()
        X_model, _ = self._prepare_X(X, countries)
        return self.pipeline_.predict_proba(X_model)[:, 1]

    def predict(
        self,
        X:         pd.DataFrame,
        countries: pd.Series | None = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Devuelve la clase predicha (0/1) según un umbral de probabilidad.

        Parámetros
        ----------
        threshold : float
            Umbral de decisión. El umbral óptimo se determina en
            WalkForwardEvaluator minimizando la función de pérdida
            asimétrica de Alessi & Detken (2011).
        """
        return (self.predict_proba(X, countries) >= threshold).astype(int)

    def get_coef_df(self) -> pd.DataFrame:
        """
        Retorna un DataFrame con los coeficientes estimados,
        sus odds-ratios y un indicador de dirección del efecto.

        Columnas
        --------
        feature     : nombre de la variable
        coef        : coeficiente logit (log-odds)
        odds_ratio  : exp(coef)
        direction   : '↑ riesgo' si coef > 0, '↓ riesgo' si coef < 0
        """
        self._check_fitted()
        return self.coef_df_.copy()

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _prepare_X(
        self,
        X:         pd.DataFrame,
        countries: pd.Series | None,
        is_train:  bool = False,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Construye la matriz de diseño:
          1. Selecciona las features disponibles.
          2. Imputa NaN con la mediana de cada columna (mediana robusta).
          3. Añade dummies de país si country_fe=True.
             En inferencia (is_train=False), alinea las dummies con las
             columnas vistas en entrenamiento para evitar errores de
             dimensionalidad cuando el conjunto de prueba no contiene
             todos los países del entrenamiento.
        """
        # Seleccionar features disponibles en X
        avail = [f for f in self.features if f in X.columns]
        X_sel = X[avail].copy().reset_index(drop=True)

        # Imputar NaN con mediana por columna (imputación robusta)
        for col in X_sel.columns:
            if X_sel[col].isna().any():
                med = X_sel[col].median()
                X_sel[col] = X_sel[col].fillna(med if not pd.isna(med) else 0.)

        feat_names = list(avail)

        # Efectos fijos de país (dummies)
        if self.country_fe and countries is not None:
            dummies = pd.get_dummies(
                countries.reset_index(drop=True),
                prefix="fe", drop_first=True, dtype=float,
            )
            if is_train:
                # Guardar columnas de dummies para alinear en predict
                self._dummy_cols_ = list(dummies.columns)
            else:
                # Alinear con las columnas del entrenamiento
                if hasattr(self, "_dummy_cols_"):
                    dummies = dummies.reindex(
                        columns=self._dummy_cols_, fill_value=0.0
                    )
            X_sel = pd.concat([X_sel, dummies], axis=1)
            feat_names = feat_names + list(dummies.columns)

        return X_sel.values, feat_names

    def _build_coef_df(self, feat_names: list[str]) -> None:
        """Construye el DataFrame de coeficientes tras el ajuste."""
        logit = self.pipeline_.named_steps["logit"]
        coefs = logit.coef_[0]
        self.coef_df_ = pd.DataFrame({
            "feature"    : feat_names,
            "coef"       : coefs,
            "odds_ratio" : np.exp(coefs),
            "direction"  : ["↑ riesgo" if c > 0 else "↓ riesgo"
                            for c in coefs],
        }).sort_values("coef", key=abs, ascending=False).reset_index(drop=True)

    def _check_fitted(self) -> None:
        if self.pipeline_ is None:
            raise RuntimeError(
                "El modelo no ha sido ajustado. Llama a fit() primero."
            )

    # ------------------------------------------------------------------
    # Representación
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "ajustado" if self.pipeline_ is not None else "sin ajustar"
        return (
            f"LogitPanel(horizon={self.horizon}, C={self.C}, "
            f"country_fe={self.country_fe}, estado={status})"
        )
