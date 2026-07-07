"""
=============================================================================
evaluation/walkforward.py
=============================================================================
Clase WalkForwardEvaluator — protocolo de evaluación walk-forward con
ventana expansiva.

El protocolo walk-forward es el estándar metodológico para la evaluación
de modelos de predicción de crisis financieras (Bluwstein et al., 2020;
Fouliard et al., 2021). Su principio es:

  Para cada año de evaluación t ∈ [t_eval_start, t_eval_end]:
    - Entrenar el modelo con todas las observaciones hasta t-1
    - Predecir en t (o en t+h para el horizonte h)
    - Acumular predicciones y evaluar fuera de muestra

Esta aproximación «expanding window» garantiza:
  1. Ausencia total de filtración de información futura (data leakage):
     el modelo nunca ve datos del período que predice.
  2. Simulación realista del proceso de predicción en tiempo real:
     el modelo se reentrena periódicamente con información actualizada.
  3. Evaluación fuera de muestra genuina en el período post-1990,
     que incluye la crisis bancaria nórdica (1991), la crisis japonesa
     (1997) y la Gran Crisis Financiera Global (2008).

El período de evaluación (post-1990) se elige para maximizar la
cobertura de datos en el período de entrenamiento y garantizar que
el conjunto de evaluación incluye episodios de crisis genuinos que
los modelos no han visto durante el entrenamiento inicial.

Referencias
-----------
Bluwstein et al. (2020). Bank of England Working Paper, 848.
Fouliard et al. (2021). BIS Working Papers, 926.
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

from config.settings import FORECAST_HORIZONS, OUTPUT_DIR
from evaluation.metrics import EvaluationMetrics, EvaluationReport

# ---------------------------------------------------------------------------
# Parámetros del protocolo
# ---------------------------------------------------------------------------

EVAL_START_YEAR = 1990   # primer año del período de evaluación OOS
EVAL_END_YEAR   = 2018   # último año del período de evaluación OOS
                          # (2019-2020 excluidos: datos COVID)
MIN_TRAIN_POS   = 5      # mínimo de positivos en entrenamiento para ajustar


class WalkForwardEvaluator:
    """
    Implementa el protocolo de evaluación walk-forward con ventana
    expansiva para cualquier modelo que implemente la interfaz:
        model.fit(X, y, countries)
        model.predict_proba(X, countries) → np.ndarray

    Parámetros
    ----------
    panel : pd.DataFrame
        Panel maestro completo (output de DataPreprocessor).
    feature_cols : list[str]
        Lista de columnas predictoras a pasar a los modelos.
    horizons : list[int]
        Horizontes de predicción a evaluar. Default: [1, 2, 3].
    eval_start : int
        Primer año del período de evaluación OOS.
    eval_end : int
        Último año del período de evaluación OOS.
    mu : float
        Parámetro de la función de pérdida asimétrica.
    n_bootstrap : int
        Réplicas bootstrap para los IC.
    random_state : int
        Semilla de aleatoriedad.

    Uso típico
    ----------
    >>> evaluator = WalkForwardEvaluator(panel, feature_cols)
    >>> results = evaluator.run(
    ...     model_factory=lambda: LogitPanel(horizon=h),
    ...     model_name="LogitPanel",
    ... )
    >>> evaluator.print_summary(results)
    >>> evaluator.save_results(results)
    """

    def __init__(
        self,
        panel:        pd.DataFrame,
        feature_cols: list[str],
        horizons:     list[int] = None,
        eval_start:   int       = EVAL_START_YEAR,
        eval_end:     int       = EVAL_END_YEAR,
        mu:           float     = 0.75,
        n_bootstrap:  int       = 1000,
        random_state: int       = 42,
    ) -> None:
        self.panel        = panel.copy()
        self.feature_cols = feature_cols
        self.horizons     = horizons if horizons is not None else FORECAST_HORIZONS
        self.eval_start   = eval_start
        self.eval_end     = eval_end
        self.metrics_calc = EvaluationMetrics(
            mu=mu, n_bootstrap=n_bootstrap, random_state=random_state
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(
        self,
        model_factory: Callable,
        model_name:    str,
    ) -> list[EvaluationReport]:
        """
        Ejecuta el protocolo walk-forward para todos los horizontes.

        Parámetros
        ----------
        model_factory : callable
            Función sin argumentos que devuelve una instancia nueva del
            modelo a evaluar. Se llama una vez por (horizonte, año), lo
            que garantiza que cada ventana arranca desde cero.
            Ejemplo: lambda: LogitPanel(horizon=h, C=0.1)
        model_name : str
            Nombre del modelo para los informes.

        Retorna
        -------
        list[EvaluationReport]
            Un informe por horizonte.
        """
        print(f"\n[WalkForwardEvaluator] Evaluando: {model_name}")
        print(f"  Período OOS: {self.eval_start}–{self.eval_end}")
        print(f"  Horizontes : {self.horizons}")

        reports = []
        for h in self.horizons:
            print(f"\n  ── Horizonte h={h} ─────────────────────────")
            target_col = f"crisis_h{h}"
            if target_col not in self.panel.columns:
                print(f"  ✗ Columna {target_col} no encontrada. Omitiendo.")
                continue

            y_true_all, y_prob_all, years_all = self._walk(
                h, target_col, model_factory
            )

            if len(y_true_all) == 0 or y_true_all.sum() == 0:
                print(f"  ✗ Sin positivos en OOS para h={h}. Omitiendo.")
                continue

            report = self.metrics_calc.evaluate(
                y_true     = y_true_all,
                y_prob     = y_prob_all,
                model_name = model_name,
                horizon    = h,
                years      = years_all,
            )
            print(report.summary())
            reports.append(report)

        return reports

    def print_summary(self, reports: list[EvaluationReport]) -> None:
        """Imprime un resumen tabular de todos los informes."""
        if not reports:
            print("Sin resultados para mostrar.")
            return
        rows = [r.to_dict() for r in reports]
        df   = pd.DataFrame(rows)
        cols = ["model", "horizon", "auroc", "auprc", "loss",
                "optimal_threshold", "fnr", "fpr", "n_pos", "n_total"]
        cols = [c for c in cols if c in df.columns]
        print("\n" + df[cols].to_string(index=False))

    def save_results(
        self,
        reports:    list[EvaluationReport],
        model_name: str,
    ) -> Path:
        """
        Guarda los resultados en CSV dentro de outputs/results/.

        Retorna
        -------
        Path del archivo guardado.
        """
        results_dir = OUTPUT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        rows    = [r.to_dict() for r in reports]
        df      = pd.DataFrame(rows)
        safe_name = model_name.replace(" ", "_").lower()
        path    = results_dir / f"walkforward_{safe_name}.csv"
        df.to_csv(path, index=False)
        print(f"  Resultados guardados → {path.name}")
        return path

    def get_predictions_df(
        self,
        model_factory: Callable,
        model_name:    str,
        horizon:       int,
    ) -> pd.DataFrame:
        """
        Ejecuta el walk-forward para un único horizonte y devuelve
        un DataFrame con todas las predicciones OOS.

        Útil para generar curvas ROC y Precisión-Recall y para el
        análisis de interpretabilidad con SHAP.

        Retorna
        -------
        pd.DataFrame con columnas:
            year, country, y_true, y_prob, model
        """
        target_col = f"crisis_h{horizon}"
        y_true_all, y_prob_all, years_all = self._walk(
            horizon, target_col, model_factory
        )
        # Recuperar países correspondientes a las predicciones OOS
        oos_mask = (
            (self.panel["year"] >= self.eval_start) &
            (self.panel["year"] <= self.eval_end) &
            self.panel[[target_col] + self.feature_cols]
                .notna().all(axis=1)
        )
        oos_panel = self.panel[oos_mask].copy()
        min_len   = min(len(y_true_all), len(oos_panel))
        return pd.DataFrame({
            "year"    : years_all[:min_len],
            "country" : oos_panel["country"].values[:min_len],
            "y_true"  : y_true_all[:min_len],
            "y_prob"  : y_prob_all[:min_len],
            "model"   : model_name,
        })

    # ------------------------------------------------------------------
    # Bucle interno walk-forward
    # ------------------------------------------------------------------

    def _walk(
        self,
        h:          int,
        target_col: str,
        model_factory: Callable,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Itera año a año en el período OOS, reentrenando en cada paso
        con toda la historia disponible hasta t-1.

        Aplica crisis window exclusion: las observaciones en que el
        país está en crisis activa (crisis_bin=1) se excluyen del
        entrenamiento, para que el modelo aprenda a anticipar el inicio
        de la crisis y no solo a identificar el estado activo.

        Retorna arrays acumulados de (y_true, y_prob, years) para el
        período OOS completo.
        """
        y_true_list, y_prob_list, years_list = [], [], []

        eval_years = range(self.eval_start, self.eval_end + 1)

        for t in eval_years:
            # ── Conjunto de entrenamiento: todo hasta t-1 ─────────────
            train_mask = self._train_mask(t, target_col)
            train_df   = self.panel[train_mask]

            # Mínimo de positivos para entrenar de forma significativa
            if train_df[target_col].sum() < MIN_TRAIN_POS:
                continue

            X_train    = train_df[self.feature_cols]
            y_train    = train_df[target_col]
            c_train    = train_df["country"]

            # ── Conjunto de prueba: año t ─────────────────────────────
            test_mask = self._test_mask(t, target_col)
            test_df   = self.panel[test_mask]

            if len(test_df) == 0:
                continue

            X_test  = test_df[self.feature_cols]
            y_test  = test_df[target_col]
            c_test  = test_df["country"]

            # ── Ajustar y predecir ────────────────────────────────────
            model = model_factory()
            try:
                model.fit(X_train, y_train, c_train)
                probs = model.predict_proba(X_test, c_test)
            except Exception as e:
                print(f"    ✗ Error en t={t}, h={h}: {e}")
                continue

            y_true_list.append(y_test.values)
            y_prob_list.append(probs)
            years_list.append(np.full(len(y_test), t))

            n_pos_test = int(y_test.sum())
            if n_pos_test > 0:
                print(
                    f"    t={t}: train={len(train_df)} obs "
                    f"({int(train_df[target_col].sum())} pos) | "
                    f"test={len(test_df)} obs ({n_pos_test} pos) ✓"
                )

        if not y_true_list:
            return np.array([]), np.array([]), np.array([])

        return (
            np.concatenate(y_true_list),
            np.concatenate(y_prob_list),
            np.concatenate(years_list),
        )

    def _train_mask(self, t: int, target_col: str) -> pd.Series:
        """
        Máscara booleana para el conjunto de entrenamiento en el año t.
        Excluye:
          - Observaciones del año t en adelante (no data leakage).
          - Observaciones con crisis activa (crisis_bin=1): crisis window
            exclusion, siguiendo Drehmann & Juselius (2014).
          - Observaciones con NaN en target o features.
        """
        base = (
            (self.panel["year"] < t) &
            (self.panel["crisis_bin"] == 0) &   # crisis window exclusion
            self.panel[target_col].notna()
        )
        # Requiere al menos una feature no-NaN (imputación se hace en fit)
        has_data = self.panel[self.feature_cols].notna().any(axis=1)
        return base & has_data

    def _test_mask(self, t: int, target_col: str) -> pd.Series:
        """
        Máscara para el conjunto de prueba: observaciones del año t
        con target y al menos una feature disponible.
        """
        base = (
            (self.panel["year"] == t) &
            self.panel[target_col].notna()
        )
        has_data = self.panel[self.feature_cols].notna().any(axis=1)
        return base & has_data
