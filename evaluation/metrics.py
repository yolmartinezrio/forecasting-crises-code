"""
=============================================================================
evaluation/metrics.py
=============================================================================
Clase EvaluationMetrics — cálculo de métricas de evaluación.

Métricas implementadas
----------------------
  1. AUROC  : área bajo la curva ROC. Mide la capacidad discriminante
              global del modelo. Insensible al umbral de decisión.
  2. AUPRC  : área bajo la curva Precisión-Recall. Más informativa que
              AUROC en presencia de fuerte desbalance de clases
              (Saito & Rehmsmeier, 2015).
  3. L(μ)   : función de pérdida asimétrica de Alessi & Detken (2011).
              Penaliza de manera diferenciada los falsos negativos
              (crisis no detectadas) y los falsos positivos (falsas
              alarmas), reflejando las preferencias del supervisor
              macroprudencial.
  4. IC 95% : intervalos de confianza bootstrap por bloques temporales
              (block bootstrap) para las tres métricas anteriores,
              que respetan la dependencia temporal de las observaciones.

Referencias
-----------
Alessi, L. & Detken, C. (2011). Quasi real time early warning indicators
for costly asset price boom/bust cycles. European Journal of Political
Economy, 27(3), 520-533.

Saito, T. & Rehmsmeier, M. (2015). The precision-recall plot is more
informative than the ROC plot when evaluating binary classifiers on
imbalanced datasets. PLOS ONE, 10(3).
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Parámetros por defecto
# ---------------------------------------------------------------------------

MU_DEFAULT   = 0.75   # peso de los falsos negativos en L(μ): μ ∈ (0.5, 1)
                       # μ=0.75 → penaliza el doble las crisis no detectadas
                       # que las falsas alarmas (preferencia revelada del
                       # supervisor; Alessi & Detken 2011)
N_BOOTSTRAP  = 1000   # réplicas bootstrap
ALPHA        = 0.05   # nivel de significación para IC (bilateral)
BLOCK_SIZE   = 5      # tamaño del bloque temporal en block bootstrap (años)


# ---------------------------------------------------------------------------
# Dataclass de resultados
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """
    Contenedor de resultados para una métrica individual.

    Atributos
    ---------
    name    : nombre legible de la métrica
    value   : valor puntual
    ci_low  : límite inferior del IC (1-α)%
    ci_high : límite superior del IC (1-α)%
    n_pos   : número de observaciones positivas en la muestra evaluada
    n_total : número total de observaciones evaluadas
    """
    name:    str
    value:   float
    ci_low:  float = np.nan
    ci_high: float = np.nan
    n_pos:   int   = 0
    n_total: int   = 0

    def __str__(self) -> str:
        ci = (f"  IC95% [{self.ci_low:.4f}, {self.ci_high:.4f}]"
              if not np.isnan(self.ci_low) else "")
        return (
            f"{self.name:<12}: {self.value:.4f}{ci}  "
            f"(n_pos={self.n_pos}, n={self.n_total})"
        )


@dataclass
class EvaluationReport:
    """
    Informe completo de evaluación para un modelo y horizonte dados.

    Atributos
    ---------
    model_name : nombre del modelo evaluado
    horizon    : horizonte de predicción h
    auroc      : MetricResult para el AUROC
    auprc      : MetricResult para el AUPRC
    loss       : MetricResult para L(μ)
    mu         : parámetro μ usado en L(μ)
    optimal_threshold : umbral de probabilidad que minimiza L(μ)
    fnr_at_threshold  : tasa de falsos negativos en el umbral óptimo
    fpr_at_threshold  : tasa de falsos positivos en el umbral óptimo
    """
    model_name:         str
    horizon:            int
    auroc:              MetricResult
    auprc:              MetricResult
    loss:               MetricResult
    mu:                 float = MU_DEFAULT
    optimal_threshold:  float = np.nan
    fnr_at_threshold:   float = np.nan
    fpr_at_threshold:   float = np.nan

    def summary(self) -> str:
        sep = "─" * 60
        return (
            f"\n{sep}\n"
            f"  Modelo : {self.model_name}  |  Horizonte h={self.horizon}\n"
            f"{sep}\n"
            f"  {self.auroc}\n"
            f"  {self.auprc}\n"
            f"  {self.loss}  (μ={self.mu})\n"
            f"  Umbral óptimo : {self.optimal_threshold:.4f}  "
            f"(FNR={self.fnr_at_threshold:.3f}, FPR={self.fpr_at_threshold:.3f})\n"
            f"{sep}"
        )

    def to_dict(self) -> dict:
        """Serializa el informe a dict plano para agregación en DataFrame."""
        return {
            "model"             : self.model_name,
            "horizon"           : self.horizon,
            "auroc"             : self.auroc.value,
            "auroc_ci_low"      : self.auroc.ci_low,
            "auroc_ci_high"     : self.auroc.ci_high,
            "auprc"             : self.auprc.value,
            "auprc_ci_low"      : self.auprc.ci_low,
            "auprc_ci_high"     : self.auprc.ci_high,
            "loss"              : self.loss.value,
            "loss_ci_low"       : self.loss.ci_low,
            "loss_ci_high"      : self.loss.ci_high,
            "mu"                : self.mu,
            "optimal_threshold" : self.optimal_threshold,
            "fnr"               : self.fnr_at_threshold,
            "fpr"               : self.fpr_at_threshold,
            "n_pos"             : self.auroc.n_pos,
            "n_total"           : self.auroc.n_total,
        }


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class EvaluationMetrics:
    """
    Calcula AUROC, AUPRC y pérdida asimétrica L(μ) con intervalos de
    confianza bootstrap por bloques temporales.

    Parámetros
    ----------
    mu : float
        Peso de los falsos negativos en L(μ). μ ∈ (0.5, 1).
        μ=0.75 → triplica la penalización de las crisis no detectadas
        respecto a las falsas alarmas.
    n_bootstrap : int
        Número de réplicas bootstrap para los intervalos de confianza.
    block_size : int
        Tamaño del bloque en el block bootstrap (en años).
    alpha : float
        Nivel de significación para los intervalos de confianza.
    random_state : int
        Semilla de aleatoriedad para reproducibilidad.

    Uso típico
    ----------
    >>> em = EvaluationMetrics(mu=0.75)
    >>> report = em.evaluate(
    ...     y_true=y_test,
    ...     y_prob=probs,
    ...     model_name="LogitPanel",
    ...     horizon=1,
    ...     years=years_test,
    ... )
    >>> print(report.summary())
    """

    def __init__(
        self,
        mu:           float = MU_DEFAULT,
        n_bootstrap:  int   = N_BOOTSTRAP,
        block_size:   int   = BLOCK_SIZE,
        alpha:        float = ALPHA,
        random_state: int   = 42,
    ) -> None:
        self.mu           = mu
        self.n_bootstrap  = n_bootstrap
        self.block_size   = block_size
        self.alpha        = alpha
        self.rng          = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def evaluate(
        self,
        y_true:     np.ndarray | pd.Series,
        y_prob:     np.ndarray,
        model_name: str,
        horizon:    int,
        years:      np.ndarray | pd.Series | None = None,
    ) -> EvaluationReport:
        """
        Calcula el informe completo de evaluación.

        Parámetros
        ----------
        y_true : array-like de 0/1
            Etiquetas verdaderas.
        y_prob : array-like de float en [0,1]
            Probabilidades predichas de crisis.
        model_name : str
            Nombre del modelo (para el informe).
        horizon : int
            Horizonte de predicción (para el informe).
        years : array-like | None
            Años correspondientes a cada observación. Si se proporciona,
            se usan para el block bootstrap temporal; si no, se usa
            bootstrap estándar (i.i.d.).

        Retorna
        -------
        EvaluationReport
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        n_pos  = int(y_true.sum())
        n_tot  = len(y_true)

        if n_pos == 0:
            raise ValueError(
                "No hay observaciones positivas en y_true. "
                "No es posible calcular métricas de clasificación."
            )

        # Métricas puntuales
        auroc_val = roc_auc_score(y_true, y_prob)
        auprc_val = average_precision_score(y_true, y_prob)
        opt_thr, fnr, fpr, loss_val = self._optimal_threshold(y_true, y_prob)

        # Bootstrap para IC
        years_arr = np.asarray(years) if years is not None else None
        bs_auroc, bs_auprc, bs_loss = self._bootstrap(
            y_true, y_prob, years_arr
        )

        def ci(samples: np.ndarray) -> tuple[float, float]:
            lo = float(np.nanpercentile(samples, 100 * self.alpha / 2))
            hi = float(np.nanpercentile(samples, 100 * (1 - self.alpha / 2)))
            return lo, hi

        auroc_lo, auroc_hi = ci(bs_auroc)
        auprc_lo, auprc_hi = ci(bs_auprc)
        loss_lo,  loss_hi  = ci(bs_loss)

        return EvaluationReport(
            model_name = model_name,
            horizon    = horizon,
            auroc      = MetricResult("AUROC",  auroc_val, auroc_lo, auroc_hi,
                                      n_pos, n_tot),
            auprc      = MetricResult("AUPRC",  auprc_val, auprc_lo, auprc_hi,
                                      n_pos, n_tot),
            loss       = MetricResult(f"L(μ={self.mu})", loss_val,
                                      loss_lo, loss_hi, n_pos, n_tot),
            mu                  = self.mu,
            optimal_threshold   = opt_thr,
            fnr_at_threshold    = fnr,
            fpr_at_threshold    = fpr,
        )

    # ------------------------------------------------------------------
    # Métricas individuales (accesibles directamente si se necesitan)
    # ------------------------------------------------------------------

    def asymmetric_loss(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ) -> float:
        """
        Función de pérdida asimétrica de Alessi & Detken (2011):

            L(μ) = μ · FNR + (1 − μ) · FPR

        donde FNR = tasa de falsos negativos (crisis no detectadas)
              FPR = tasa de falsos positivos (falsas alarmas)
              μ   = peso relativo del coste de las crisis no detectadas

        Un μ > 0.5 penaliza más los falsos negativos, reflejando que
        los supervisores macroprudenciales prefieren activar el colchón
        anticíclico de forma preventiva aunque no haya crisis, antes
        que dejar pasar una crisis sin advertencia.

        Parámetros
        ----------
        threshold : float
            Umbral de probabilidad para convertir y_prob en predicción
            binaria.
        """
        y_pred = (y_prob >= threshold).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return self.mu * fnr + (1 - self.mu) * fpr

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """
        Encuentra el umbral de probabilidad que minimiza L(μ).

        Evalúa todos los umbrales definidos por la curva ROC y devuelve
        (umbral_óptimo, FNR, FPR, L_mínima).
        """
        fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
        fnr_arr = 1 - tpr_arr

        losses = self.mu * fnr_arr + (1 - self.mu) * fpr_arr
        idx    = np.argmin(losses)

        opt_thr  = float(thresholds[idx])
        opt_fnr  = float(fnr_arr[idx])
        opt_fpr  = float(fpr_arr[idx])
        opt_loss = float(losses[idx])
        return opt_thr, opt_fnr, opt_fpr, opt_loss

    def _bootstrap(
        self,
        y_true:    np.ndarray,
        y_prob:    np.ndarray,
        years:     np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Block bootstrap temporal para los IC de AUROC, AUPRC y L(μ).

        Si se proporcionan años, agrupa las observaciones en bloques
        de tamaño block_size años consecutivos y remuestrea bloques
        completos, preservando la estructura de dependencia temporal.
        Si no se proporcionan años, usa bootstrap estándar (i.i.d.).
        """
        n        = len(y_true)
        bs_auroc = np.full(self.n_bootstrap, np.nan)
        bs_auprc = np.full(self.n_bootstrap, np.nan)
        bs_loss  = np.full(self.n_bootstrap, np.nan)

        if years is not None:
            # Construir bloques por año
            unique_years = np.sort(np.unique(years))
            blocks = []
            i = 0
            while i < len(unique_years):
                block_years = unique_years[i:i + self.block_size]
                idx = np.where(np.isin(years, block_years))[0]
                if len(idx) > 0:
                    blocks.append(idx)
                i += self.block_size
            n_blocks = len(blocks)
        else:
            blocks    = None
            n_blocks  = None

        for b in range(self.n_bootstrap):
            if blocks is not None:
                # Remuestrear bloques con reemplazamiento
                chosen = self.rng.integers(0, n_blocks,
                                           size=n_blocks)
                idx_bs = np.concatenate([blocks[c] for c in chosen])
            else:
                idx_bs = self.rng.integers(0, n, size=n)

            yt = y_true[idx_bs]
            yp = y_prob[idx_bs]

            if yt.sum() < 2:
                # Bootstrap sin positivos: métrica no definida
                continue

            try:
                bs_auroc[b] = roc_auc_score(yt, yp)
                bs_auprc[b] = average_precision_score(yt, yp)
                _, _, _, bs_loss[b] = self._optimal_threshold(yt, yp)
            except Exception:
                pass

        return bs_auroc, bs_auprc, bs_loss
