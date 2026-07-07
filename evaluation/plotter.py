"""
=============================================================================
evaluation/plotter.py
=============================================================================
Clase ResultsPlotter — visualización de resultados de evaluación.

Figuras generadas
-----------------
  fig09 — Curvas ROC para los tres horizontes de predicción
  fig10 — Curvas Precisión-Recall para los tres horizontes
  fig11 — Función de pérdida L(μ) en función del umbral de probabilidad
  fig12 — Coeficientes del modelo logit con odds-ratios
  fig13 — Probabilidades predichas OOS vs episodios de crisis reales
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

from config.settings import (
    COLORS, PLOT_STYLE, PLOT_PALETTE, PLOT_FONT_SCALE,
    FIGURE_DPI, FIGURES_DIR, FORECAST_HORIZONS,
)
from evaluation.metrics import EvaluationReport


class ResultsPlotter:
    """
    Genera las figuras de evaluación del modelo logit baseline y, en
    módulos posteriores, de los modelos híbridos generativos.

    Parámetros
    ----------
    model_name : str
        Nombre del modelo (aparece en los títulos de las figuras).

    Uso típico
    ----------
    >>> plotter = ResultsPlotter("LogitPanel")
    >>> plotter.plot_roc_curves(pred_dfs)
    >>> plotter.plot_pr_curves(pred_dfs)
    >>> plotter.plot_loss_vs_threshold(pred_dfs)
    >>> plotter.plot_coefficients(coef_df)
    >>> plotter.plot_predicted_probs(pred_df, panel)
    """

    def __init__(self, model_name: str = "LogitPanel") -> None:
        self.model_name = model_name
        self._out       = FIGURES_DIR
        sns.set_theme(
            style=PLOT_STYLE, palette=PLOT_PALETTE,
            font_scale=PLOT_FONT_SCALE,
        )

    # ------------------------------------------------------------------
    # Figura 9 — Curvas ROC
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        pred_dfs: dict[int, pd.DataFrame],
        reports:  list[EvaluationReport] | None = None,
    ) -> None:
        """
        Curvas ROC para cada horizonte de predicción.

        Parámetros
        ----------
        pred_dfs : dict {horizon → DataFrame con columnas y_true, y_prob}
        reports  : lista de EvaluationReport (para anotar AUROC en la leyenda)
        """
        auroc_map = {}
        if reports:
            auroc_map = {r.horizon: r.auroc.value for r in reports}

        n_h   = len(pred_dfs)
        palette = sns.color_palette("tab10", n_h)

        fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=True)
        if n_h == 1:
            axes = [axes]

        for ax, (h, df), color in zip(axes, sorted(pred_dfs.items()), palette):
            fpr, tpr, _ = roc_curve(df["y_true"], df["y_prob"])
            auroc_str   = (f"AUROC = {auroc_map[h]:.3f}"
                           if h in auroc_map else "")
            ax.plot(fpr, tpr, color=color, lw=2, label=auroc_str)
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
            ax.fill_between(fpr, tpr, alpha=0.10, color=color)
            ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=9)
            ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=9)
            ax.set_title(f"h = {h} año{'s' if h > 1 else ''}", fontsize=10)
            if auroc_str:
                ax.legend(fontsize=9, loc="lower right")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle(
            f"Figura 9 — Curvas ROC  ({self.model_name})\n"
            "Período de evaluación out-of-sample 1990–2018",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig09_roc_curves.png")

    # ------------------------------------------------------------------
    # Figura 10 — Curvas Precisión-Recall
    # ------------------------------------------------------------------

    def plot_pr_curves(
        self,
        pred_dfs: dict[int, pd.DataFrame],
        reports:  list[EvaluationReport] | None = None,
    ) -> None:
        """
        Curvas Precisión-Recall para cada horizonte.
        Más informativas que las ROC bajo desbalance severo de clases.
        La línea horizontal discontinua indica la precisión del clasificador
        aleatorio (= proporción de positivos en el conjunto).
        """
        auprc_map = {}
        if reports:
            auprc_map = {r.horizon: r.auprc.value for r in reports}

        n_h     = len(pred_dfs)
        palette = sns.color_palette("tab10", n_h)

        fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=True)
        if n_h == 1:
            axes = [axes]

        for ax, (h, df), color in zip(axes, sorted(pred_dfs.items()), palette):
            prec, rec, _ = precision_recall_curve(df["y_true"], df["y_prob"])
            baseline     = df["y_true"].mean()
            auprc_str    = (f"AUPRC = {auprc_map[h]:.3f}"
                            if h in auprc_map else "")
            ax.plot(rec, prec, color=color, lw=2, label=auprc_str)
            ax.axhline(baseline, color="grey", lw=0.9, ls="--",
                       label=f"Aleatorio = {baseline:.3f}")
            ax.set_xlabel("Recall (Sensibilidad)", fontsize=9)
            ax.set_ylabel("Precisión", fontsize=9)
            ax.set_title(f"h = {h} año{'s' if h > 1 else ''}", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle(
            f"Figura 10 — Curvas Precisión-Recall  ({self.model_name})\n"
            "Línea discontinua = clasificador aleatorio (baseline de precisión)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig10_pr_curves.png")

    # ------------------------------------------------------------------
    # Figura 11 — Función de pérdida vs umbral
    # ------------------------------------------------------------------

    def plot_loss_vs_threshold(
        self,
        pred_dfs: dict[int, pd.DataFrame],
        mu:       float = 0.75,
    ) -> None:
        """
        Función de pérdida asimétrica L(μ) en función del umbral de
        probabilidad, para cada horizonte.

        Permite visualizar el umbral óptimo (mínimo de la curva) y la
        sensibilidad de la pérdida a desviaciones del umbral óptimo.
        También muestra los componentes FNR y FPR por separado para
        ilustrar el trade-off entre ambos tipos de error.
        """
        n_h     = len(pred_dfs)
        palette = sns.color_palette("tab10", n_h)
        thresholds = np.linspace(0.01, 0.99, 200)

        fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5))
        if n_h == 1:
            axes = [axes]

        for ax, (h, df), color in zip(axes, sorted(pred_dfs.items()), palette):
            y_true = df["y_true"].values
            y_prob = df["y_prob"].values
            losses, fnrs, fprs = [], [], []

            for thr in thresholds:
                y_pred = (y_prob >= thr).astype(int)
                tp = ((y_pred == 1) & (y_true == 1)).sum()
                fn = ((y_pred == 0) & (y_true == 1)).sum()
                fp = ((y_pred == 1) & (y_true == 0)).sum()
                tn = ((y_pred == 0) & (y_true == 0)).sum()
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                fnrs.append(fnr); fprs.append(fpr)
                losses.append(mu * fnr + (1 - mu) * fpr)

            losses = np.array(losses)
            opt_idx = np.argmin(losses)
            opt_thr = thresholds[opt_idx]

            ax.plot(thresholds, losses, color=color,   lw=2,   label=f"L(μ={mu})")
            ax.plot(thresholds, fnrs,   color="red",   lw=1.2, ls="--",
                    alpha=0.7, label="FNR")
            ax.plot(thresholds, fprs,   color="blue",  lw=1.2, ls="--",
                    alpha=0.7, label="FPR")
            ax.axvline(opt_thr, color="black", lw=0.9, ls=":",
                       label=f"Umbral óptimo = {opt_thr:.3f}")
            ax.set_xlabel("Umbral de probabilidad", fontsize=9)
            ax.set_ylabel("Valor de la función", fontsize=9)
            ax.set_title(f"h = {h} año{'s' if h > 1 else ''}", fontsize=10)
            ax.legend(fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        fig.suptitle(
            f"Figura 11 — Función de pérdida asimétrica L(μ={mu}) vs umbral\n"
            f"({self.model_name})  ·  μ={mu} penaliza más las crisis no detectadas",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig11_loss_vs_threshold.png")

    # ------------------------------------------------------------------
    # Figura 12 — Coeficientes del logit
    # ------------------------------------------------------------------

    def plot_coefficients(
        self,
        coef_df:  pd.DataFrame,
        top_n:    int = 15,
        horizon:  int = 1,
    ) -> None:
        """
        Gráfico de barras horizontales de los coeficientes del modelo
        logit (excluidas las dummies de país).

        Los coeficientes positivos indican variables asociadas con mayor
        riesgo de crisis; los negativos con menor riesgo.
        Las barras se colorean en rojo (riesgo creciente) o azul
        (riesgo decreciente) para facilitar la interpretación.

        Parámetros
        ----------
        coef_df : pd.DataFrame
            Output de LogitPanel.get_coef_df() con columnas
            [feature, coef, odds_ratio, direction].
        top_n : int
            Número de variables (por valor absoluto del coeficiente)
            a mostrar. Default: 15.
        horizon : int
            Horizonte de predicción (para el título).
        """
        # Excluir dummies de país
        df = coef_df[~coef_df["feature"].str.startswith("fe_")].copy()
        df = df.head(top_n)

        colors = [
            COLORS["crisis"] if d == "↑ riesgo" else COLORS["no_crisis"]
            for d in df["direction"]
        ]

        fig, ax = plt.subplots(figsize=(9, 0.45 * len(df) + 2))
        bars = ax.barh(df["feature"], df["coef"], color=colors, alpha=0.80)
        ax.axvline(0, color="black", lw=0.8)

        # Anotar odds-ratios
        for bar, (_, row) in zip(bars, df.iterrows()):
            x    = bar.get_width()
            sign = 1 if x >= 0 else -1
            ax.text(
                x + sign * 0.02, bar.get_y() + bar.get_height() / 2,
                f"OR={row['odds_ratio']:.2f}",
                va="center", ha="left" if x >= 0 else "right",
                fontsize=7, color="black",
            )

        ax.set_xlabel("Coeficiente logit (log-odds)", fontsize=9)
        ax.set_title(
            f"Figura 12 — Coeficientes estimados del modelo logit  (h={horizon})\n"
            f"Rojo = ↑ riesgo · Azul = ↓ riesgo · OR = odds-ratio",
            fontsize=10,
        )
        ax.invert_yaxis()
        plt.tight_layout()
        self._save(fig, f"fig12_logit_coefficients_h{horizon}.png")

    # ------------------------------------------------------------------
    # Figura 13 — Probabilidades predichas OOS vs crisis reales
    # ------------------------------------------------------------------

    def plot_predicted_probs(
        self,
        pred_df:    pd.DataFrame,
        panel:      pd.DataFrame,
        horizon:    int = 1,
        countries:  list[str] | None = None,
    ) -> None:
        """
        Serie temporal de probabilidades predichas OOS para países
        seleccionados, con marcas de los años de crisis real.

        Permite verificar visualmente si el modelo genera señales de
        alerta temprana efectivas antes del inicio de las crisis.

        Parámetros
        ----------
        pred_df    : DataFrame con columnas [year, country, y_prob, y_true]
        panel      : Panel maestro (para extraer crisis_bin)
        horizon    : Horizonte de predicción (para el título)
        countries  : Países a graficar. Default: 6 países con crisis OOS.
        """
        if countries is None:
            # Países con al menos un positivo en el período OOS
            countries_with_crisis = (
                pred_df[pred_df["y_true"] == 1]["country"]
                .value_counts().head(6).index.tolist()
            )
            countries = countries_with_crisis if countries_with_crisis else (
                pred_df["country"].unique()[:6].tolist()
            )

        n  = len(countries)
        nc = min(2, n)
        nr = -(-n // nc)

        fig, axes = plt.subplots(nr, nc, figsize=(7 * nc, 3.5 * nr),
                                 sharex=False)
        axes = np.array(axes).flatten()

        for ax, country in zip(axes, countries):
            sub = pred_df[pred_df["country"] == country].sort_values("year")
            if sub.empty:
                ax.set_visible(False)
                continue

            ax.plot(sub["year"], sub["y_prob"],
                    color=COLORS["no_crisis"], lw=1.6, label="P(crisis)")
            ax.fill_between(sub["year"], sub["y_prob"],
                            alpha=0.15, color=COLORS["no_crisis"])

            # Sombrear años con crisis real activa (crisis_bin=1)
            crisis_years = panel[
                (panel["country"] == country) &
                (panel["crisis_bin"] == 1)
            ]["year"]
            for yr in crisis_years:
                ax.axvspan(yr - 0.5, yr + 0.5,
                           color=COLORS["crisis"], alpha=0.30, lw=0)

            ax.set_ylim(0, 1)
            ax.set_title(country, fontsize=10, fontweight="bold")
            ax.set_xlabel("Año", fontsize=8)
            ax.set_ylabel("P(crisis)", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(len(countries), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            f"Figura 13 — Probabilidades predichas OOS  ({self.model_name}, h={horizon})\n"
            "Áreas rojas = años de crisis activa (LV)  ·  "
            "Período 1990–2018",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, f"fig13_predicted_probs_h{horizon}.png")

    # ------------------------------------------------------------------
    # Auxiliar
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = self._out / filename
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {filename}")
