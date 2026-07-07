"""
=============================================================================
eda/plotter.py
=============================================================================
Clase EDAPlotter — generación de todas las figuras del EDA.

Cada método público produce una figura, la guarda en disco y cierra
la figura para liberar memoria. El método ``plot_all()`` ejecuta
la secuencia completa.

Figuras generadas
-----------------
  fig01 — Mapa de disponibilidad de datos (heatmap de NaN)
  fig02 — Línea temporal de episodios de crisis por país
  fig03 — Frecuencia anual de crisis (nº países en crisis)
  fig04 — Distribuciones crisis vs no-crisis de variables clave
  fig05 — Matriz de correlación de Spearman entre predictores
  fig06 — Evolución temporal (media panel) de predictores clave
  fig07 — Desbalance de clases por variable objetivo
  fig08 — Brecha crédito/PIB por país con marcas de crisis
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from config.settings import (
    YEAR_MIN, YEAR_MAX,
    FORECAST_HORIZONS,
    COLORS, PLOT_STYLE, PLOT_PALETTE, PLOT_FONT_SCALE, FIGURE_DPI,
    FIGURES_DIR,
)


class EDAPlotter:
    """
    Genera y guarda todas las figuras del Análisis Exploratorio de Datos.

    Parámetros
    ----------
    panel : pd.DataFrame
        Panel maestro producido por DataPreprocessor.
    feature_list : list[str]
        Lista de columnas predictoras seleccionadas.

    Uso típico
    ----------
    >>> plotter = EDAPlotter(prep.panel_master, prep.feature_list)
    >>> plotter.plot_all()
    """

    def __init__(self, panel: pd.DataFrame, feature_list: list[str]) -> None:
        self.panel        = panel.copy()
        self.feature_list = feature_list
        self._out         = FIGURES_DIR

        # Aplicar tema global
        sns.set_theme(
            style=PLOT_STYLE,
            palette=PLOT_PALETTE,
            font_scale=PLOT_FONT_SCALE,
        )

        # Caché: nº países en crisis por año (usado en varias figuras)
        self._crisis_by_year: pd.Series = (
            panel.groupby("year")["crisis_bin"].sum()
                 .reindex(range(YEAR_MIN, YEAR_MAX + 1), fill_value=0)
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def plot_all(self) -> None:
        """Genera las 8 figuras del EDA en secuencia."""
        print("[EDAPlotter] Generando figuras…")
        self.plot_missing_heatmap()
        self.plot_crisis_timeline()
        self.plot_crisis_frequency()
        self.plot_distributions()
        self.plot_correlation_matrix()
        self.plot_feature_trends()
        self.plot_class_imbalance()
        self.plot_credit_gap_by_country()
        print(f"[EDAPlotter] 8 figuras guardadas en {self._out}")

    # ------------------------------------------------------------------
    # Figura 1 — Disponibilidad de datos
    # ------------------------------------------------------------------

    def plot_missing_heatmap(self) -> None:
        """
        Mapa de calor de disponibilidad de datos.
        Verde = disponible · Rojo = faltante.
        Permite identificar de un vistazo qué variables y períodos
        tienen cobertura incompleta.
        """
        feat_cols = [f for f in self.feature_list if f in self.panel.columns]
        heat = self.panel[feat_cols].isnull().astype(int)

        fig, ax = plt.subplots(figsize=(14, 7))
        im = ax.imshow(
            heat.T, aspect="auto", cmap="RdYlGn_r",
            vmin=0, vmax=1, interpolation="nearest",
        )
        ax.set_yticks(range(len(feat_cols)))
        ax.set_yticklabels(feat_cols, fontsize=8)
        step = max(1, len(self.panel) // 20)
        ax.set_xticks(range(0, len(self.panel), step))
        ax.set_xticklabels(
            self.panel["year"].iloc[::step].astype(int),
            rotation=90, fontsize=7,
        )
        ax.set_title(
            "Figura 1 — Mapa de disponibilidad de datos\n"
            "(verde = disponible · rojo = faltante)",
            fontsize=11, pad=12,
        )
        ax.set_xlabel("Observación (país × año)")
        ax.set_ylabel("Variable")
        plt.colorbar(
            im, ax=ax, fraction=0.015, pad=0.02,
            label="0 = disponible  |  1 = faltante",
        )
        self._save(fig, "fig01_missing_data_heatmap.png")

    # ------------------------------------------------------------------
    # Figura 2 — Línea temporal de crisis
    # ------------------------------------------------------------------

    def plot_crisis_timeline(self) -> None:
        """
        Muestra cada año de crisis activa como una barra vertical roja
        para cada uno de los 18 países. Permite identificar visualmente
        los episodios sincronizados (crisis globales de 1929 y 2008).
        """
        countries = sorted(self.panel["country"].unique())
        y_pos = {c: i for i, c in enumerate(countries)}
        crisis_rows = self.panel[self.panel["crisis_bin"] == 1]

        fig, ax = plt.subplots(figsize=(14, 6))
        for _, row in crisis_rows.iterrows():
            ax.scatter(
                row["year"], y_pos[row["country"]],
                color=COLORS["crisis"], marker="|", s=200, linewidths=2,
            )
        ax.set_yticks(list(y_pos.values()))
        ax.set_yticklabels(list(y_pos.keys()), fontsize=9)
        ax.set_xlabel("Año")
        ax.set_title(
            "Figura 2 — Episodios de crisis bancaria sistémica (LV 2018)\n"
            "Cada barra vertical = un año de crisis activa",
            fontsize=11,
        )
        for yr, label in [
            (1929, "Gran Depresión"),
            (1991, "Crisis nórdica"),
            (2008, "GFC 2008"),
        ]:
            ax.axvline(yr, color="grey", lw=0.8, ls="--", alpha=0.6)
            ax.text(
                yr + 0.5, len(countries) - 0.5, label,
                fontsize=7.5, color="grey", rotation=90, va="top",
            )
        self._save(fig, "fig02_crisis_timeline.png")

    # ------------------------------------------------------------------
    # Figura 3 — Frecuencia anual
    # ------------------------------------------------------------------

    def plot_crisis_frequency(self) -> None:
        """
        Barras: número de países con crisis activa en cada año.
        Ilustra la naturaleza sistémica y sincronizada de las crisis
        (picos claros en 1929-1933 y 2008-2010).
        """
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.bar(
            self._crisis_by_year.index,
            self._crisis_by_year.values,
            color=COLORS["crisis"], alpha=0.75, width=0.9,
        )
        ax.set_xlabel("Año")
        ax.set_ylabel("N.º de países en crisis")
        ax.set_title(
            "Figura 3 — Número de países con crisis bancaria activa por año",
            fontsize=11,
        )
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        self._save(fig, "fig03_crisis_frequency.png")

    # ------------------------------------------------------------------
    # Figura 4 — Distribuciones crisis vs no-crisis
    # ------------------------------------------------------------------

    def plot_distributions(self) -> None:
        """
        Histogramas de densidad de las variables clave separados por
        crisis (rojo) y no-crisis (azul). Un desplazamiento visible
        entre las distribuciones indica poder predictivo univariante.
        Los valores se recortan al percentil 1-99 para evitar que
        outliers distorsionen la escala.
        """
        key_feats = [
            f for f in [
                "tloans", "tloans_growth", "tloans_gap", "hpnom_gap",
                "lev", "ltd", "term_spread", "ca", "debtgdp",
            ]
            if f in self.panel.columns
        ]

        n_cols = 4
        n_rows = -(-len(key_feats) // n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows)
        )
        axes = axes.flatten()

        for i, feat in enumerate(key_feats):
            ax  = axes[i]
            sub = self.panel[[feat, "crisis_bin"]].dropna()
            lo, hi = np.nanpercentile(sub[feat], [1, 99])
            c0 = sub[sub["crisis_bin"] == 0][feat].clip(lo, hi)
            c1 = sub[sub["crisis_bin"] == 1][feat].clip(lo, hi)
            ax.hist(c0, bins=40, color=COLORS["no_crisis"],
                    alpha=0.6, density=True, label="Sin crisis")
            ax.hist(c1, bins=40, color=COLORS["crisis"],
                    alpha=0.65, density=True, label="Crisis")
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Figura 4 — Distribución de variables clave: crisis vs no-crisis\n"
            "(densidades · valores en percentil 1-99)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig04_distributions_crisis_vs_nocrisis.png")

    # ------------------------------------------------------------------
    # Figura 5 — Matriz de correlación
    # ------------------------------------------------------------------

    def plot_correlation_matrix(self) -> None:
        """
        Matriz de correlación de Spearman (triangular inferior) entre
        todos los predictores seleccionados. La correlación de Spearman
        es más robusta que Pearson ante outliers y no linealidades,
        habituales en series macro-financieras.
        """
        corr_vars = [f for f in self.feature_list if f in self.panel.columns]
        corr = self.panel[corr_vars].corr(method="spearman")
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(
            corr, mask=mask, cmap="RdBu_r",
            center=0, vmin=-1, vmax=1,
            linewidths=0.3, linecolor="white",
            annot=False, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Correlación de Spearman"},
        )
        ax.set_title(
            "Figura 5 — Matriz de correlación de Spearman\n"
            "entre variables predictoras",
            fontsize=11, pad=14,
        )
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
        self._save(fig, "fig05_correlation_matrix.png")

    # ------------------------------------------------------------------
    # Figura 6 — Tendencias temporales
    # ------------------------------------------------------------------

    def plot_feature_trends(self) -> None:
        """
        Evolución de la media del panel para cuatro predictores clave.
        Las áreas rojas sombrean los años en que ≥3 países están
        simultáneamente en crisis (sincronización sistémica).
        La línea discontinua marca el cero para las variables de brecha.
        """
        trend_feats = [
            ("tloans",       "Crédito total / PIB"),
            ("tloans_gap",   "Brecha crédito / PIB  (HP gap, λ=400)"),
            ("hpnom_gap",    "Brecha precios vivienda (HP gap, λ=400)"),
            ("term_spread",  "Spread de tipos  (LP − CP)"),
        ]
        trend_feats = [(f, l) for f, l in trend_feats if f in self.panel.columns]

        fig, axes = plt.subplots(
            len(trend_feats), 1,
            figsize=(13, 3.5 * len(trend_feats)),
            sharex=True,
        )
        if len(trend_feats) == 1:
            axes = [axes]

        for ax, (feat, label) in zip(axes, trend_feats):
            annual = (
                self.panel.groupby("year")[feat]
                           .mean()
                           .reindex(range(YEAR_MIN, YEAR_MAX + 1))
            )
            ax.plot(annual.index, annual.values,
                    color=COLORS["no_crisis"], lw=1.6)
            ax.fill_between(annual.index, annual.values,
                            alpha=0.15, color=COLORS["no_crisis"])
            if "gap" in feat:
                ax.axhline(0, color="black", lw=0.7, ls="--")
            for yr in self._crisis_by_year[self._crisis_by_year >= 3].index:
                ax.axvspan(yr - 0.4, yr + 0.4,
                           color=COLORS["crisis"], alpha=0.20, lw=0)
            ax.set_ylabel(label, fontsize=9)
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Año")
        fig.suptitle(
            "Figura 6 — Evolución temporal (media panel) de predictores clave\n"
            "Áreas rojas = años con ≥3 países en crisis  ·  "
            "línea discontinua = cero",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig06_feature_trends.png")

    # ------------------------------------------------------------------
    # Figura 7 — Desbalance de clases
    # ------------------------------------------------------------------

    def plot_class_imbalance(self) -> None:
        """
        Barras que muestran la distribución de clases (crisis / no-crisis)
        para la variable activa y los tres horizontes de predicción.
        El porcentaje indica la proporción de observaciones positivas.
        El desbalance severo (~2%) justifica el uso de datos sintéticos
        y métricas de evaluación asimétricas.
        """
        targets = ["crisis_bin"] + [f"crisis_h{h}" for h in FORECAST_HORIZONS]
        labels  = [
            "crisis_bin\n(activa)",
            "crisis_h1\n(h=1 año)",
            "crisis_h2\n(h=2 años)",
            "crisis_h3\n(h=3 años)",
        ]
        fig, axes = plt.subplots(1, 4, figsize=(13, 4))

        for ax, tgt, lbl in zip(axes, targets, labels):
            if tgt not in self.panel.columns:
                ax.set_visible(False)
                continue
            counts = self.panel[tgt].value_counts().sort_index()
            ax.bar(
                [0, 1],
                [counts.get(0, 0), counts.get(1, 0)],
                color=[COLORS["no_crisis"], COLORS["crisis"]],
            )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["No crisis", "Crisis"], fontsize=9)
            ax.set_title(lbl, fontsize=9)
            ax.set_ylabel("N.º observaciones")
            pct = 100 * counts.get(1, 0) / counts.sum()
            ax.text(
                1, counts.get(1, 0) + 5, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
                color=COLORS["crisis"], fontweight="bold",
            )

        fig.suptitle(
            "Figura 7 — Desbalance de clases (variable dependiente)\n"
            "Porcentaje = proporción de observaciones positivas (crisis)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig07_class_imbalance.png")

    # ------------------------------------------------------------------
    # Figura 8 — Credit gap por país
    # ------------------------------------------------------------------

    def plot_credit_gap_by_country(self) -> None:
        """
        Brecha crédito/PIB (HP gap) para seis países seleccionados.
        Las áreas rojas sombrean los años de crisis activa.
        Un gap positivo elevado antes de las áreas rojas confirma
        que el auge crediticio precede a las crisis sistémicas,
        validando el poder predictivo de este indicador.
        """
        if "tloans_gap" not in self.panel.columns:
            print("  [EDAPlotter] tloans_gap no disponible; fig08 omitida.")
            return

        sel = ["USA", "Spain", "Sweden", "Finland", "Japan", "Germany"]
        sel = [c for c in sel if c in self.panel["country"].unique()]

        rows = -(-len(sel) // 2)
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = axes.flatten()

        for ax, country in zip(axes, sel):
            sub = self.panel[self.panel["country"] == country].sort_values("year")
            ax.plot(sub["year"], sub["tloans_gap"],
                    color=COLORS["no_crisis"], lw=1.5)
            ax.axhline(0, color="black", lw=0.6, ls="--")
            for yr in sub[sub["crisis_bin"] == 1]["year"]:
                ax.axvspan(yr - 0.5, yr + 0.5,
                           color=COLORS["crisis"], alpha=0.30, lw=0)
            ax.set_title(country, fontsize=10, fontweight="bold")
            ax.set_xlabel("Año", fontsize=8)
            ax.set_ylabel("Brecha crédito/PIB", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(len(sel), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Figura 8 — Brecha crédito/PIB (HP gap) por país\n"
            "Áreas rojas = crisis activa (LV)  ·  "
            "gap positivo = auge crediticio",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig08_credit_gap_by_country.png")

    # ------------------------------------------------------------------
    # Método auxiliar privado
    # ------------------------------------------------------------------

    def _save(self, fig: plt.Figure, filename: str) -> None:
        """Guarda la figura, cierra y libera memoria."""
        path = self._out / filename
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {filename}")
