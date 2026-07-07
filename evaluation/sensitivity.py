"""
=============================================================================
evaluation/sensitivity.py
=============================================================================
Clase SensitivityAnalyzer — análisis de sensibilidad e interpretabilidad.

Análisis implementados
----------------------
  1. Sensibilidad al parámetro μ (función de pérdida asimétrica):
     para μ ∈ [0.55, 0.95] calcula L(μ), τ*, FNR y FPR de los tres
     clasificadores en h = 3.

  2. Sensibilidad al parámetro C (regularización L2 del logit):
     para C ∈ {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0} evalúa el
     rendimiento OOS del LogitPanel en h = 3.

  3. Leave-one-country-out (LOCO):
     excluye cada país del conjunto de evaluación OOS y calcula el
     AUROC resultante del LogitPanel en h = 3.

  4. Importancia de variables (logit coefficient ranking):
     extrae y visualiza los coeficientes del LogitPanel entrenado
     sobre toda la muestra pre-1990 para h ∈ {1, 2, 3}.

Figuras generadas
-----------------
  fig29 — Pérdida L(μ) vs μ para los tres clasificadores (h=3)
  fig30 — FNR y FPR en τ* vs μ para los tres clasificadores (h=3)
  fig31 — AUROC y L(μ) vs C (regularización) para LogitPanel (h=3)
  fig32 — LOCO AUROC: impacto de cada país (LogitPanel, h=3)
  fig33 — Importancia de variables: coeficientes logit para h=1,2,3
  fig34 — Curvas L(μ, τ) para los tres clasificadores (h=3, μ=0.75)

Referencias
-----------
Alessi, L. & Detken, C. (2011). European Journal of Political Economy, 27(3).
Bluwstein et al. (2020). Bank of England Working Paper, 848.
=============================================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    OUTPUT_DIR, FIGURES_DIR, COLORS,
    PLOT_STYLE, PLOT_FONT_SCALE, FIGURE_DPI,
)
from evaluation.comparative import ComparativeEvaluator
from models.logit_panel      import LogitPanel, LOGIT_FEATURES


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MU_GRID  = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
C_GRID   = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
MODELS   = ["LogitPanel", "LogitVAE", "LogitDDPM"]
LABELS   = ["Logit baseline", "Logit+VAE", "Logit+DDPM"]
PALETTE  = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]
MU_REF   = 0.75
H_SENS   = 3       # horizonte para los análisis de sensibilidad
C_REF    = 0.1    # valor de referencia del Capítulo 8


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class SensitivityAnalyzer:
    """
    Ejecuta los cuatro análisis de sensibilidad e interpretabilidad.

    Parámetros
    ----------
    panel : pd.DataFrame        Panel maestro.
    latent_df : pd.DataFrame    Representaciones latentes VAE.
    syn_vae : pd.DataFrame      Datos sintéticos VAE.
    syn_ddpm : pd.DataFrame     Datos sintéticos DDPM.
    feature_cols : list[str]    Features del clasificador logit.
    eval_start/end : int        Período OOS.
    """

    def __init__(
        self,
        panel:        pd.DataFrame,
        latent_df:    pd.DataFrame,
        syn_vae:      pd.DataFrame,
        syn_ddpm:     pd.DataFrame,
        feature_cols: list[str] = None,
        eval_start:   int       = 1990,
        eval_end:     int       = 2018,
    ) -> None:
        self.panel        = panel.copy()
        self.latent_df    = latent_df.copy()
        self.syn_vae      = syn_vae.copy()
        self.syn_ddpm     = syn_ddpm.copy()
        self.feature_cols = feature_cols or [f for f in LOGIT_FEATURES
                                             if f in panel.columns]
        self.eval_start   = eval_start
        self.eval_end     = eval_end

        sns.set_theme(style=PLOT_STYLE, font_scale=PLOT_FONT_SCALE)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "results").mkdir(parents=True, exist_ok=True)

        # Cache de predicciones base (se rellena en run)
        self._pred_base: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> None:
        """Ejecuta todos los análisis en secuencia."""
        print("\n[SensitivityAnalyzer] Iniciando análisis de sensibilidad…")

        # Obtener predicciones base con C=0.1
        print("  Obteniendo predicciones base (C=0.1)…")
        self._pred_base = self._get_predictions(C_val=C_REF)

        print("  [1/4] Sensibilidad a μ…")
        df_mu = self._sensitivity_mu()
        self._save_csv(df_mu, "sensitivity_mu.csv")
        self._plot_mu_sensitivity(df_mu)

        print("  [2/4] Sensibilidad a C…")
        df_c = self._sensitivity_C()
        self._save_csv(df_c, "sensitivity_C.csv")
        self._plot_C_sensitivity(df_c)

        print("  [3/4] Leave-one-country-out (LOCO)…")
        df_loco = self._loco()
        self._save_csv(df_loco, "sensitivity_loco.csv")
        self._plot_loco(df_loco)

        print("  [4/4] Importancia de variables (coeficientes logit)…")
        coef_dfs = self._compute_coefficients()
        self._plot_coefficients(coef_dfs)

        # Figura adicional: curvas L(μ, τ) a μ=0.75
        self._plot_loss_curves()

        print("[SensitivityAnalyzer] Análisis completado.")

    # ------------------------------------------------------------------
    # 1. Sensibilidad a μ
    # ------------------------------------------------------------------

    def _sensitivity_mu(self) -> pd.DataFrame:
        """
        Para cada μ ∈ MU_GRID calcula L(μ, τ*), τ*, FNR y FPR
        de los tres clasificadores con h=H_SENS.
        """
        rows = []
        yt = self._pred_base["LogitPanel"]["y_true"].values

        for mu in MU_GRID:
            for model_name in MODELS:
                if model_name not in self._pred_base:
                    continue
                yp = self._pred_base[model_name]["y_prob"].values
                loss, thr, fnr, fpr = self._opt_threshold(yt, yp, mu)
                rows.append({
                    "mu": mu, "model": model_name,
                    "loss": loss, "threshold": thr,
                    "fnr": fnr, "fpr": fpr,
                })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 2. Sensibilidad a C
    # ------------------------------------------------------------------

    def _sensitivity_C(self) -> pd.DataFrame:
        """
        Para cada C ∈ C_GRID reentrena el LogitPanel y evalúa en h=H_SENS.
        """
        rows = []
        for C_val in C_GRID:
            pred = self._get_predictions(
                C_val=C_val, models_only=["LogitPanel"]
            )
            if "LogitPanel" not in pred:
                continue
            df = pred["LogitPanel"]
            yt = df["y_true"].values
            yp = df["y_prob"].values
            if yt.sum() < 2:
                continue
            auroc = roc_auc_score(yt, yp)
            loss, thr, fnr, fpr = self._opt_threshold(yt, yp, MU_REF)
            from sklearn.metrics import average_precision_score
            auprc = average_precision_score(yt, yp)
            rows.append({
                "C": C_val, "model": "LogitPanel",
                "auroc": auroc, "auprc": auprc,
                "loss": loss, "threshold": thr,
                "fnr": fnr, "fpr": fpr,
            })
            print(f"    C={C_val:.3f}  AUROC={auroc:.4f}  "
                  f"AUPRC={auprc:.4f}  L={loss:.5f}  τ*={thr:.5f}  "
                  f"FNR={fnr:.3f}  FPR={fpr:.3f}")
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. LOCO
    # ------------------------------------------------------------------

    def _loco(self) -> pd.DataFrame:
        """
        Excluye cada país del conjunto de evaluación OOS y calcula
        el AUROC del LogitPanel en h=H_SENS.
        """
        base_df = self._pred_base.get("LogitPanel")
        if base_df is None:
            return pd.DataFrame()

        base_auroc = roc_auc_score(
            base_df["y_true"], base_df["y_prob"]
        )
        countries = sorted(base_df["country"].unique())
        rows = [{"country": "(Muestra completa)", "auroc": base_auroc,
                 "delta": 0.0, "n_pos": int(base_df["y_true"].sum())}]

        for country in countries:
            sub = base_df[base_df["country"] != country]
            if sub["y_true"].sum() < 2:
                continue
            a = roc_auc_score(sub["y_true"], sub["y_prob"])
            rows.append({
                "country": country,
                "auroc":   a,
                "delta":   a - base_auroc,
                "n_pos":   int(sub["y_true"].sum()),
            })
            print(f"    Excl. {country:<15} AUROC={a:.4f}  "
                  f"Δ={a-base_auroc:+.4f}")

        return pd.DataFrame(rows).sort_values("delta")

    # ------------------------------------------------------------------
    # 4. Coeficientes del logit
    # ------------------------------------------------------------------

    def _compute_coefficients(self) -> dict[int, pd.DataFrame]:
        """
        Ajusta el LogitPanel sobre toda la muestra pre-1990 para
        h ∈ {1, 2, 3} y devuelve los DataFrames de coeficientes.
        """
        coef_dfs = {}
        train = self.panel[self.panel["year"] < 1990].copy()
        avail = [f for f in self.feature_cols if f in train.columns]

        for h in [1, 2, 3]:
            tgt = f"crisis_h{h}"
            sub = train[train[tgt].notna() & (train["crisis_bin"] == 0)]
            if sub[tgt].sum() < 3:
                continue
            model = LogitPanel(horizon=h, features=avail,
                               C=C_REF, country_fe=True)
            model.fit(sub[avail], sub[tgt], sub["country"])
            coef_dfs[h] = model.get_coef_df()
            print(f"    h={h}: modelo ajustado sobre {len(sub)} obs "
                  f"({int(sub[tgt].sum())} positivas)")

        return coef_dfs

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------

    def _plot_mu_sensitivity(self, df: pd.DataFrame) -> None:
        """
        fig29 — L(μ) vs μ para los tres clasificadores (h=3).
        fig30 — FNR y FPR en τ* vs μ (h=3).
        """
        # ── fig29 ────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, label, color in zip(MODELS, LABELS, PALETTE):
            sub = df[df["model"] == model].sort_values("mu")
            ax.plot(sub["mu"], sub["loss"], color=color,
                    lw=2, marker="o", ms=5, label=label)

        ax.axvline(MU_REF, color="black", lw=0.9, ls="--", alpha=0.6,
                   label=f"μ de referencia ({MU_REF})")
        ax.set_xlabel("Parámetro de asimetría μ", fontsize=10)
        ax.set_ylabel("Pérdida asimétrica L(μ) en τ*", fontsize=10)
        ax.set_title(
            f"Figura 29 — Sensibilidad de L(μ) al parámetro de asimetría\n"
            f"(h={H_SENS}, período OOS 1990–2018)",
            fontsize=11,
        )
        ax.legend(fontsize=9)
        ax.set_xlim(0.50, 1.00)
        plt.tight_layout()
        self._save(fig, "fig29_sensitivity_mu_loss.png")

        # ── fig30 ────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        for model, label, color in zip(MODELS, LABELS, PALETTE):
            sub = df[df["model"] == model].sort_values("mu")
            axes[0].plot(sub["mu"], sub["fnr"], color=color,
                         lw=2, marker="o", ms=5, label=label)
            axes[1].plot(sub["mu"], sub["fpr"], color=color,
                         lw=2, marker="o", ms=5, label=label)

        for ax, title, ylabel in zip(
            axes,
            ["FNR en umbral óptimo τ*", "FPR en umbral óptimo τ*"],
            ["Tasa de Falsos Negativos (FNR)", "Tasa de Falsos Positivos (FPR)"],
        ):
            ax.axvline(MU_REF, color="black", lw=0.9, ls="--", alpha=0.6)
            ax.set_xlabel("μ", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=10)
            ax.set_xlim(0.50, 1.00)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Figura 30 — FNR y FPR en el umbral óptimo τ* vs μ\n"
            f"(h={H_SENS}, período OOS 1990–2018)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig30_sensitivity_mu_fnr_fpr.png")

    def _plot_C_sensitivity(self, df: pd.DataFrame) -> None:
        """
        fig31 — AUROC y L(μ=0.75) vs C para LogitPanel (h=3).
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        df_s = df.sort_values("C")

        for ax, metric, ylabel, color in [
            (axes[0], "auroc",  "AUROC",           COLORS["no_crisis"]),
            (axes[1], "loss",   "L(μ=0.75)",       COLORS["crisis"]),
        ]:
            ax.semilogx(df_s["C"], df_s[metric],
                        color=color, lw=2, marker="o", ms=6)
            ax.axvline(C_REF, color="black", lw=0.9, ls="--", alpha=0.6,
                       label=f"C ref = {C_REF}")
            # Annotate optimal
            best_idx = df_s[metric].idxmin() if metric == "loss" \
                       else df_s[metric].idxmax()
            best_C   = df_s.loc[best_idx, "C"]
            best_val = df_s.loc[best_idx, metric]
            ax.annotate(
                f"Óptimo\nC={best_C}, {metric[:5]}={best_val:.4f}",
                xy=(best_C, best_val),
                xytext=(best_C * 3, best_val + (0.005 if metric=="auroc" else -0.005)),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=1),
            )
            ax.set_xlabel("Parámetro de regularización C (escala log)", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{ylabel} vs C — LogitPanel (h={H_SENS})", fontsize=10)
            ax.legend(fontsize=8)

        fig.suptitle(
            f"Figura 31 — Sensibilidad al parámetro de regularización C\n"
            f"LogitPanel, h={H_SENS}, μ={MU_REF}, período OOS 1990–2018",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig31_sensitivity_C.png")

    def _plot_loco(self, df: pd.DataFrame) -> None:
        """
        fig32 — AUROC LOCO: impacto de cada país (LogitPanel, h=3).
        Gráfico de barras horizontales ordenadas por Δ.
        """
        df_plot = df[df["country"] != "(Muestra completa)"].copy()
        df_plot = df_plot.sort_values("delta")
        base_auroc = df[df["country"] == "(Muestra completa)"]["auroc"].values[0]

        colors = [
            COLORS["crisis"] if d < 0 else COLORS["no_crisis"]
            for d in df_plot["delta"]
        ]

        fig, ax = plt.subplots(figsize=(9, 0.40 * len(df_plot) + 2))
        bars = ax.barh(df_plot["country"], df_plot["delta"],
                       color=colors, alpha=0.82)
        ax.axvline(0, color="black", lw=0.8)

        for bar, (_, row) in zip(bars, df_plot.iterrows()):
            x = bar.get_width()
            ax.text(
                x + (0.0003 if x >= 0 else -0.0003),
                bar.get_y() + bar.get_height() / 2,
                f"{row['auroc']:.4f}",
                va="center", ha="left" if x >= 0 else "right",
                fontsize=7.5,
            )

        ax.set_xlabel("ΔAUROC = AUROC(sin país) − AUROC(todos)", fontsize=9)
        ax.set_title(
            f"Figura 32 — Análisis leave-one-country-out (LOCO)\n"
            f"LogitPanel, h={H_SENS}. "
            f"AUROC muestra completa = {base_auroc:.4f}.\n"
            f"Azul = exclusión mejora el AUROC · Rojo = exclusión lo empeora",
            fontsize=10,
        )
        plt.tight_layout()
        self._save(fig, "fig32_loco_auroc.png")

    def _plot_coefficients(self, coef_dfs: dict) -> None:
        """
        fig33 — Coeficientes logit (excl. dummies de país) para h=1,2,3.
        Tres subgráficos horizontales con barras y odds-ratios.
        """
        if not coef_dfs:
            return

        horizons = sorted(coef_dfs.keys())
        fig, axes = plt.subplots(
            1, len(horizons),
            figsize=(6.5 * len(horizons), 8),
            sharey=False,
        )
        if len(horizons) == 1:
            axes = [axes]

        for ax, h in zip(axes, horizons):
            df = coef_dfs[h]
            # Excluir dummies de país
            df = df[~df["feature"].str.startswith("fe_")].head(16)

            colors = [
                COLORS["crisis"] if d == "↑ riesgo" else COLORS["no_crisis"]
                for d in df["direction"]
            ]
            ax.barh(df["feature"], df["coef"], color=colors, alpha=0.80)
            ax.axvline(0, color="black", lw=0.8)

            for _, row in df.iterrows():
                x = row["coef"]
                ax.text(
                    x + (0.01 if x >= 0 else -0.01),
                    df[df["feature"] == row["feature"]].index[
                        df["feature"].tolist().index(row["feature"])
                        if row["feature"] in df["feature"].tolist() else 0
                    ] if False else list(df["feature"]).index(row["feature"]),
                    f'OR={row["odds_ratio"]:.2f}',
                    va="center",
                    ha="left" if x >= 0 else "right",
                    fontsize=6.5,
                )

            ax.set_xlabel("Coeficiente logit (log-odds)", fontsize=9)
            ax.set_title(f"h = {h} año{'s' if h > 1 else ''}", fontsize=10)
            ax.invert_yaxis()
            ax.tick_params(labelsize=8)

        fig.suptitle(
            "Figura 33 — Coeficientes estimados del LogitPanel\n"
            "Rojo = ↑ riesgo · Azul = ↓ riesgo · OR = odds-ratio · "
            "Entrenamiento sobre muestra pre-1990",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig33_logit_coefficients_all_horizons.png")

    def _plot_loss_curves(self) -> None:
        """
        fig34 — Curvas L(μ=0.75, τ) para los tres clasificadores (h=3).
        Muestra la función de pérdida completa como función del umbral,
        revelando por qué el DDPM tiene un mínimo más desplazado.
        """
        if "LogitPanel" not in self._pred_base:
            return

        yt = self._pred_base["LogitPanel"]["y_true"].values
        thresholds = np.linspace(0.001, 0.60, 300)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel izquierdo: L(μ) completa
        ax = axes[0]
        for model, label, color, ls in zip(
            MODELS, LABELS, PALETTE, ["--", "-.", "-"]
        ):
            if model not in self._pred_base:
                continue
            yp = self._pred_base[model]["y_prob"].values
            losses, fnrs, fprs = [], [], []
            for thr in thresholds:
                yp_bin = (yp >= thr).astype(int)
                tp = ((yp_bin==1)&(yt==1)).sum()
                fn = ((yp_bin==0)&(yt==1)).sum()
                fp = ((yp_bin==1)&(yt==0)).sum()
                tn = ((yp_bin==0)&(yt==0)).sum()
                fnr = fn/(fn+tp) if (fn+tp)>0 else 0.
                fpr = fp/(fp+tn) if (fp+tn)>0 else 0.
                losses.append(MU_REF*fnr + (1-MU_REF)*fpr)
                fnrs.append(fnr); fprs.append(fpr)

            losses = np.array(losses)
            opt_idx = np.argmin(losses)
            ax.plot(thresholds, losses, color=color, lw=2, ls=ls,
                    label=f"{label}  (min={losses[opt_idx]:.4f} @ τ={thresholds[opt_idx]:.3f})")
            ax.axvline(thresholds[opt_idx], color=color, lw=0.8, ls=":", alpha=0.7)

        ax.set_xlabel("Umbral de probabilidad τ", fontsize=10)
        ax.set_ylabel(f"L(μ={MU_REF}, τ)", fontsize=10)
        ax.set_title(f"Función de pérdida L(μ={MU_REF}) vs τ", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.set_xlim(0, 0.60)

        # Panel derecho: zoom región [0, 0.15]
        ax2 = axes[1]
        thr_zoom = np.linspace(0.001, 0.15, 300)
        for model, label, color, ls in zip(
            MODELS, LABELS, PALETTE, ["--", "-.", "-"]
        ):
            if model not in self._pred_base:
                continue
            yp = self._pred_base[model]["y_prob"].values
            losses_z = []
            for thr in thr_zoom:
                yp_bin = (yp >= thr).astype(int)
                tp = ((yp_bin==1)&(yt==1)).sum()
                fn = ((yp_bin==0)&(yt==1)).sum()
                fp = ((yp_bin==1)&(yt==0)).sum()
                tn = ((yp_bin==0)&(yt==0)).sum()
                fnr = fn/(fn+tp) if (fn+tp)>0 else 0.
                fpr = fp/(fp+tn) if (fp+tn)>0 else 0.
                losses_z.append(MU_REF*fnr + (1-MU_REF)*fpr)
            losses_z = np.array(losses_z)
            opt_idx  = np.argmin(losses_z)
            ax2.plot(thr_zoom, losses_z, color=color, lw=2, ls=ls, label=label)
            ax2.axvline(thr_zoom[opt_idx], color=color, lw=0.8, ls=":", alpha=0.7)

        ax2.set_xlabel("Umbral de probabilidad τ", fontsize=10)
        ax2.set_ylabel(f"L(μ={MU_REF}, τ)", fontsize=10)
        ax2.set_title("Zoom: región τ ∈ [0.001, 0.15]", fontsize=10)
        ax2.legend(fontsize=7.5)
        ax2.set_xlim(0, 0.15)

        fig.suptitle(
            f"Figura 34 — Función de pérdida asimétrica L(μ={MU_REF}) como función del umbral τ\n"
            f"(h={H_SENS}, OOS 1990–2018). La línea de puntos vertical marca τ* de cada modelo.",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig34_loss_curves_vs_threshold.png")

    # ------------------------------------------------------------------
    # Utilidades privadas
    # ------------------------------------------------------------------

    def _get_predictions(
        self,
        C_val:       float,
        models_only: list[str] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Ejecuta el walk-forward con el C especificado y devuelve
        un diccionario {nombre_modelo → DataFrame de predicciones OOS}.
        """
        ev = ComparativeEvaluator(
            panel        = self.panel,
            latent_df    = self.latent_df,
            syn_vae      = self.syn_vae,
            syn_ddpm     = self.syn_ddpm,
            feature_cols = self.feature_cols,
            horizons     = [H_SENS],
            eval_start   = self.eval_start,
            eval_end     = self.eval_end,
            C_logit      = C_val,
        )
        ev.run(verbose=False)
        pred = {}
        selected = models_only or MODELS
        for m in selected:
            if m in ev.pred_dfs_ and H_SENS in ev.pred_dfs_[m]:
                pred[m] = ev.pred_dfs_[m][H_SENS]
        return pred

    @staticmethod
    def _opt_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        mu:     float,
    ) -> tuple[float, float, float, float]:
        """Retorna (L_min, τ*, FNR, FPR) para la μ dada."""
        fpr_arr, tpr_arr, thrs = roc_curve(y_true, y_prob)
        fnr_arr = 1.0 - tpr_arr
        losses  = mu * fnr_arr + (1 - mu) * fpr_arr
        idx     = np.argmin(losses)
        return (float(losses[idx]), float(thrs[idx]),
                float(fnr_arr[idx]), float(fpr_arr[idx]))

    def _save_csv(self, df: pd.DataFrame, name: str) -> None:
        path = OUTPUT_DIR / "results" / name
        df.to_csv(path, index=False)
        print(f"    CSV → {name}")

    def _save(self, fig: plt.Figure, name: str) -> None:
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ {name}")
