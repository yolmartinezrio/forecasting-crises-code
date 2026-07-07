"""
=============================================================================
evaluation/comparative.py
=============================================================================
Clase ComparativeEvaluator — evaluación walk-forward comparativa de los
tres clasificadores híbridos del proyecto.

Clasificadores evaluados
------------------------
  1. LogitPanel        : logit de panel con efectos fijos (baseline)
  2. LogitVAE          : logit+VAE (augmentación + indicadores latentes)
  3. LogitDDPM         : logit+DDPM (augmentación de alta fidelidad)

Diferencias entre clasificadores
---------------------------------
  LogitPanel  → entrena sobre el panel real exclusivamente.
  LogitVAE    → (a) añade las 4 dimensiones latentes del VAE como features
                    adicionales al vector de predictores (concatenación);
                (b) aumenta el conjunto de entrenamiento con los datos
                    sintéticos de pre-crisis generados por el VAE.
  LogitDDPM   → aumenta el conjunto de entrenamiento con los datos
                    sintéticos de pre-crisis generados por el DDPM
                    (sin indicadores latentes adicionales).

Protocolo de evaluación
------------------------
  Walk-forward con ventana expansiva (expanding window) sobre 1990-2018.
  Crisis window exclusion: se excluyen del entrenamiento las observaciones
  con crisis activa (crisis_bin=1) para evitar contaminación de la señal.
  Métricas: AUROC, AUPRC, L(μ=0.75) con IC 95% por block bootstrap.

Notas de implementación
-----------------------
  Para cada año t del período OOS el clasificador se reentrena desde cero
  con todos los datos disponibles hasta t-1. Los datos sintéticos se añaden
  al conjunto de entrenamiento de cada ventana como observaciones adicionales
  etiquetadas como pre-crisis (no se usan en el conjunto de prueba).
  La fusión de representaciones latentes (logit+VAE) se realiza concatenando
  las columnas z1-z4 a las features del panel en el conjunto de entrenamiento
  y de prueba; las observaciones sin representación latente disponible se
  imputan con la media de las representaciones latentes del conjunto de
  entrenamiento de cada ventana.
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import (
    FORECAST_HORIZONS, OUTPUT_DIR, DATA_OUT, FIGURES_DIR,
    COLORS, PLOT_STYLE, PLOT_FONT_SCALE, FIGURE_DPI,
)
from models.logit_panel     import LogitPanel, LOGIT_FEATURES
from evaluation.metrics     import EvaluationMetrics, EvaluationReport
from evaluation.walkforward import EVAL_START_YEAR, EVAL_END_YEAR, MIN_TRAIN_POS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

LATENT_COLS  = ["z1", "z2", "z3", "z4"]   # columnas del espacio latente VAE
MU           = 0.75                         # parámetro asimétrico Alessi-Detken
N_BOOTSTRAP  = 1000
BLOCK_SIZE   = 5
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ComparativeEvaluator:
    """
    Ejecuta el protocolo walk-forward para los tres clasificadores y
    produce el informe comparativo completo.

    Parámetros
    ----------
    panel : pd.DataFrame
        Panel maestro (output de DataPreprocessor).
    latent_df : pd.DataFrame
        Representaciones latentes del VAE (columnas: country, year, z1-z4).
    syn_vae : pd.DataFrame
        Datos sintéticos de pre-crisis generados por el VAE.
    syn_ddpm : pd.DataFrame
        Datos sintéticos de pre-crisis generados por el DDPM.
    feature_cols : list[str]
        Features base del clasificador logit.
    horizons : list[int]
        Horizontes de predicción a evaluar.
    eval_start / eval_end : int
        Período de evaluación OOS.
    C_logit : float
        Parámetro de regularización L2 del logit.
    """

    def __init__(
        self,
        panel:        pd.DataFrame,
        latent_df:    pd.DataFrame,
        syn_vae:      pd.DataFrame,
        syn_ddpm:     pd.DataFrame,
        feature_cols: list[str]  = None,
        horizons:     list[int]  = None,
        eval_start:   int        = EVAL_START_YEAR,
        eval_end:     int        = EVAL_END_YEAR,
        C_logit:      float      = 0.1,
    ) -> None:
        self.panel       = panel.copy()
        self.latent_df   = latent_df.copy()
        self.syn_vae     = syn_vae.copy()
        self.syn_ddpm    = syn_ddpm.copy()
        self.feature_cols= feature_cols or [f for f in LOGIT_FEATURES
                                            if f in panel.columns]
        self.horizons    = horizons or FORECAST_HORIZONS
        self.eval_start  = eval_start
        self.eval_end    = eval_end
        self.C           = C_logit
        self.metrics_    = EvaluationMetrics(
            mu=MU, n_bootstrap=N_BOOTSTRAP,
            block_size=BLOCK_SIZE, random_state=RANDOM_STATE,
        )

        # Enriquecer panel con representaciones latentes (merge left)
        self.panel_with_latent = self.panel.merge(
            self.latent_df[["country","year"] + LATENT_COLS],
            on=["country","year"], how="left",
        )

        # Resultados (se pueblan en run)
        self.reports_:    dict[str, list[EvaluationReport]] = {}
        self.pred_dfs_:   dict[str, dict[int, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> None:
        """Ejecuta el walk-forward para los tres clasificadores."""
        print("\n[ComparativeEvaluator] Iniciando evaluación comparativa…")
        print(f"  Período OOS : {self.eval_start}–{self.eval_end}")
        print(f"  Horizontes  : {self.horizons}")
        print(f"  μ           : {MU}  (Alessi & Detken, 2011)")

        for name, fn in [
            ("LogitPanel", self._run_logit),
            ("LogitVAE",   self._run_logit_vae),
            ("LogitDDPM",  self._run_logit_ddpm),
        ]:
            print(f"\n{'─'*55}")
            print(f"  Clasificador: {name}")
            print(f"{'─'*55}")
            reports, pred_dfs = fn(verbose=verbose)
            self.reports_[name]  = reports
            self.pred_dfs_[name] = pred_dfs

        print("\n[ComparativeEvaluator] Evaluación completada.")

    def summary_table(self) -> pd.DataFrame:
        """
        Construye la tabla comparativa de métricas para todos los
        clasificadores y horizontes.

        Retorna
        -------
        pd.DataFrame con columnas:
            model, horizon, auroc, auroc_ci_low, auroc_ci_high,
            auprc, auprc_ci_low, auprc_ci_high,
            loss, loss_ci_low, loss_ci_high,
            optimal_threshold, fnr, fpr, n_pos, n_total
        """
        rows = []
        for name, reports in self.reports_.items():
            for r in reports:
                rows.append(r.to_dict())
        return pd.DataFrame(rows).sort_values(["horizon","model"])

    def save(self) -> Path:
        """Guarda la tabla comparativa en CSV."""
        results_dir = OUTPUT_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        df   = self.summary_table()
        path = results_dir / "comparative_walkforward.csv"
        df.to_csv(path, index=False)
        print(f"  Tabla comparativa → {path.name}")
        return path

    def plot_all(self) -> None:
        """Genera todas las figuras del Módulo 5."""
        sns.set_theme(style=PLOT_STYLE, font_scale=PLOT_FONT_SCALE)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self._plot_auroc_comparison()
        self._plot_auprc_comparison()
        self._plot_loss_comparison()
        self._plot_roc_overlay()
        self._plot_pr_overlay()
        self._plot_prob_timeseries()

    # ------------------------------------------------------------------
    # Walk-forward por clasificador
    # ------------------------------------------------------------------

    def _run_logit(self, verbose=True):
        """LogitPanel: solo datos reales del panel."""
        return self._walk(
            name        = "LogitPanel",
            use_latent  = False,
            syn_df      = None,
            verbose     = verbose,
        )

    def _run_logit_vae(self, verbose=True):
        """
        LogitVAE:
          - Features aumentadas con z1-z4 (concatenación).
          - Datos de entrenamiento aumentados con sintéticos VAE.
        """
        return self._walk(
            name        = "LogitVAE",
            use_latent  = True,
            syn_df      = self.syn_vae,
            verbose     = verbose,
        )

    def _run_logit_ddpm(self, verbose=True):
        """
        LogitDDPM:
          - Solo features originales (sin indicadores latentes).
          - Datos de entrenamiento aumentados con sintéticos DDPM.
        """
        return self._walk(
            name        = "LogitDDPM",
            use_latent  = False,
            syn_df      = self.syn_ddpm,
            verbose     = verbose,
        )

    # ------------------------------------------------------------------
    # Bucle walk-forward genérico
    # ------------------------------------------------------------------

    def _walk(
        self,
        name:       str,
        use_latent: bool,
        syn_df:     pd.DataFrame | None,
        verbose:    bool = True,
    ) -> tuple[list[EvaluationReport], dict[int, pd.DataFrame]]:
        """
        Itera año a año en el período OOS reentrenando el clasificador
        con todos los datos disponibles hasta t-1 y evaluando en t.

        Parámetros
        ----------
        use_latent : bool
            Si True, concatena z1-z4 a las features del clasificador.
        syn_df : pd.DataFrame | None
            Datos sintéticos a añadir al conjunto de entrenamiento.
            Si None, no se usa aumento de datos.
        """
        # Panel base: con latentes si use_latent=True
        base_panel = (self.panel_with_latent
                      if use_latent else self.panel.copy())

        # Features efectivas
        feat_cols = list(self.feature_cols)
        if use_latent:
            feat_cols = feat_cols + LATENT_COLS

        # Sintéticos: garantizar que tienen las features necesarias
        syn_ready = None
        if syn_df is not None:
            syn_ready = self._prepare_synthetic(syn_df, feat_cols)

        reports, pred_dfs = [], {}

        for h in self.horizons:
            target = f"crisis_h{h}"
            if target not in base_panel.columns:
                continue

            y_true_all, y_prob_all, years_all, country_all = [], [], [], []

            for t in range(self.eval_start, self.eval_end + 1):

                # ── Conjunto de entrenamiento ────────────────────────
                tr_mask = (
                    (base_panel["year"] < t) &
                    (base_panel["crisis_bin"] == 0) &
                    base_panel[target].notna() &
                    base_panel[feat_cols].notna().any(axis=1)
                )
                tr_df = base_panel[tr_mask].copy()

                # Imputar latentes faltantes con mediana del entrenamiento
                if use_latent:
                    for zc in LATENT_COLS:
                        med = tr_df[zc].median()
                        tr_df[zc] = tr_df[zc].fillna(
                            med if not np.isnan(med) else 0.0)

                # Añadir datos sintéticos
                if syn_ready is not None and len(syn_ready) > 0:
                    tr_df = self._augment(tr_df, syn_ready, feat_cols, target)

                if tr_df[target].sum() < MIN_TRAIN_POS:
                    continue

                X_tr = tr_df[feat_cols]
                y_tr = tr_df[target]
                c_tr = tr_df["country"]

                # ── Conjunto de prueba ───────────────────────────────
                te_mask = (
                    (base_panel["year"] == t) &
                    base_panel[target].notna() &
                    base_panel[feat_cols].notna().any(axis=1)
                )
                te_df = base_panel[te_mask].copy()
                if use_latent:
                    for zc in LATENT_COLS:
                        med = tr_df[zc].median()
                        te_df[zc] = te_df[zc].fillna(
                            med if not np.isnan(med) else 0.0)

                if len(te_df) == 0:
                    continue

                X_te = te_df[feat_cols]
                y_te = te_df[target]
                c_te = te_df["country"]

                # ── Entrenar y predecir ──────────────────────────────
                try:
                    model = LogitPanel(
                        horizon    = h,
                        features   = feat_cols,
                        C          = self.C,
                        country_fe = True,
                    )
                    model.fit(X_tr, y_tr, c_tr)
                    probs = model.predict_proba(X_te, c_te)
                except Exception as e:
                    if verbose:
                        print(f"    ✗ t={t}, h={h}: {e}")
                    continue

                y_true_all.append(y_te.values)
                y_prob_all.append(probs)
                years_all.append(np.full(len(y_te), t))
                country_all.append(c_te.values)

                if verbose and int(y_te.sum()) > 0:
                    print(f"    {name} t={t} h={h}: "
                          f"train={len(tr_df)}({int(tr_df[target].sum())}+) "
                          f"test={len(te_df)}({int(y_te.sum())}+) ✓")

            if not y_true_all:
                continue

            yt = np.concatenate(y_true_all)
            yp = np.concatenate(y_prob_all)
            yr = np.concatenate(years_all)
            yc = np.concatenate(country_all)

            if yt.sum() < 2:
                continue

            report = self.metrics_.evaluate(yt, yp, name, h, yr)
            reports.append(report)
            print(report.summary())

            pred_dfs[h] = pd.DataFrame({
                "year": yr, "country": yc,
                "y_true": yt, "y_prob": yp, "model": name,
            })

        return reports, pred_dfs

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def _prepare_synthetic(
        self,
        syn_df:    pd.DataFrame,
        feat_cols: list[str],
    ) -> pd.DataFrame:
        """
        Alinea el DataFrame sintético con las features requeridas.
        Las features ausentes se imputan con la mediana del panel real.
        Las columnas de crisis se garantizan.
        """
        df = syn_df.copy()
        for fc in feat_cols:
            if fc not in df.columns:
                # Imputar con mediana del panel real
                med = self.panel[fc].median() if fc in self.panel.columns else 0.0
                df[fc] = med if not np.isnan(med) else 0.0
        for col in ["crisis_bin","crisis_h1","crisis_h2","crisis_h3"]:
            if col not in df.columns:
                df[col] = 1 if col != "crisis_bin" else 0
        if "country" not in df.columns:
            df["country"] = "SYNTHETIC"
        if "year" not in df.columns:
            df["year"] = -1
        return df

    def _augment(
        self,
        tr_df:     pd.DataFrame,
        syn_df:    pd.DataFrame,
        feat_cols: list[str],
        target:    str,
    ) -> pd.DataFrame:
        """
        Concatena los datos sintéticos al conjunto de entrenamiento.
        Solo toma las columnas necesarias del DataFrame sintético.
        """
        needed = feat_cols + [target, "crisis_bin", "country"]
        needed = [c for c in needed if c in syn_df.columns]
        syn_sub = syn_df[needed].copy()

        # Garantizar que el target existe y vale 1
        syn_sub[target] = 1
        syn_sub["crisis_bin"] = 0

        return pd.concat([tr_df, syn_sub], ignore_index=True)

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------

    def _plot_auroc_comparison(self) -> None:
        """
        Figura 23 — AUROC por horizonte y clasificador con IC 95%.
        """
        df = self.summary_table()
        if df.empty:
            return

        fig, axes = plt.subplots(1, len(self.horizons),
                                 figsize=(5*len(self.horizons), 5),
                                 sharey=True)
        if len(self.horizons) == 1:
            axes = [axes]

        models   = ["LogitPanel","LogitVAE","LogitDDPM"]
        palette  = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]
        x_pos    = np.arange(len(models))

        for ax, h in zip(axes, self.horizons):
            sub = df[df["horizon"] == h]
            for xi, (m, c) in enumerate(zip(models, palette)):
                row = sub[sub["model"] == m]
                if row.empty:
                    continue
                row = row.iloc[0]
                bar = ax.bar(xi, row["auroc"], color=c, alpha=0.80, width=0.6)
                ax.errorbar(
                    xi, row["auroc"],
                    yerr=[[row["auroc"]-row["auroc_ci_low"]],
                          [row["auroc_ci_high"]-row["auroc"]]],
                    fmt="none", color="black", capsize=5, lw=1.5,
                )
                ax.text(xi, row["auroc"] + 0.01,
                        f'{row["auroc"]:.3f}', ha="center",
                        fontsize=8, fontweight="bold")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(["Logit\n(baseline)","Logit\n+VAE","Logit\n+DDPM"],
                               fontsize=9)
            ax.set_ylim(0, 1)
            ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.6)
            ax.set_title(f"h = {h} año{'s' if h>1 else ''}", fontsize=10)
            ax.set_ylabel("AUROC" if h == self.horizons[0] else "")

        fig.suptitle(
            "Figura 23 — AUROC fuera de muestra (1990–2018) por clasificador\n"
            "Barras de error = IC 95% (block bootstrap, B=1.000)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig23_auroc_comparison.png")

    def _plot_auprc_comparison(self) -> None:
        """Figura 24 — AUPRC por horizonte y clasificador con IC 95%."""
        df = self.summary_table()
        if df.empty:
            return

        fig, axes = plt.subplots(1, len(self.horizons),
                                 figsize=(5*len(self.horizons), 5),
                                 sharey=True)
        if len(self.horizons) == 1:
            axes = [axes]

        models  = ["LogitPanel","LogitVAE","LogitDDPM"]
        palette = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]

        for ax, h in zip(axes, self.horizons):
            sub = df[df["horizon"] == h]
            prevalence = (self.panel[f"crisis_h{h}"].sum()
                          / self.panel[f"crisis_h{h}"].notna().sum())
            for xi, (m, c) in enumerate(zip(models, palette)):
                row = sub[sub["model"] == m]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.bar(xi, row["auprc"], color=c, alpha=0.80, width=0.6)
                ax.errorbar(
                    xi, row["auprc"],
                    yerr=[[max(0,row["auprc"]-row["auprc_ci_low"])],
                          [row["auprc_ci_high"]-row["auprc"]]],
                    fmt="none", color="black", capsize=5, lw=1.5,
                )
                ax.text(xi, row["auprc"] + 0.005,
                        f'{row["auprc"]:.3f}', ha="center",
                        fontsize=8, fontweight="bold")

            ax.axhline(prevalence, color="grey", lw=1.0, ls="--",
                       label=f"Aleatorio={prevalence:.3f}")
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(["Logit\n(baseline)","Logit\n+VAE","Logit\n+DDPM"],
                               fontsize=9)
            ax.set_ylim(0, max(0.6, df[df["horizon"]==h]["auprc_ci_high"].max()+0.05))
            ax.set_title(f"h = {h} año{'s' if h>1 else ''}", fontsize=10)
            ax.set_ylabel("AUPRC" if h == self.horizons[0] else "")
            ax.legend(fontsize=7, loc="upper right")

        fig.suptitle(
            "Figura 24 — AUPRC fuera de muestra (1990–2018) por clasificador\n"
            "Línea discontinua = clasificador aleatorio (prevalencia de la clase positiva)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig24_auprc_comparison.png")

    def _plot_loss_comparison(self) -> None:
        """Figura 25 — L(μ=0.75) por horizonte y clasificador."""
        df = self.summary_table()
        if df.empty:
            return

        fig, axes = plt.subplots(1, len(self.horizons),
                                 figsize=(5*len(self.horizons), 5),
                                 sharey=True)
        if len(self.horizons) == 1:
            axes = [axes]

        models  = ["LogitPanel","LogitVAE","LogitDDPM"]
        palette = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]

        for ax, h in zip(axes, self.horizons):
            sub = df[df["horizon"] == h]
            for xi, (m, c) in enumerate(zip(models, palette)):
                row = sub[sub["model"] == m]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.bar(xi, row["loss"], color=c, alpha=0.80, width=0.6)
                ax.errorbar(
                    xi, row["loss"],
                    yerr=[[max(0,row["loss"]-row["loss_ci_low"])],
                          [row["loss_ci_high"]-row["loss"]]],
                    fmt="none", color="black", capsize=5, lw=1.5,
                )
                ax.text(xi, row["loss"] + 0.004,
                        f'{row["loss"]:.3f}', ha="center",
                        fontsize=8, fontweight="bold")

            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(["Logit\n(baseline)","Logit\n+VAE","Logit\n+DDPM"],
                               fontsize=9)
            ax.set_ylim(0, 0.5)
            ax.set_title(f"h = {h} año{'s' if h>1 else ''}", fontsize=10)
            ax.set_ylabel("L(μ=0.75)" if h == self.horizons[0] else "")

        fig.suptitle(
            "Figura 25 — Pérdida asimétrica L(μ=0.75) por clasificador\n"
            "Menor valor = mejor rendimiento · μ=0.75 prioriza crisis no detectadas",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, "fig25_loss_comparison.png")

    def _plot_roc_overlay(self) -> None:
        """Figura 26 — Curvas ROC superpuestas para h=3."""
        h = max(self.horizons)
        fig, ax = plt.subplots(figsize=(6, 6))

        models  = ["LogitPanel","LogitVAE","LogitDDPM"]
        labels  = ["Logit baseline","Logit+VAE","Logit+DDPM"]
        palette = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]
        lss     = ["--", "-.", "-"]

        from sklearn.metrics import roc_curve, roc_auc_score

        for m, lbl, c, ls in zip(models, labels, palette, lss):
            if m not in self.pred_dfs_ or h not in self.pred_dfs_[m]:
                continue
            pdf = self.pred_dfs_[m][h]
            try:
                fpr, tpr, _ = roc_curve(pdf["y_true"], pdf["y_prob"])
                auc = roc_auc_score(pdf["y_true"], pdf["y_prob"])
                ax.plot(fpr, tpr, color=c, lw=2, ls=ls,
                        label=f"{lbl}  (AUROC={auc:.3f})")
                ax.fill_between(fpr, tpr, alpha=0.06, color=c)
            except Exception:
                pass

        ax.plot([0,1],[0,1], "k--", lw=0.8, alpha=0.5, label="Aleatorio")
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=10)
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=10)
        ax.set_title(
            f"Figura 26 — Curvas ROC superpuestas (h={h} años)\n"
            "Período OOS 1990–2018",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="lower right")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        plt.tight_layout()
        self._save(fig, f"fig26_roc_overlay_h{h}.png")

    def _plot_pr_overlay(self) -> None:
        """Figura 27 — Curvas Precisión-Recall superpuestas para h=3."""
        h = max(self.horizons)
        fig, ax = plt.subplots(figsize=(6, 6))

        from sklearn.metrics import precision_recall_curve, average_precision_score

        models  = ["LogitPanel","LogitVAE","LogitDDPM"]
        labels  = ["Logit baseline","Logit+VAE","Logit+DDPM"]
        palette = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]
        lss     = ["--","-.","- "]

        for m, lbl, c, ls in zip(models, labels, palette, lss):
            if m not in self.pred_dfs_ or h not in self.pred_dfs_[m]:
                continue
            pdf = self.pred_dfs_[m][h]
            try:
                prec, rec, _ = precision_recall_curve(
                    pdf["y_true"], pdf["y_prob"])
                ap = average_precision_score(pdf["y_true"], pdf["y_prob"])
                ax.plot(rec, prec, color=c, lw=2,
                        label=f"{lbl}  (AUPRC={ap:.3f})")
            except Exception:
                pass

        base = self.panel[f"crisis_h{h}"].mean()
        ax.axhline(base, color="grey", lw=0.9, ls=":",
                   label=f"Aleatorio (prevalencia={base:.3f})")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precisión", fontsize=10)
        ax.set_title(
            f"Figura 27 — Curvas Precisión-Recall superpuestas (h={h} años)\n"
            "Período OOS 1990–2018",
            fontsize=11,
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        plt.tight_layout()
        self._save(fig, f"fig27_pr_overlay_h{h}.png")

    def _plot_prob_timeseries(self) -> None:
        """
        Figura 28 — Probabilidades predichas OOS por los tres modelos
        para 4 países con episodios de crisis en el período OOS.
        """
        h = 2    # horizonte intermedio: mejor relación señal/ruido
        countries_with_crisis = []
        for m in ["LogitPanel","LogitVAE","LogitDDPM"]:
            if m in self.pred_dfs_ and h in self.pred_dfs_[m]:
                pdf = self.pred_dfs_[m][h]
                cands = (pdf[pdf["y_true"]==1]["country"]
                         .value_counts().head(4).index.tolist())
                countries_with_crisis = cands[:4]
                break

        if not countries_with_crisis:
            return

        n   = len(countries_with_crisis)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3.5*n), sharex=False)
        if n == 1:
            axes = [axes]

        models  = ["LogitPanel","LogitVAE","LogitDDPM"]
        labels  = ["Logit","Logit+VAE","Logit+DDPM"]
        palette = [COLORS["neutral"], COLORS["no_crisis"], COLORS["crisis"]]
        lss     = ["--","-.","- "]

        for ax, country in zip(axes, countries_with_crisis):
            for m, lbl, c, ls in zip(models, labels, palette, lss):
                if m not in self.pred_dfs_ or h not in self.pred_dfs_[m]:
                    continue
                pdf = (self.pred_dfs_[m][h]
                       [self.pred_dfs_[m][h]["country"]==country]
                       .sort_values("year"))
                if pdf.empty:
                    continue
                ax.plot(pdf["year"], pdf["y_prob"],
                        color=c, lw=1.6, ls=ls, label=lbl)

            # Sombrear crisis activas
            crisis_yrs = self.panel[
                (self.panel["country"]==country) &
                (self.panel["crisis_bin"]==1)
            ]["year"]
            for yr in crisis_yrs:
                ax.axvspan(yr-0.5, yr+0.5,
                           color=COLORS["crisis"], alpha=0.25, lw=0)

            ax.set_ylim(0, 1)
            ax.set_ylabel("P(crisis)", fontsize=8)
            ax.set_title(country, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=7)
            if ax == axes[0]:
                ax.legend(fontsize=7, loc="upper left", ncol=3)

        axes[-1].set_xlabel("Año")
        fig.suptitle(
            f"Figura 28 — Probabilidades predichas OOS (h={h} años) por los tres clasificadores\n"
            "Áreas rojas = años de crisis activa (LV 2018)  ·  Período 1990–2018",
            fontsize=11,
        )
        plt.tight_layout()
        self._save(fig, f"fig28_prob_timeseries_h{h}.png")

    def _save(self, fig, name: str) -> None:
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {name}")
