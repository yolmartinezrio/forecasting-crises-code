"""
main.py — Punto de entrada del proyecto. Módulos 1-4.
Uso: python main.py [--module {1,2,3,4,all}]
"""
import sys, argparse
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import FORECAST_HORIZONS, DATA_OUT

def run_module1():
    from data.loader import DataLoader
    from data.preprocessor import DataPreprocessor
    from eda.plotter import EDAPlotter
    print("\n" + "="*65 + "\n  MÓDULO 1 — EDA\n" + "="*65)
    loader = DataLoader(); loader.load_all()
    prep = DataPreprocessor(loader.jst_raw, loader.lv_long); prep.build()
    EDAPlotter(prep.panel_master, prep.feature_list).plot_all()
    return prep

def run_module2(prep):
    from models.logit_panel import LogitPanel, LOGIT_FEATURES
    from evaluation.walkforward import WalkForwardEvaluator
    from evaluation.plotter import ResultsPlotter
    print("\n" + "="*65 + "\n  MÓDULO 2 — LOGIT BASELINE\n" + "="*65)
    panel = prep.panel_master
    feature_cols = [f for f in LOGIT_FEATURES if f in panel.columns]
    ev = WalkForwardEvaluator(panel=panel, feature_cols=feature_cols,
        horizons=FORECAST_HORIZONS, mu=0.75, n_bootstrap=1000, random_state=42)
    reports = ev.run(model_factory=lambda: LogitPanel(C=0.1, country_fe=True),
                     model_name="LogitPanel")
    ev.print_summary(reports); ev.save_results(reports, "LogitPanel")
    pl = ResultsPlotter("LogitPanel")
    pred_dfs = {h: ev.get_predictions_df(
        model_factory=lambda h=h: LogitPanel(horizon=h,C=0.1,country_fe=True),
        model_name="LogitPanel", horizon=h) for h in FORECAST_HORIZONS}
    pl.plot_roc_curves(pred_dfs, reports)
    pl.plot_pr_curves(pred_dfs, reports)
    pl.plot_loss_vs_threshold(pred_dfs, mu=0.75)
    train_full = panel[panel["year"] < 1990].copy()
    for h in FORECAST_HORIZONS:
        tgt = f"crisis_h{h}"
        sub = train_full[train_full[tgt].notna()&(train_full["crisis_bin"]==0)]
        if sub[tgt].sum() >= 5:
            avail = [f for f in LOGIT_FEATURES if f in sub.columns]
            m = LogitPanel(horizon=h, C=0.1, country_fe=True)
            m.fit(sub[avail], sub[tgt], sub["country"])
            pl.plot_coefficients(m.get_coef_df(), top_n=15, horizon=h)
            if h==1 and h in pred_dfs:
                pl.plot_predicted_probs(pred_dfs[h], panel, horizon=h)

def run_module3(prep):
    from models.vae import VAEPipeline, VAE_FEATURES
    print("\n" + "="*65 + "\n  MÓDULO 3 — VAE TEMPORAL\n" + "="*65)
    VAEPipeline(panel=prep.panel_master, feature_cols=VAE_FEATURES,
        latent_dim=4, window_size=5, hidden_dim=32, epochs=300,
        train_cutoff=1990, n_synthetic=200, random_state=42).run(verbose=True)

def run_module4(prep):
    from models.ddpm import DDPMPipeline, DDPM_FEATURES
    print("\n" + "="*65 + "\n  MÓDULO 4 — DDPM\n" + "="*65)
    DDPMPipeline(panel=prep.panel_master, feature_cols=DDPM_FEATURES,
        T=200, hidden_dim=64, epochs=300, train_cutoff=1990,
        n_synthetic=200, noise_level=0.5,
        vae_synthetic_path=DATA_OUT/"vae_synthetic_precrisis.csv",
        random_state=42).run(verbose=True)

def run_module5(prep):
    from models.logit_panel import LOGIT_FEATURES
    from evaluation.sensitivity import SensitivityAnalyzer
    import pandas as pd
    from config.settings import DATA_OUT

    print("\n" + "="*65 + "\n  MÓDULO 5 — ANÁLISIS DE SENSIBILIDAD E INTERPRETABILIDAD\n" + "="*65)
    lat      = pd.read_csv(DATA_OUT / "vae_latent_representations.csv")
    syn_vae  = pd.read_csv(DATA_OUT / "vae_synthetic_precrisis.csv")
    syn_ddpm = pd.read_csv(DATA_OUT / "ddpm_synthetic_precrisis.csv")
    feat_cols = [f for f in LOGIT_FEATURES if f in prep.panel_master.columns]

    SensitivityAnalyzer(
        panel=prep.panel_master, latent_df=lat,
        syn_vae=syn_vae, syn_ddpm=syn_ddpm,
        feature_cols=feat_cols, eval_start=1990, eval_end=2018,
    ).run(verbose=True)

def _load_prep():
    import pandas as pd
    class _P:
        def __init__(self, path):
            self.panel_master = pd.read_csv(path)
            skip = {"year","country","iso","crisis_bin","crisis_lv",
                    "crisisJST","crisis_union","crisis_h1","crisis_h2","crisis_h3"}
            self.feature_list=[c for c in self.panel_master.columns if c not in skip]
    path = DATA_OUT/"panel_maestro.csv"
    if not path.exists(): raise FileNotFoundError(f"Panel no encontrado: {path}")
    return _P(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["1","2","3","4","5","all"], default="all")
    args = parser.parse_args()
    print("="*65 + "\n  FUSIÓN DE MODELOS GENERATIVOS Y CUANTITATIVOS\n" + "="*65)
    prep = None
    if args.module in ("1","all"): prep = run_module1()
    if args.module in ("2","3","4","5","all"):
        if prep is None: prep = _load_prep()
    if args.module in ("2","all"): run_module2(prep)
    if args.module in ("3","all"): run_module3(prep)
    if args.module in ("4","all"): run_module4(prep)
    if args.module in ("5","all"): run_module5(prep)
    print("\n" + "="*65 + f"\n  Pipeline completado. Outputs: {PROJECT_ROOT/'outputs'}\n" + "="*65)

if __name__ == "__main__":
    main()
