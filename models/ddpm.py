"""
=============================================================================
models/ddpm.py
=============================================================================
Clase DDPM — Denoising Diffusion Probabilistic Model para series temporales
macro-financieras de panel.

Fundamento teórico
------------------
El DDPM (Ho et al., 2020) define un proceso de difusión hacia adelante
que degrada progresivamente los datos añadiendo ruido gaussiano en T pasos:

  q(x_τ | x_{τ-1}) = N(x_τ; √(1-β_τ)·x_{τ-1},  β_τ·I)

donde {β_τ}_{τ=1}^T es una agenda de ruido lineal predefinida. Una red
de denoising ε_θ(x_τ, τ) se entrena para predecir el ruido añadido:

  L_simple = E_{x_0, ε, τ} [ ||ε − ε_θ(√ᾱ_τ·x_0 + √(1−ᾱ_τ)·ε, τ)||² ]

La generación parte de ruido puro x_T ~ N(0,I) y revierte el proceso
iterativamente usando ε_θ en cada paso.

Ventajas sobre el VAE
---------------------
1. Sin posterior collapse: no optimiza ELBO; la varianza de las muestras
   es estructuralmente controlada por la agenda de ruido.
2. Mayor fidelidad estadística: σ_syn/σ_real ≈ 0.75-1.37 frente a
   σ_syn ≈ 0 que producía el VAE (Takahashi & Mizuno, 2025).
3. Generación condicionada por difusión parcial: añade ruido hasta τ*
   sobre observaciones de referencia y revierte, produciendo variantes
   plausibles con diversidad controlada.

Implementación
--------------
Red de denoising: MLP de 3 capas ocultas con codificación posicional
sinusoidal del paso τ (Fourier features). Todo en NumPy puro + Adam
propio. Sin dependencias de PyTorch.

Referencias
-----------
Ho, J., Jain, A. & Abbeel, P. (2020). Denoising diffusion probabilistic
  models. NeurIPS 33, 6840-6851.
Takahashi, T. & Mizuno, T. (2025). Generation of synthetic financial
  time series by diffusion models. Quantitative Finance.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from config.settings import (
    OUTPUT_DIR, FIGURES_DIR, COLORS,
    PLOT_STYLE, PLOT_FONT_SCALE, FIGURE_DPI,
)

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------

DDPM_FEATURES: list[str] = [
    "tloans", "tmort", "lev", "ltd", "noncore",
    "hpnom", "housing_capgain", "eq_capgain",
    "stir", "ltrate", "bill_rate",
    "gdp", "iy", "ca", "cpi", "debtgdp", "money",
    "bond_rate", "housing_tr", "eq_tr",
    "tloans_gap", "hpnom_gap", "term_spread", "tloans_growth",
]

DEFAULT_T_STEPS    = 200
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END   = 0.02
DEFAULT_HIDDEN_DIM = 64
DEFAULT_EPOCHS     = 300
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 5e-4
DEFAULT_N_FOURIER  = 16
DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------

def _relu(x):      return np.maximum(0.0, x)
def _relu_g(x):    return (x > 0).astype(float)

def _sinusoidal_emb(tau: np.ndarray, n: int, mp: float = 10000.) -> np.ndarray:
    """
    Codificación posicional sinusoidal del paso temporal τ.
    Devuelve (N, 2n): concatenación de senos y cosenos a distintas frecuencias.
    Permite a la red distinguir el nivel de ruido en cada paso de difusión
    sin embeddings aprendidos (Vaswani et al., 2017).
    """
    freqs = np.exp(-np.log(mp) * np.arange(n) / n)
    args  = tau[:, None] * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)


# ---------------------------------------------------------------------------
# Red de denoising: MLP de 3 capas con codificación posicional
# ---------------------------------------------------------------------------

class _DenoiseMLP:
    """
    MLP de 3 capas ocultas para predecir el ruido ε_θ(x_τ, τ).
    Entrada: [x_τ || emb(τ)] ∈ ℝ^{d + 2·F}
    Salida : ε̂ ∈ ℝ^d
    """

    def __init__(self, d: int, h: int, F: int, rng: np.random.Generator):
        tin = d + 2 * F
        s1 = np.sqrt(2. / tin);  s2 = np.sqrt(2. / h);  sO = np.sqrt(1. / h)
        self.W1=rng.normal(0,s1,(tin,h)); self.b1=np.zeros(h)
        self.W2=rng.normal(0,s2,(h,h));   self.b2=np.zeros(h)
        self.W3=rng.normal(0,s2,(h,h));   self.b3=np.zeros(h)
        self.Wo=rng.normal(0,sO,(h,d));   self.bo=np.zeros(d)
        self._m={k:np.zeros_like(v) for k,v in self._p().items()}
        self._v={k:np.zeros_like(v) for k,v in self._p().items()}
        self._t=0

    def forward(self, xt, te):
        inp=np.concatenate([xt,te],1)
        h1=_relu(inp@self.W1+self.b1); h2=_relu(h1@self.W2+self.b2)
        h3=_relu(h2@self.W3+self.b3); out=h3@self.Wo+self.bo
        return out, (inp,h1,h2,h3)

    def backward(self, dout, cache):
        inp,h1,h2,h3=cache; n=inp.shape[0]
        dWo=h3.T@dout/n; dbo=dout.mean(0); dh3=dout@self.Wo.T
        dh3p=dh3*_relu_g(h3); dW3=h2.T@dh3p/n; db3=dh3p.mean(0); dh2=dh3p@self.W3.T
        dh2p=dh2*_relu_g(h2); dW2=h1.T@dh2p/n; db2=dh2p.mean(0); dh1=dh2p@self.W2.T
        dh1p=dh1*_relu_g(h1); dW1=inp.T@dh1p/n; db1=dh1p.mean(0)
        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,
                "W3":dW3,"b3":db3,"Wo":dWo,"bo":dbo}

    def adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
        self._t += 1
        for k,g in grads.items():
            self._m[k]=b1*self._m[k]+(1-b1)*g
            self._v[k]=b2*self._v[k]+(1-b2)*g**2
            mh=self._m[k]/(1-b1**self._t); vh=self._v[k]/(1-b2**self._t)
            getattr(self,k)[:] -= lr*mh/(np.sqrt(vh)+eps)

    def _p(self):
        return {"W1":self.W1,"b1":self.b1,"W2":self.W2,"b2":self.b2,
                "W3":self.W3,"b3":self.b3,"Wo":self.Wo,"bo":self.bo}


# ---------------------------------------------------------------------------
# Clase principal: DDPM
# ---------------------------------------------------------------------------

class DDPM:
    """
    Denoising Diffusion Probabilistic Model para datos tabulares.

    Parámetros
    ----------
    input_dim : int        Dimensión del vector de observación d.
    T : int                Pasos del proceso de difusión.
    beta_start/end : float Límites de la agenda de ruido lineal.
    hidden_dim : int       Neuronas ocultas de la red ε_θ.
    n_fourier : int        Componentes de la codificación posicional.
    learning_rate : float  Tasa de aprendizaje Adam.
    epochs : int           Épocas de entrenamiento.
    batch_size : int       Tamaño de mini-lote.
    random_state : int     Semilla.
    """

    def __init__(
        self,
        input_dim:     int   = 24,
        T:             int   = DEFAULT_T_STEPS,
        beta_start:    float = DEFAULT_BETA_START,
        beta_end:      float = DEFAULT_BETA_END,
        hidden_dim:    int   = DEFAULT_HIDDEN_DIM,
        n_fourier:     int   = DEFAULT_N_FOURIER,
        learning_rate: float = DEFAULT_LR,
        epochs:        int   = DEFAULT_EPOCHS,
        batch_size:    int   = DEFAULT_BATCH_SIZE,
        random_state:  int   = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.input_dim = input_dim
        self.T         = T
        self.hidden_dim= hidden_dim
        self.n_fourier = n_fourier
        self.lr        = learning_rate
        self.epochs    = epochs
        self.batch_size= batch_size
        self.rng       = np.random.default_rng(random_state)

        # Agenda de ruido lineal y coeficientes derivados
        self.betas      = np.linspace(beta_start, beta_end, T)
        self.alphas     = 1.0 - self.betas
        self.alpha_bar  = np.cumprod(self.alphas)      # ᾱ_τ
        self.sqrt_ab    = np.sqrt(self.alpha_bar)
        self.sqrt_1m_ab = np.sqrt(1.0 - self.alpha_bar)

        self._net = _DenoiseMLP(input_dim, hidden_dim, n_fourier, self.rng)
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None

        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # Proceso forward (forma cerrada): q(x_τ|x_0)
    # ------------------------------------------------------------------

    def _q_sample(self, x0, tau, noise):
        """
        Muestrea x_τ directamente desde x_0 en un paso:
            x_τ = √ᾱ_τ · x_0  +  √(1−ᾱ_τ) · ε
        """
        return (self.sqrt_ab[tau][:, None] * x0
                + self.sqrt_1m_ab[tau][:, None] * noise)

    # ------------------------------------------------------------------
    # Entrenamiento
    # ------------------------------------------------------------------

    def fit(
        self,
        X:           np.ndarray,
        val_frac:    float = 0.15,
        verbose:     bool  = True,
        verbose_every: int = 50,
    ) -> "DDPM":
        """
        Entrena ε_θ minimizando MSE entre ruido real y predicho.

        Parámetros
        ----------
        X : np.ndarray (N, d)  Observaciones de entrenamiento.
        """
        self._mean = X.mean(0, keepdims=True)
        self._std  = X.std(0,  keepdims=True) + 1e-8
        X_sc = (X - self._mean) / self._std

        N    = len(X_sc)
        nv   = max(1, int(N * val_frac))
        X_tr = X_sc[:-nv];  X_val = X_sc[-nv:]
        Nt   = len(X_tr)

        if verbose:
            print(f"  [DDPM] Entrenando: {Nt} train · {nv} val · "
                  f"d={self.input_dim} · T={self.T} · épocas={self.epochs}")

        for epoch in range(1, self.epochs + 1):
            idx = self.rng.permutation(Nt); ep_loss = []
            for s in range(0, Nt, self.batch_size):
                bi  = idx[s:s + self.batch_size]
                x0  = X_tr[bi]; nb = len(x0)
                tau = self.rng.integers(0, self.T, size=nb)
                eps = self.rng.standard_normal(x0.shape)
                xt  = self._q_sample(x0, tau, eps)
                te  = _sinusoidal_emb(tau.astype(float) + 1, self.n_fourier)
                pred, cache = self._net.forward(xt, te)
                diff = pred - eps;  loss = np.mean(diff ** 2)
                ep_loss.append(loss)
                dout = 2. * diff / (nb * self.input_dim)
                self._net.adam_step(self._net.backward(dout, cache), self.lr)

            vl = self._val_loss(X_val)
            self.train_losses_.append(float(np.mean(ep_loss)))
            self.val_losses_.append(vl)

            if verbose and epoch % verbose_every == 0:
                print(f"    Época {epoch:4d}/{self.epochs}  "
                      f"MSE_train={self.train_losses_[-1]:.6f}  "
                      f"MSE_val={vl:.6f}")

        self._fitted = True
        if verbose:
            print(f"  [DDPM] Entrenamiento completado. "
                  f"MSE_val final = {self.val_losses_[-1]:.6f}")
        return self

    # ------------------------------------------------------------------
    # Generación: proceso reverse
    # ------------------------------------------------------------------

    def generate(self, n: int = 100) -> np.ndarray:
        """
        Genera n observaciones desde ruido puro x_T ~ N(0,I)
        aplicando T pasos del proceso de difusión inversa.
        Retorna np.ndarray (n, d) en escala original.
        """
        self._check_fitted()
        x = self.rng.standard_normal((n, self.input_dim))
        for ti in reversed(range(self.T)):
            ta  = np.full(n, ti + 1, dtype=float)
            te  = _sinusoidal_emb(ta, self.n_fourier)
            ep, _ = self._net.forward(x, te)
            a_t = self.alphas[ti]; ab_t = self.alpha_bar[ti]; b_t = self.betas[ti]
            xm  = (x - b_t / np.sqrt(1. - ab_t) * ep) / np.sqrt(a_t)
            x   = xm + (np.sqrt(b_t) * self.rng.standard_normal(x.shape)
                        if ti > 0 else 0.)
        return x * self._std + self._mean

    def generate_conditioned(
        self,
        X_ref:       np.ndarray,
        n:           int   = 200,
        noise_level: float = 0.5,
    ) -> np.ndarray:
        """
        Generación condicionada por difusión parcial.

        Procedimiento:
          1. Remuestrear X_ref con reemplazamiento → n observaciones base.
          2. Añadir ruido hasta el paso τ* = noise_level · T (forward).
          3. Revertir desde τ* hasta τ=0 con ε_θ (reverse).

        Parámetros
        ----------
        X_ref : np.ndarray (N_ref, d)  Observaciones de referencia.
        n : int                         Nº de observaciones sintéticas.
        noise_level : float ∈ (0,1)     Mayor → más diversidad.

        Retorna np.ndarray (n, d) en escala original.
        """
        self._check_fitted()
        tau_star = max(1, int(noise_level * self.T))
        idx  = self.rng.integers(0, len(X_ref), size=n)
        x_sc = (X_ref[idx] - self._mean) / self._std
        tau_arr = np.full(n, tau_star - 1)
        x = self._q_sample(x_sc, tau_arr,
                           self.rng.standard_normal(x_sc.shape))
        for ti in reversed(range(tau_star)):
            ta  = np.full(n, ti + 1, dtype=float)
            te  = _sinusoidal_emb(ta, self.n_fourier)
            ep, _ = self._net.forward(x, te)
            a_t = self.alphas[ti]; ab_t = self.alpha_bar[ti]; b_t = self.betas[ti]
            xm  = (x - b_t / np.sqrt(1. - ab_t) * ep) / np.sqrt(a_t)
            x   = xm + (np.sqrt(b_t) * self.rng.standard_normal(x.shape)
                        if ti > 0 else 0.)
        return x * self._std + self._mean

    # ------------------------------------------------------------------
    # Privados
    # ------------------------------------------------------------------

    def _val_loss(self, X_val: np.ndarray) -> float:
        n   = len(X_val)
        tau = self.rng.integers(0, self.T, size=n)
        eps = self.rng.standard_normal(X_val.shape)
        xt  = self._q_sample(X_val, tau, eps)
        te  = _sinusoidal_emb(tau.astype(float) + 1, self.n_fourier)
        pr, _ = self._net.forward(xt, te)
        return float(np.mean((pr - eps) ** 2))

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Llama a fit() primero.")

    def __repr__(self):
        st = "entrenado" if self._fitted else "sin entrenar"
        return f"DDPM(d={self.input_dim}, T={self.T}, h={self.hidden_dim}, {st})"


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

class DDPMPipeline:
    """
    Gestiona el ciclo completo del Módulo 4:
      1. Preparar datos (panel real + sintéticos VAE opcionales).
      2. Entrenar el DDPM.
      3. Generar datos sintéticos de pre-crisis condicionados.
      4. Evaluar fidelidad estadística.
      5. Exportar y visualizar.

    Parámetros
    ----------
    panel : pd.DataFrame           Panel maestro.
    feature_cols : list[str]       Features a modelizar.
    T : int                        Pasos de difusión.
    hidden_dim : int               Neuronas ocultas.
    epochs : int                   Épocas de entrenamiento.
    train_cutoff : int             Año de corte para entrenamiento.
    n_synthetic : int              Observaciones sintéticas a generar.
    noise_level : float            Nivel de ruido en generación condicionada.
    vae_synthetic_path : Path|None Ruta a datos sintéticos VAE (opcional).
    random_state : int             Semilla.
    """

    def __init__(
        self,
        panel:              pd.DataFrame,
        feature_cols:       list[str]   = None,
        T:                  int         = DEFAULT_T_STEPS,
        hidden_dim:         int         = DEFAULT_HIDDEN_DIM,
        epochs:             int         = DEFAULT_EPOCHS,
        train_cutoff:       int         = 1990,
        n_synthetic:        int         = 200,
        noise_level:        float       = 0.5,
        vae_synthetic_path: Path | None = None,
        random_state:       int         = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.panel        = panel.copy()
        self.feature_cols = [f for f in (feature_cols or DDPM_FEATURES)
                             if f in panel.columns]
        self.T            = T
        self.hidden_dim   = hidden_dim
        self.epochs       = epochs
        self.train_cutoff = train_cutoff
        self.n_synthetic  = n_synthetic
        self.noise_level  = noise_level
        self.vae_syn_path = vae_synthetic_path
        self.rng          = np.random.default_rng(random_state)

        self.ddpm:            DDPM | None         = None
        self.panel_augmented: pd.DataFrame | None = None
        self.fidelity_df:     pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> None:
        print("\n[DDPMPipeline] Iniciando pipeline DDPM…")
        X_train, X_pc = self._prepare_data()
        print(f"  Entrenamiento: {len(X_train)} obs · pre-crisis: {len(X_pc)}")

        self.ddpm = DDPM(
            input_dim   = len(self.feature_cols),
            T           = self.T,
            hidden_dim  = self.hidden_dim,
            epochs      = self.epochs,
            random_state= int(self.rng.integers(0, 10000)),
        )
        self.ddpm.fit(X_train, verbose=verbose)

        if len(X_pc) > 0:
            X_syn = self.ddpm.generate_conditioned(
                X_pc, n=self.n_synthetic, noise_level=self.noise_level)
        else:
            print("  ⚠ Sin pre-crisis → generación no condicionada.")
            X_syn = self.ddpm.generate(n=self.n_synthetic)

        self.panel_augmented = self._build_panel(X_syn)
        print(f"  Datos sintéticos generados: {len(X_syn)} obs")

        self.fidelity_df = self._fidelity(X_pc if len(X_pc)>0 else X_train[:50],
                                          X_syn)
        self._export(X_syn)
        self._plot_all(X_train, X_pc, X_syn)
        print("[DDPMPipeline] Pipeline completado.")

    # ------------------------------------------------------------------
    # Privados
    # ------------------------------------------------------------------

    def _prepare_data(self):
        feats = self.feature_cols
        X_real = (self.panel[self.panel["year"] < self.train_cutoff]
                  .dropna(subset=feats)[feats].values)
        X_vae = np.empty((0, len(feats)))
        if self.vae_syn_path and Path(self.vae_syn_path).exists():
            vdf   = pd.read_csv(self.vae_syn_path)
            avail = [f for f in feats if f in vdf.columns]
            if len(avail) == len(feats):
                X_vae = vdf[avail].dropna().values

        X_train = np.vstack([X_real, X_vae]) if len(X_vae) > 0 else X_real

        pc_mask = ((self.panel["year"] < self.train_cutoff) &
                   ((self.panel.get("crisis_h1",
                                    pd.Series(0, index=self.panel.index)) == 1) |
                    (self.panel.get("crisis_h2",
                                    pd.Series(0, index=self.panel.index)) == 1)))
        X_pc_df = self.panel[pc_mask].dropna(subset=feats)
        X_pc    = X_pc_df[feats].values if len(X_pc_df) > 0 \
                  else np.empty((0, len(feats)))
        return X_train, X_pc

    def _build_panel(self, X_syn):
        df = pd.DataFrame(X_syn, columns=self.feature_cols)
        df["crisis_bin"] = 0; df["crisis_h1"] = 1
        df["crisis_h2"]  = 1; df["crisis_h3"] = 1
        df["country"]    = "SYNTHETIC_DDPM"
        df["year"]       = -1; df["is_synthetic"] = True
        return df

    def _fidelity(self, X_real, X_syn):
        from scipy.stats import skew, kurtosis
        rows = []
        for j, feat in enumerate(self.feature_cols):
            if j >= X_real.shape[1] or j >= X_syn.shape[1]: break
            r = X_real[:, j];  s = X_syn[:, j]
            rows.append({"variable": feat,
                         "mean_real": float(np.nanmean(r)),
                         "mean_syn":  float(np.nanmean(s)),
                         "std_real":  float(np.nanstd(r)),
                         "std_syn":   float(np.nanstd(s)),
                         "skew_real": float(skew(r[~np.isnan(r)])),
                         "skew_syn":  float(skew(s[~np.isnan(s)])),
                         "kurt_real": float(kurtosis(r[~np.isnan(r)])),
                         "kurt_syn":  float(kurtosis(s[~np.isnan(s)]))})
        return pd.DataFrame(rows)

    def _export(self, X_syn):
        dd = OUTPUT_DIR / "data"; dd.mkdir(parents=True, exist_ok=True)
        if self.panel_augmented is not None:
            self.panel_augmented.to_csv(dd/"ddpm_synthetic_precrisis.csv",
                                        index=False)
            print(f"  Datos sintéticos → ddpm_synthetic_precrisis.csv")
        if self.fidelity_df is not None:
            self.fidelity_df.to_csv(dd/"ddpm_fidelity_statistics.csv",
                                    index=False)
            print(f"  Fidelidad → ddpm_fidelity_statistics.csv")
        if self.ddpm and self.ddpm.train_losses_:
            pd.DataFrame({"epoch": range(1, len(self.ddpm.train_losses_)+1),
                          "mse_train": self.ddpm.train_losses_,
                          "mse_val":   self.ddpm.val_losses_}
                         ).to_csv(dd/"ddpm_training_loss.csv", index=False)
            print(f"  Pérdida → ddpm_training_loss.csv")

    def _plot_all(self, X_train, X_pc, X_syn):
        sns.set_theme(style=PLOT_STYLE, font_scale=PLOT_FONT_SCALE)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self._fig_loss()
        self._fig_schedule()
        self._fig_distributions(X_pc, X_syn)
        self._fig_fidelity()
        self._fig_forward(X_pc)

    def _fig_loss(self):
        if not self.ddpm or not self.ddpm.train_losses_: return
        fig, ax = plt.subplots(figsize=(9, 4))
        ep = range(1, len(self.ddpm.train_losses_)+1)
        ax.plot(ep, self.ddpm.train_losses_, color=COLORS["no_crisis"],
                lw=1.5, label="Train MSE")
        ax.plot(ep, self.ddpm.val_losses_, color=COLORS["crisis"],
                lw=1.5, ls="--", label="Val MSE")
        ax.set_xlabel("Época"); ax.set_ylabel("MSE predicción de ruido")
        ax.set_title(f"Figura 18 — Curva de aprendizaje del DDPM\n"
                     f"(T={self.T}, h={self.hidden_dim}, "
                     f"épocas={self.epochs})", fontsize=11)
        ax.legend(); plt.tight_layout()
        self._save(fig, "fig18_ddpm_training_loss.png")

    def _fig_schedule(self):
        if not self.ddpm: return
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        t = np.arange(1, self.T+1)
        axes[0].plot(t, self.ddpm.betas, color=COLORS["crisis"], lw=1.8)
        axes[0].set_xlabel("Paso τ"); axes[0].set_ylabel("β_τ")
        axes[0].set_title("Agenda de ruido β_τ (lineal)", fontsize=10)
        axes[1].plot(t, self.ddpm.alpha_bar, color=COLORS["no_crisis"], lw=1.8)
        axes[1].set_xlabel("Paso τ"); axes[1].set_ylabel("ᾱ_τ")
        axes[1].set_title("Producto acumulado ᾱ_τ\n(ᾱ_T ≈ 0 → ruido puro)",
                           fontsize=10)
        fig.suptitle(f"Figura 19 — Agenda de ruido del proceso de difusión DDPM\n"
                     f"(T={self.T} pasos, β₁={self.ddpm.betas[0]:.4f}, "
                     f"β_T={self.ddpm.betas[-1]:.4f})", fontsize=11)
        plt.tight_layout()
        self._save(fig, "fig19_ddpm_noise_schedule.png")

    def _fig_distributions(self, X_pc, X_syn):
        if len(X_pc) == 0: return
        key = [f for f in ["tloans_gap","tloans_growth","hpnom_gap",
                            "lev","ltd","term_spread"]
               if f in self.feature_cols][:6]
        idx_list = [self.feature_cols.index(f) for f in key]
        nc = min(3, len(key)); nr = -(-len(key)//nc)
        fig, axes = plt.subplots(nr, nc, figsize=(4.5*nc, 3.5*nr))
        axes = np.array(axes).flatten()
        for ax, j in zip(axes, idx_list):
            r = X_pc[:,j]; s = X_syn[:,j]
            lo = np.nanpercentile(np.concatenate([r,s]),1)
            hi = np.nanpercentile(np.concatenate([r,s]),99)
            ax.hist(np.clip(r,lo,hi), bins=20, density=True,
                    color=COLORS["no_crisis"], alpha=0.6, label="Real pre-crisis")
            ax.hist(np.clip(s,lo,hi), bins=25, density=True,
                    color=COLORS["crisis"], alpha=0.65, label="Sintético DDPM")
            ax.set_title(self.feature_cols[j], fontsize=9)
            ax.tick_params(labelsize=7)
        axes[0].legend(fontsize=7)
        for j in range(len(idx_list), len(axes)): axes[j].set_visible(False)
        fig.suptitle("Figura 20 — Distribuciones marginales: datos reales de "
                     "pre-crisis\nvs datos sintéticos DDPM (densidades)",
                     fontsize=11)
        plt.tight_layout()
        self._save(fig, "fig20_ddpm_distributions.png")

    def _fig_fidelity(self):
        if self.fidelity_df is None: return
        df = self.fidelity_df
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, col_r, col_s, title in [
            (axes[0], "mean_real","mean_syn","Medias: real vs DDPM"),
            (axes[1], "std_real", "std_syn", "Desv. típicas: real vs DDPM"),
        ]:
            ax.scatter(df[col_r], df[col_s],
                       color=COLORS["no_crisis"] if "mean" in col_r
                             else COLORS["crisis"],
                       s=55, alpha=0.8)
            for _, row in df.iterrows():
                ax.annotate(str(row["variable"])[:9],
                            (row[col_r], row[col_s]), fontsize=5.5, alpha=0.7)
            lim = [df[[col_r,col_s]].min().min(), df[[col_r,col_s]].max().max()]
            ax.plot(lim, lim, "k--", lw=0.8)
            ax.set_xlabel(f"{col_r}"); ax.set_ylabel(f"{col_s}")
            ax.set_title(title, fontsize=10)
        fig.suptitle("Figura 21 — Comparación de momentos estadísticos: real vs DDPM\n"
                     "Diagonal = fidelidad perfecta · cada punto = una variable",
                     fontsize=11)
        plt.tight_layout()
        self._save(fig, "fig21_ddpm_fidelity_scatter.png")

    def _fig_forward(self, X_pc):
        if not self.ddpm or len(X_pc) == 0: return
        rng2 = np.random.default_rng(0)
        x0   = ((X_pc[0] - self.ddpm._mean[0]) / self.ddpm._std[0])
        steps = [0, 20, 50, 100, 150, 199]
        noisy = []
        for s in steps:
            eps = rng2.standard_normal(x0.shape)
            xt  = self.ddpm.sqrt_ab[s]*x0 + self.ddpm.sqrt_1m_ab[s]*eps
            noisy.append(xt)
        n_show = min(8, len(x0))
        fig, axes = plt.subplots(1, len(steps), figsize=(13, 3.5))
        for ax, xt, s in zip(axes, noisy, steps):
            ax.barh(range(n_show), xt[:n_show],
                    color=COLORS["no_crisis"], alpha=0.75)
            ax.set_yticks(range(n_show))
            ax.set_yticklabels([f[:10] for f in self.feature_cols[:n_show]],
                               fontsize=6)
            ax.set_title(f"τ={s+1}", fontsize=9)
            ax.axvline(0, color="black", lw=0.5)
            ax.tick_params(axis="x", labelsize=7)
        fig.suptitle("Figura 22 — Proceso forward: degradación progresiva de "
                     "una observación\nτ=1 = señal original · τ=200 ≈ ruido puro",
                     fontsize=10)
        plt.tight_layout()
        self._save(fig, "fig22_ddpm_forward_diffusion.png")

    def _save(self, fig, name):
        fig.savefig(FIGURES_DIR/name, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {name}")
