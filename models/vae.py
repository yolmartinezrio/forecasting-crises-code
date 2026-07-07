"""
=============================================================================
models/vae.py
=============================================================================
Clase TemporalVAE — Autoencoder Variacional para series temporales
macro-financieras.

Arquitectura
------------
El VAE aprende una representación latente probabilística del estado
macro-financiero del sistema. Dado un vector de observaciones x ∈ ℝ^d
para el país i en el año t, el modelo aprende:

  Encoder : x  →  (μ_z, log σ²_z)   distribución posterior q_φ(z|x)
  Decoder : z  →  x̂                  distribución generativa p_θ(x|z)

con espacio latente z ∈ ℝ^k  (k << d).

La dimensión temporal se incorpora mediante ventanas deslizantes de
W años consecutivos: en lugar de codificar una sola observación, el
encoder recibe la secuencia [x_{t-W+1}, ..., x_t] y la resume en una
representación latente z_t que condensa la historia reciente del país.
Esto permite al modelo capturar patrones de acumulación de riesgo que
se despliegan a lo largo de múltiples años (ciclo crediticio).

Implementación
--------------
Al no disponer de PyTorch en el entorno de ejecución, el VAE se
implementa en NumPy puro con redes neuronales feed-forward de dos capas
para el encoder y el decoder. La retroalimentación temporal se
aproxima concatenando la ventana de W observaciones como vector de
entrada aplanado de dimensión W·d, lo que captura las dependencias de
corto plazo sin necesidad de una arquitectura LSTM explícita.

El entrenamiento se realiza mediante Adam (implementación propia) con
el objetivo ELBO estándar del VAE:

  L_ELBO = E_{q_φ}[log p_θ(x|z)] - D_KL(q_φ(z|x) || p(z))

donde p(z) = N(0,I) es la prior gaussiana estándar sobre el espacio
latente y el término de reconstrucción se implementa como MSE negativo.

Uso principal
-------------
El VAE se usa para dos tareas complementarias en el pipeline:

  1. Extracción de indicadores latentes de vulnerabilidad sistémica:
     el vector μ_z(x_{i,t}) condensado por el encoder sirve como señal
     de alerta temprana adicional que se concatena a las features del
     clasificador logit+VAE.

  2. Generación de datos sintéticos de pre-crisis (data augmentation):
     el decoder genera nuevas observaciones muestreando z ~ N(0,I)
     condicionadas a las características estadísticas de los períodos
     de pre-crisis históricos, mitigando el desbalance de clases.

Referencias
-----------
Kingma, D.P. & Welling, M. (2014). Auto-encoding variational Bayes.
  ICLR 2014.

Chen, R.T.Q. et al. (2018). Isolating sources of disentanglement in
  variational autoencoders. NeurIPS 31.
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config.settings import (
    FORECAST_HORIZONS, OUTPUT_DIR, FIGURES_DIR,
    COLORS, FIGURE_DPI, PLOT_STYLE, PLOT_FONT_SCALE,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Hiperparámetros por defecto
# ---------------------------------------------------------------------------

VAE_FEATURES: list[str] = [
    "tloans", "tmort", "lev", "ltd", "noncore",
    "hpnom", "housing_capgain", "eq_capgain",
    "stir", "ltrate", "bill_rate",
    "gdp", "iy", "ca", "cpi", "debtgdp", "money",
    "bond_rate", "housing_tr", "eq_tr",
    "tloans_gap", "hpnom_gap", "term_spread", "tloans_growth",
]

DEFAULT_LATENT_DIM   = 4     # k: dimensión del espacio latente
DEFAULT_HIDDEN_DIM   = 32    # neuronas en capas ocultas del encoder/decoder
DEFAULT_WINDOW_SIZE  = 5     # W: años de historia que ve el encoder
DEFAULT_EPOCHS       = 300   # épocas de entrenamiento
DEFAULT_BATCH_SIZE   = 64    # tamaño de mini-lote
DEFAULT_LR           = 1e-3  # tasa de aprendizaje Adam
DEFAULT_BETA         = 1.0   # peso del término KL (β-VAE)
DEFAULT_RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Red neuronal feed-forward de dos capas (encoder y decoder comparten
# la misma clase base para no duplicar código)
# ---------------------------------------------------------------------------

class _MLP:
    """
    Red neuronal fully-connected de dos capas ocultas con activación ReLU.
    Inicialización He para capas ocultas; inicialización Xavier para la capa
    de salida (coherente con activación lineal en la salida).
    """

    def __init__(
        self,
        in_dim:  int,
        hid_dim: int,
        out_dim: int,
        rng:     np.random.Generator,
    ) -> None:
        scale1 = np.sqrt(2.0 / in_dim)
        scale2 = np.sqrt(2.0 / hid_dim)
        scaleO = np.sqrt(1.0 / hid_dim)

        self.W1 = rng.normal(0, scale1, (in_dim,  hid_dim))
        self.b1 = np.zeros(hid_dim)
        self.W2 = rng.normal(0, scale2, (hid_dim, hid_dim))
        self.b2 = np.zeros(hid_dim)
        self.Wo = rng.normal(0, scaleO, (hid_dim, out_dim))
        self.bo = np.zeros(out_dim)

        # Momentos Adam (primer y segundo momento)
        self._m  = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v  = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t  = 0  # paso de tiempo Adam

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        """Propagación hacia adelante. Devuelve salida y caché para backward."""
        h1  = _relu(x @ self.W1 + self.b1)
        h2  = _relu(h1 @ self.W2 + self.b2)
        out = h2 @ self.Wo + self.bo
        cache = {"x": x, "h1": h1, "h2": h2}
        return out, cache

    def backward(
        self, d_out: np.ndarray, cache: dict
    ) -> tuple[dict, np.ndarray]:
        """
        Backpropagation. Devuelve gradientes de los parámetros y
        gradiente respecto a la entrada x (para poder encadenar modelos).
        """
        x, h1, h2 = cache["x"], cache["h1"], cache["h2"]
        n = x.shape[0]

        # Capa de salida (lineal)
        dWo = h2.T @ d_out / n
        dbo = d_out.mean(axis=0)
        dh2 = d_out @ self.Wo.T

        # Capa oculta 2
        dh2_pre = dh2 * _relu_grad(h2)
        dW2 = h1.T @ dh2_pre / n
        db2 = dh2_pre.mean(axis=0)
        dh1 = dh2_pre @ self.W2.T

        # Capa oculta 1
        dh1_pre = dh1 * _relu_grad(h1)
        dW1 = x.T @ dh1_pre / n
        db1 = dh1_pre.mean(axis=0)
        dx  = dh1_pre @ self.W1.T

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "Wo": dWo, "bo": dbo}
        return grads, dx

    def adam_step(
        self,
        grads: dict,
        lr:    float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps:   float = 1e-8,
    ) -> None:
        """Actualización Adam de los parámetros."""
        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * g ** 2
            m_hat = self._m[k] / (1 - beta1 ** self._t)
            v_hat = self._v[k] / (1 - beta2 ** self._t)
            getattr(self, k)[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2,
                "b2": self.b2, "Wo": self.Wo, "bo": self.bo}


# ---------------------------------------------------------------------------
# Clase principal: TemporalVAE
# ---------------------------------------------------------------------------

class TemporalVAE:
    """
    Autoencoder Variacional Temporal para datos macro-financieros de panel.

    El encoder mapea una ventana temporal de W observaciones a una
    distribución gaussiana diagonal en el espacio latente z ∈ ℝ^k.
    El decoder reconstruye la observación actual x_t a partir de z.

    Parámetros
    ----------
    input_dim : int
        Dimensión del vector de observación d (número de features).
    latent_dim : int
        Dimensión del espacio latente k.
    hidden_dim : int
        Número de neuronas en cada capa oculta del encoder y decoder.
    window_size : int
        Número de observaciones históricas W que usa el encoder.
    beta : float
        Peso del término de divergencia KL (β-VAE). β=1 corresponde
        al VAE estándar de Kingma & Welling (2014).
    learning_rate : float
        Tasa de aprendizaje para el optimizador Adam.
    epochs : int
        Número máximo de épocas de entrenamiento.
    batch_size : int
        Tamaño del mini-lote.
    random_state : int
        Semilla de aleatoriedad.

    Atributos (tras llamar a fit)
    -----------------------------
    train_losses_ : list[float]
        Historial del ELBO en entrenamiento por época.
    val_losses_ : list[float]
        Historial del ELBO en validación por época.

    Uso típico
    ----------
    >>> vae = TemporalVAE(input_dim=24, latent_dim=4)
    >>> vae.fit(X_sequences)
    >>> z_mu = vae.encode(X_sequences)      # representaciones latentes
    >>> X_syn = vae.generate(n=100)         # muestras sintéticas
    """

    def __init__(
        self,
        input_dim:    int   = 24,
        latent_dim:   int   = DEFAULT_LATENT_DIM,
        hidden_dim:   int   = DEFAULT_HIDDEN_DIM,
        window_size:  int   = DEFAULT_WINDOW_SIZE,
        beta:         float = DEFAULT_BETA,
        learning_rate:float = DEFAULT_LR,
        epochs:       int   = DEFAULT_EPOCHS,
        batch_size:   int   = DEFAULT_BATCH_SIZE,
        random_state: int   = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.input_dim     = input_dim
        self.latent_dim    = latent_dim
        self.hidden_dim    = hidden_dim
        self.window_size   = window_size
        self.beta          = beta
        self.lr            = learning_rate
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.rng           = np.random.default_rng(random_state)

        # Dimensión de entrada al encoder: W observaciones aplanadas
        enc_in = window_size * input_dim

        # Encoder: x_window → (μ_z, log σ²_z)
        # La salida tiene dimensión 2·k: los primeros k valores son μ_z,
        # los últimos k son log σ²_z
        self._enc = _MLP(enc_in, hidden_dim, 2 * latent_dim, self.rng)

        # Decoder: z → x̂ (reconstrucción de la observación actual)
        self._dec = _MLP(latent_dim, hidden_dim, input_dim, self.rng)

        # Estandarización (se ajusta en fit)
        self._mean: np.ndarray | None = None
        self._std:  np.ndarray | None = None

        # Historial de pérdidas
        self.train_losses_: list[float] = []
        self.val_losses_:   list[float] = []
        self._fitted = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def fit(
        self,
        X_seq:   np.ndarray,
        X_target:np.ndarray,
        val_frac: float = 0.15,
        verbose:  bool  = True,
        verbose_every: int = 50,
    ) -> "TemporalVAE":
        """
        Entrena el VAE sobre las secuencias de ventanas temporales.

        Parámetros
        ----------
        X_seq : np.ndarray de shape (N, W*d)
            Ventanas temporales aplanadas. Cada fila es la concatenación
            de W observaciones consecutivas del país i.
        X_target : np.ndarray de shape (N, d)
            Observación actual x_t que el decoder debe reconstruir.
            Corresponde a la última observación de cada ventana.
        val_frac : float
            Fracción de datos reservada para validación (sin reentrenar).
        verbose : bool
            Si True, imprime el ELBO cada verbose_every épocas.
        verbose_every : int
            Frecuencia de reporte en épocas.

        Retorna
        -------
        self
        """
        # Estandarizar con estadísticos del conjunto de entrenamiento
        self._mean = X_seq.mean(axis=0, keepdims=True)
        self._std  = X_seq.std(axis=0, keepdims=True) + 1e-8
        X_seq_sc   = (X_seq - self._mean) / self._std

        mean_tgt   = X_target.mean(axis=0, keepdims=True)
        std_tgt    = X_target.std(axis=0, keepdims=True) + 1e-8
        X_tgt_sc   = (X_target - mean_tgt) / std_tgt
        self._mean_tgt = mean_tgt
        self._std_tgt  = std_tgt

        # Split train/val (temporal: los últimos val_frac% como val)
        N       = len(X_seq_sc)
        n_val   = max(1, int(N * val_frac))
        n_train = N - n_val

        X_tr, X_val_s   = X_seq_sc[:n_train],    X_seq_sc[n_train:]
        Xt_tr, Xt_val   = X_tgt_sc[:n_train],    X_tgt_sc[n_train:]

        if verbose:
            print(
                f"  [VAE] Entrenando: {n_train} train · {n_val} val · "
                f"d={self.input_dim} · k={self.latent_dim} · "
                f"W={self.window_size} · épocas={self.epochs}"
            )

        for epoch in range(1, self.epochs + 1):
            # Mini-lotes aleatorios
            idx    = self.rng.permutation(n_train)
            losses = []

            for start in range(0, n_train, self.batch_size):
                batch_idx = idx[start:start + self.batch_size]
                x_enc = X_tr[batch_idx]
                x_dec = Xt_tr[batch_idx]

                loss, _ = self._train_step(x_enc, x_dec)
                losses.append(loss)

            train_loss = float(np.mean(losses))
            val_loss   = self._elbo(X_val_s, Xt_val)
            self.train_losses_.append(train_loss)
            self.val_losses_.append(val_loss)

            if verbose and epoch % verbose_every == 0:
                print(
                    f"    Época {epoch:4d}/{self.epochs}  "
                    f"ELBO_train={train_loss:.4f}  "
                    f"ELBO_val={val_loss:.4f}"
                )

        self._fitted = True
        if verbose:
            print(
                f"  [VAE] Entrenamiento completado. "
                f"ELBO_val final = {self.val_losses_[-1]:.4f}"
            )
        return self

    def encode(self, X_seq: np.ndarray) -> np.ndarray:
        """
        Calcula la media posterior μ_z para cada ventana de entrada.

        Parámetros
        ----------
        X_seq : np.ndarray (N, W*d)
            Ventanas temporales (sin estandarizar; la estandarización
            se aplica internamente).

        Retorna
        -------
        np.ndarray (N, k) con los vectores latentes μ_z.
        """
        self._check_fitted()
        X_sc = (X_seq - self._mean) / self._std
        enc_out, _ = self._enc.forward(X_sc)
        return enc_out[:, :self.latent_dim]   # μ_z

    def encode_logvar(self, X_seq: np.ndarray) -> np.ndarray:
        """Devuelve log σ²_z para cada ventana (incertidumbre latente)."""
        self._check_fitted()
        X_sc = (X_seq - self._mean) / self._std
        enc_out, _ = self._enc.forward(X_sc)
        return enc_out[:, self.latent_dim:]   # log σ²_z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        Reconstruye x̂ a partir del código latente z.

        Retorna
        -------
        np.ndarray (N, d) en la escala original de los datos.
        """
        self._check_fitted()
        x_hat_sc, _ = self._dec.forward(z)
        return x_hat_sc * self._std_tgt + self._mean_tgt

    def generate(
        self,
        n:         int = 100,
        z_mean:    np.ndarray | None = None,
        z_std:     float = 1.0,
    ) -> np.ndarray:
        """
        Genera n observaciones sintéticas muestreando del espacio latente.

        Si z_mean es None, muestrea de la prior estándar N(0, I).
        Si z_mean se proporciona (p. ej. media de los códigos de
        pre-crisis), genera muestras en torno a esa región del espacio
        latente, produciendo datos sintéticos de pre-crisis.

        Parámetros
        ----------
        n : int
            Número de observaciones a generar.
        z_mean : np.ndarray (k,) | None
            Centro del espacio latente desde el que samplear.
        z_std : float
            Desviación típica del ruido añadido al código latente.

        Retorna
        -------
        np.ndarray (n, d) con las observaciones sintéticas en escala
        original.
        """
        self._check_fitted()
        if z_mean is None:
            z = self.rng.standard_normal((n, self.latent_dim))
        else:
            noise = self.rng.standard_normal((n, self.latent_dim))
            z     = z_mean[np.newaxis, :] + z_std * noise
        return self.decode(z)

    def generate_precrisis_synthetic(
        self,
        X_seq_precrisis: np.ndarray,
        n_synthetic:     int = 200,
    ) -> np.ndarray:
        """
        Genera datos sintéticos de pre-crisis condicionados a la
        región del espacio latente correspondiente a los períodos
        de pre-crisis históricos observados.

        El procedimiento es:
          1. Codificar las ventanas de pre-crisis históricas → {μ_z^(i)}
          2. Calcular la media μ̄_z del conjunto de códigos de pre-crisis
          3. Muestrear z ~ N(μ̄_z, I) y decodificar → datos sintéticos

        Parámetros
        ----------
        X_seq_precrisis : np.ndarray (N_pc, W*d)
            Ventanas temporales correspondientes a períodos de pre-crisis.
        n_synthetic : int
            Número de observaciones sintéticas a generar.

        Retorna
        -------
        np.ndarray (n_synthetic, d)
        """
        self._check_fitted()
        z_precrisis = self.encode(X_seq_precrisis)
        z_mean      = z_precrisis.mean(axis=0)
        z_std       = z_precrisis.std(axis=0).mean() + 0.1
        return self.generate(n_synthetic, z_mean=z_mean, z_std=z_std)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _reparametrize(
        self, mu: np.ndarray, log_var: np.ndarray
    ) -> np.ndarray:
        """
        Truco de reparametrización: z = μ + σ·ε, ε ~ N(0,I).
        Permite que el gradiente fluya a través de la operación de
        muestreo, que de otro modo sería no diferenciable.
        """
        eps = self.rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def _elbo(
        self, X_seq: np.ndarray, X_target: np.ndarray
    ) -> float:
        """
        Calcula el ELBO (Evidence Lower BOund) sin actualizar parámetros.
        ELBO = E[log p(x|z)] - β · D_KL(q(z|x) || p(z))

        La pérdida de reconstrucción se implementa como MSE negativo
        (equivalente a asumir p(x|z) gaussiana).
        """
        enc_out, _ = self._enc.forward(X_seq)
        mu, log_var = enc_out[:, :self.latent_dim], enc_out[:, self.latent_dim:]
        z           = self._reparametrize(mu, log_var)
        x_hat, _    = self._dec.forward(z)

        # Pérdida de reconstrucción (MSE, promediada sobre features y batch)
        recon = -np.mean((X_target - x_hat) ** 2)

        # Divergencia KL analítica: D_KL(N(μ,σ²) || N(0,1))
        # = -0.5 · Σ (1 + log σ² - μ² - σ²)
        kl = -0.5 * np.mean(1 + log_var - mu ** 2 - np.exp(log_var))

        return float(recon - self.beta * kl)

    def _train_step(
        self, X_seq: np.ndarray, X_target: np.ndarray
    ) -> tuple[float, None]:
        """
        Un paso de actualización de parámetros sobre un mini-lote.
        Devuelve el valor del ELBO (negativo = pérdida a minimizar).
        """
        n = X_seq.shape[0]

        # ── Forward pass ─────────────────────────────────────────────
        enc_out, enc_cache = self._enc.forward(X_seq)
        mu      = enc_out[:, :self.latent_dim]
        log_var = enc_out[:, self.latent_dim:]

        z = self._reparametrize(mu, log_var)

        x_hat, dec_cache = self._dec.forward(z)

        # ── Pérdidas ──────────────────────────────────────────────────
        recon_err = X_target - x_hat                   # (N, d)
        recon_loss = np.mean(recon_err ** 2)
        kl_loss    = -0.5 * np.mean(1 + log_var - mu ** 2 - np.exp(log_var))
        elbo       = -(- recon_loss - self.beta * kl_loss)

        # ── Backward pass — decoder ───────────────────────────────────
        # d(MSE)/d(x_hat) = -2/N · (X_target - x_hat) / d
        d_xhat = -2.0 * recon_err / (n * self.input_dim)
        dec_grads, d_z = self._dec.backward(d_xhat, dec_cache)
        self._dec.adam_step(dec_grads, lr=self.lr)

        # ── Backward pass — encoder ───────────────────────────────────
        # Gradiente respecto a z fluye de d_z (reconstrucción)
        # más el gradiente del truco de reparametrización
        eps     = (z - mu) / (np.exp(0.5 * log_var) + 1e-8)

        # Gradiente KL respecto a mu y log_var
        d_mu      = (mu / n) * self.beta + d_z
        d_logvar  = (0.5 * (np.exp(log_var) - 1) / n * self.beta
                     + d_z * eps * 0.5 * np.exp(0.5 * log_var))

        d_enc_out = np.concatenate([d_mu, d_logvar], axis=1)
        enc_grads, _ = self._enc.backward(d_enc_out, enc_cache)
        self._enc.adam_step(enc_grads, lr=self.lr)

        return float(elbo), None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "El VAE no ha sido entrenado. Llama a fit() primero."
            )

    def __repr__(self) -> str:
        st = "entrenado" if self._fitted else "sin entrenar"
        return (
            f"TemporalVAE(d={self.input_dim}, k={self.latent_dim}, "
            f"W={self.window_size}, {st})"
        )


# ---------------------------------------------------------------------------
# Clase de gestión del pipeline VAE completo
# ---------------------------------------------------------------------------

class VAEPipeline:
    """
    Gestiona el ciclo completo de trabajo con el VAE:

      1. Construcción de secuencias temporales desde el panel maestro.
      2. Entrenamiento del TemporalVAE.
      3. Extracción de representaciones latentes (indicadores de riesgo).
      4. Generación de datos sintéticos de pre-crisis.
      5. Evaluación estadística de la fidelidad de los datos sintéticos.
      6. Exportación de resultados y figuras.

    Parámetros
    ----------
    panel : pd.DataFrame
        Panel maestro completo (output de DataPreprocessor).
    feature_cols : list[str]
        Variables a usar como entrada del VAE.
    latent_dim : int
        Dimensión del espacio latente k.
    window_size : int
        Tamaño de la ventana temporal W.
    train_cutoff : int
        Año hasta el que se entrena el VAE (excluido del OOS).
    random_state : int
        Semilla de aleatoriedad.

    Uso típico
    ----------
    >>> vae_pipe = VAEPipeline(panel, feature_cols)
    >>> vae_pipe.run()
    >>> panel_aug = vae_pipe.panel_augmented   # panel con datos sintéticos
    >>> latent_df = vae_pipe.latent_df         # indicadores latentes
    """

    def __init__(
        self,
        panel:        pd.DataFrame,
        feature_cols: list[str]  = None,
        latent_dim:   int        = DEFAULT_LATENT_DIM,
        window_size:  int        = DEFAULT_WINDOW_SIZE,
        hidden_dim:   int        = DEFAULT_HIDDEN_DIM,
        epochs:       int        = DEFAULT_EPOCHS,
        train_cutoff: int        = 1990,
        n_synthetic:  int        = 200,
        random_state: int        = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.panel        = panel.copy()
        self.feature_cols = feature_cols or VAE_FEATURES
        self.feature_cols = [f for f in self.feature_cols
                             if f in panel.columns]
        self.latent_dim   = latent_dim
        self.window_size  = window_size
        self.hidden_dim   = hidden_dim
        self.epochs       = epochs
        self.train_cutoff = train_cutoff
        self.n_synthetic  = n_synthetic
        self.rng          = np.random.default_rng(random_state)

        # Outputs
        self.vae:              TemporalVAE | None  = None
        self.latent_df:        pd.DataFrame | None = None
        self.panel_augmented:  pd.DataFrame | None = None
        self._scaler_mean:     np.ndarray | None   = None
        self._scaler_std:      np.ndarray | None   = None

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def run(self, verbose: bool = True) -> None:
        """Ejecuta el pipeline completo."""
        print("\n[VAEPipeline] Iniciando pipeline VAE…")

        # Paso 1: construir secuencias
        X_seq, X_tgt, meta = self._build_sequences()
        print(f"  Secuencias construidas: {len(X_seq)} ventanas "
              f"(W={self.window_size}, d={len(self.feature_cols)})")

        # Paso 2: separar entrenamiento (pre-1990)
        train_mask = meta["year"] < self.train_cutoff
        X_tr  = X_seq[train_mask]
        Xt_tr = X_tgt[train_mask]
        print(f"  Ventanas de entrenamiento (año < {self.train_cutoff}): "
              f"{train_mask.sum()}")

        # Paso 3: entrenar VAE
        d   = len(self.feature_cols)
        self.vae = TemporalVAE(
            input_dim    = d,
            latent_dim   = self.latent_dim,
            hidden_dim   = self.hidden_dim,
            window_size  = self.window_size,
            epochs       = self.epochs,
            random_state = int(self.rng.integers(0, 10000)),
        )
        self.vae.fit(X_tr, Xt_tr, verbose=verbose)

        # Paso 4: extraer representaciones latentes para todo el panel
        z_mu = self.vae.encode(X_seq)     # (N, k)
        self.latent_df = self._build_latent_df(z_mu, meta)
        print(f"  Representaciones latentes extraídas: shape {z_mu.shape}")

        # Paso 5: generar datos sintéticos de pre-crisis
        X_precrisis = self._get_precrisis_sequences(X_seq, meta)
        if len(X_precrisis) > 0:
            X_syn = self.vae.generate_precrisis_synthetic(
                X_precrisis, n_synthetic=self.n_synthetic
            )
            self.panel_augmented = self._build_augmented_panel(X_syn)
            print(f"  Datos sintéticos generados: {len(X_syn)} observaciones")
        else:
            print("  ⚠ No hay secuencias de pre-crisis en el conjunto de "
                  "entrenamiento. Datos sintéticos omitidos.")
            self.panel_augmented = None

        # Paso 6: evaluar fidelidad y guardar
        if X_precrisis is not None and len(X_precrisis) > 0 and \
           self.panel_augmented is not None:
            self._evaluate_fidelity(X_tgt[meta["crisis_h1"] == 1],
                                    X_syn)
        self._export()
        self._plot_all(z_mu, meta)
        print("[VAEPipeline] Pipeline completado.")

    # ------------------------------------------------------------------
    # Construcción de secuencias
    # ------------------------------------------------------------------

    def _build_sequences(
        self,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Construye las ventanas temporales deslizantes para cada país.

        Para cada país i y año t ≥ t_min + W - 1:
          X_seq[i,t] = [x_{i, t-W+1}, ..., x_{i,t}]  (aplanado)
          X_tgt[i,t] = x_{i,t}                        (observación actual)

        Solo se incluyen filas sin NaN en ninguna feature.
        """
        feats  = self.feature_cols
        W      = self.window_size
        seqs, tgts, metas = [], [], []

        panel_sorted = self.panel.sort_values(["country", "year"])

        for country, grp in panel_sorted.groupby("country"):
            grp = grp.dropna(subset=feats).reset_index(drop=True)
            X   = grp[feats].values  # (T_c, d)
            T_c = len(X)

            if T_c < W:
                continue

            for t in range(W - 1, T_c):
                window = X[t - W + 1: t + 1].flatten()  # (W*d,)
                target = X[t]                             # (d,)
                seqs.append(window)
                tgts.append(target)
                metas.append({
                    "country"  : country,
                    "year"     : grp["year"].iloc[t],
                    "crisis_bin": grp["crisis_bin"].iloc[t],
                    "crisis_h1": grp["crisis_h1"].iloc[t]
                                 if "crisis_h1" in grp.columns else 0,
                    "crisis_h2": grp["crisis_h2"].iloc[t]
                                 if "crisis_h2" in grp.columns else 0,
                    "crisis_h3": grp["crisis_h3"].iloc[t]
                                 if "crisis_h3" in grp.columns else 0,
                })

        X_seq = np.array(seqs)
        X_tgt = np.array(tgts)
        meta  = pd.DataFrame(metas)
        return X_seq, X_tgt, meta

    def _get_precrisis_sequences(
        self, X_seq: np.ndarray, meta: pd.DataFrame
    ) -> np.ndarray:
        """Filtra las ventanas correspondientes a períodos de pre-crisis."""
        mask = (
            (meta["crisis_h1"] == 1) &
            (meta["year"] < self.train_cutoff)
        ).values
        if mask.sum() == 0:
            # Ampliar a h=2 si h=1 no tiene suficientes
            mask = (
                (meta["crisis_h2"] == 1) &
                (meta["year"] < self.train_cutoff)
            ).values
        return X_seq[mask]

    # ------------------------------------------------------------------
    # Construcción de DataFrames de salida
    # ------------------------------------------------------------------

    def _build_latent_df(
        self, z_mu: np.ndarray, meta: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construye un DataFrame con las representaciones latentes μ_z
        junto a los metadatos (país, año, etiquetas de crisis).
        """
        df = meta.copy().reset_index(drop=True)
        for k in range(self.latent_dim):
            df[f"z{k+1}"] = z_mu[:, k]
        return df

    def _build_augmented_panel(
        self, X_syn: np.ndarray
    ) -> pd.DataFrame:
        """
        Construye un DataFrame con los datos sintéticos etiquetados
        como pre-crisis (crisis_h1=1) para su uso como datos de
        aumento de datos en el entrenamiento del clasificador híbrido.
        """
        df = pd.DataFrame(X_syn, columns=self.feature_cols)
        df["crisis_bin"]  = 0
        df["crisis_h1"]   = 1
        df["crisis_h2"]   = 1
        df["crisis_h3"]   = 1
        df["country"]     = "SYNTHETIC_VAE"
        df["year"]        = -1
        df["is_synthetic"]= True
        return df

    # ------------------------------------------------------------------
    # Evaluación de fidelidad estadística
    # ------------------------------------------------------------------

    def _evaluate_fidelity(
        self,
        X_real: np.ndarray,
        X_syn:  np.ndarray,
    ) -> None:
        """
        Evalúa la fidelidad de los datos sintéticos comparando los
        primeros cuatro momentos estadísticos (media, desviación típica,
        curtosis y skewness) con los datos reales de pre-crisis.

        Imprime un resumen por variable.
        """
        if len(X_real) == 0:
            return

        from scipy.stats import kurtosis, skew

        print("\n  Evaluación de fidelidad estadística "
              "(datos reales vs sintéticos):")
        print(f"  {'Variable':<22} {'μ_real':>8} {'μ_syn':>8} "
              f"{'σ_real':>8} {'σ_syn':>8}")
        print("  " + "-" * 60)

        for j, feat in enumerate(self.feature_cols):
            if j >= X_real.shape[1] or j >= X_syn.shape[1]:
                break
            r = X_real[:, j]
            s = X_syn[:, j]
            print(
                f"  {feat:<22} "
                f"{r.mean():>8.3f} {s.mean():>8.3f} "
                f"{r.std():>8.3f} {s.std():>8.3f}"
            )

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def _export(self) -> None:
        """Guarda los DataFrames de salida en disco."""
        data_out = OUTPUT_DIR / "data"
        data_out.mkdir(parents=True, exist_ok=True)

        if self.latent_df is not None:
            path = data_out / "vae_latent_representations.csv"
            self.latent_df.to_csv(path, index=False)
            print(f"  Representaciones latentes → {path.name}")

        if self.panel_augmented is not None:
            path = data_out / "vae_synthetic_precrisis.csv"
            self.panel_augmented.to_csv(path, index=False)
            print(f"  Datos sintéticos → {path.name}")

        if self.vae is not None and self.vae.train_losses_:
            loss_df = pd.DataFrame({
                "epoch"     : range(1, len(self.vae.train_losses_) + 1),
                "elbo_train": self.vae.train_losses_,
                "elbo_val"  : self.vae.val_losses_,
            })
            path = data_out / "vae_training_loss.csv"
            loss_df.to_csv(path, index=False)
            print(f"  Historial de pérdida → {path.name}")

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------

    def _plot_all(
        self, z_mu: np.ndarray, meta: pd.DataFrame
    ) -> None:
        """Genera las figuras del módulo VAE."""
        sns.set_theme(
            style=PLOT_STYLE, font_scale=PLOT_FONT_SCALE
        )
        self._plot_training_loss()
        self._plot_latent_space(z_mu, meta)
        self._plot_latent_timeseries(self.latent_df)
        self._plot_reconstruction(z_mu, self.latent_df)

    def _plot_training_loss(self) -> None:
        """Figura 14 — Curva de aprendizaje del VAE."""
        if not self.vae or not self.vae.train_losses_:
            return
        fig, ax = plt.subplots(figsize=(9, 4))
        epochs = range(1, len(self.vae.train_losses_) + 1)
        ax.plot(epochs, self.vae.train_losses_,
                color=COLORS["no_crisis"], lw=1.5, label="Entrenamiento")
        ax.plot(epochs, self.vae.val_losses_,
                color=COLORS["crisis"],    lw=1.5, label="Validación",
                ls="--")
        ax.set_xlabel("Época")
        ax.set_ylabel("ELBO (negativo = pérdida)")
        ax.set_title(
            "Figura 14 — Curva de aprendizaje del VAE temporal\n"
            f"(k={self.latent_dim}, W={self.window_size}, "
            f"β={self.vae.beta})",
            fontsize=11,
        )
        ax.legend()
        plt.tight_layout()
        self._save_fig(fig, "fig14_vae_training_loss.png")

    def _plot_latent_space(
        self, z_mu: np.ndarray, meta: pd.DataFrame
    ) -> None:
        """Figura 15 — Espacio latente (z1 vs z2) coloreado por crisis."""
        if self.latent_dim < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        mask0 = meta["crisis_bin"].values == 0
        mask1 = meta["crisis_bin"].values == 1
        ax.scatter(z_mu[mask0, 0], z_mu[mask0, 1],
                   c=COLORS["no_crisis"], alpha=0.3, s=8, label="Sin crisis")
        ax.scatter(z_mu[mask1, 0], z_mu[mask1, 1],
                   c=COLORS["crisis"],    alpha=0.9, s=40,
                   marker="*", label="Crisis activa")
        ax.set_xlabel("z₁ (primera dimensión latente)")
        ax.set_ylabel("z₂ (segunda dimensión latente)")
        ax.set_title(
            "Figura 15 — Espacio latente aprendido por el VAE\n"
            "★ = años de crisis activa · puntos = años normales",
            fontsize=11,
        )
        ax.legend(markerscale=2)
        plt.tight_layout()
        self._save_fig(fig, "fig15_vae_latent_space.png")

    def _plot_latent_timeseries(self, meta: pd.DataFrame) -> None:
        """Figura 16 — Series temporales de z1 para países seleccionados."""
        sel_countries = ["USA", "Spain", "Sweden", "Japan"]
        sel_countries = [c for c in sel_countries
                         if c in meta["country"].unique()]
        if not sel_countries:
            return

        fig, axes = plt.subplots(
            len(sel_countries), 1,
            figsize=(12, 3.2 * len(sel_countries)),
            sharex=False,
        )
        if len(sel_countries) == 1:
            axes = [axes]

        for ax, country in zip(axes, sel_countries):
            sub = meta[meta["country"] == country].sort_values("year")
            col = "z1" if "z1" in sub.columns else \
                  [c for c in sub.columns if c.startswith("z")][0]
            z_vals = sub[col].values
            years  = sub["year"].values
            ax.plot(years, z_vals, color=COLORS["no_crisis"], lw=1.5)
            ax.axhline(0, color="black", lw=0.6, ls="--")
            # Sombrear crisis activas
            for yr in sub[sub["crisis_bin"] == 1]["year"]:
                ax.axvspan(yr - 0.5, yr + 0.5,
                           color=COLORS["crisis"], alpha=0.30, lw=0)
            ax.set_ylabel("z₁", fontsize=9)
            ax.set_title(country, fontsize=10, fontweight="bold")
            ax.tick_params(labelsize=8)

        axes[-1].set_xlabel("Año")
        fig.suptitle(
            "Figura 16 — Primera dimensión latente z₁ a lo largo del tiempo\n"
            "Áreas rojas = años de crisis activa (LV)",
            fontsize=11,
        )
        plt.tight_layout()
        self._save_fig(fig, "fig16_vae_latent_timeseries.png")

    def _plot_reconstruction(
        self, z_mu: np.ndarray, meta: pd.DataFrame
    ) -> None:
        """Figura 17 — Calidad de reconstrucción: real vs reconstruido."""
        if self.vae is None:
            return

        # Reconstruir desde el espacio latente
        X_hat = self.vae.decode(z_mu)
        feats = self.feature_cols
        n_feats = min(6, len(feats))

        fig, axes = plt.subplots(
            2, n_feats // 2,
            figsize=(4 * (n_feats // 2), 7),
        )
        axes = axes.flatten()

        # Obtener X_tgt en escala original
        panel_sorted = self.panel.sort_values(["country", "year"])
        X_real_list  = []
        for country, grp in panel_sorted.groupby("country"):
            grp2 = grp.dropna(subset=feats).reset_index(drop=True)
            if len(grp2) >= self.window_size:
                X_real_list.append(grp2[feats].values[self.window_size - 1:])
        X_real = np.vstack(X_real_list)[:len(X_hat)]

        for i, feat in enumerate(feats[:n_feats]):
            ax  = axes[i]
            lo  = np.nanpercentile(X_real[:, i], 1)
            hi  = np.nanpercentile(X_real[:, i], 99)
            ax.scatter(
                np.clip(X_real[:, i], lo, hi),
                np.clip(X_hat[:, i],  lo, hi),
                alpha=0.2, s=4, color=COLORS["no_crisis"],
            )
            lims = [min(lo, X_hat[:, i].min()), max(hi, X_hat[:, i].max())]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_title(feat, fontsize=8)
            ax.set_xlabel("Real", fontsize=7)
            ax.set_ylabel("Reconstruido", fontsize=7)
            ax.tick_params(labelsize=6)

        for j in range(n_feats, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(
            "Figura 17 — Calidad de reconstrucción del VAE\n"
            "Cada punto = una observación · "
            "línea diagonal = reconstrucción perfecta",
            fontsize=11,
        )
        plt.tight_layout()
        self._save_fig(fig, "fig17_vae_reconstruction.png")

    def _save_fig(self, fig: plt.Figure, name: str) -> None:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {name}")
