"""
=============================================================================
utils/filters.py
=============================================================================
Funciones numéricas auxiliares reutilizables en todo el proyecto.

Contenido:
  - hp_filter_gap : filtro Hodrick-Prescott (implementación pura NumPy,
                    sin dependencia de statsmodels).
=============================================================================
"""

import numpy as np
import pandas as pd


def hp_filter_gap(series: pd.Series, lamb: int = 400) -> np.ndarray:
    """
    Calcula la brecha entre una serie y su tendencia Hodrick-Prescott.

    La brecha (gap) = serie_original − tendencia_HP.
    Un gap positivo indica que la variable se encuentra por encima de su
    tendencia de largo plazo (señal de auge / boom).

    Implementación matricial pura en NumPy. No requiere statsmodels.

    Parámetros
    ----------
    series : pd.Series
        Serie temporal (puede contener NaN). Solo se procesan los valores
        no nulos; el resultado se realinea al índice original.
    lamb : int, default 400
        Parámetro de suavizado λ. Valores estándar:
            100  → datos anuales (convención HP original)
            400  → datos anuales (convención BIS para ciclo financiero)
            1600 → datos trimestrales

    Retorna
    -------
    np.ndarray
        Array del mismo tamaño que ``series`` con la brecha HP.
        Las posiciones que eran NaN en la entrada permanecen como NaN.

    Notas
    -----
    La solución se obtiene resolviendo el sistema lineal:
        (I + λ · DᵀD) · τ = y
    donde D es la matriz de segundas diferencias de dimensión (T-2)×T.
    """
    s = series.values.copy().astype(float)
    valid_mask = ~np.isnan(s)

    if valid_mask.sum() < 5:
        # Serie demasiado corta para estimar tendencia HP fiable
        return np.full(len(s), np.nan)

    y = s[valid_mask]
    T = len(y)

    # Matriz de segundas diferencias D ∈ R^{(T-2) × T}
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i]     =  1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] =  1.0

    # Resolver (I + λ DᵀD) τ = y
    A = np.eye(T) + lamb * D.T @ D
    trend = np.linalg.solve(A, y)
    gap   = y - trend

    # Reconstruir array completo con NaN en posiciones originales
    result = np.full(len(s), np.nan)
    result[valid_mask] = gap
    return result
