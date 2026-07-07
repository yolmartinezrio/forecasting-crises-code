"""
=============================================================================
data/preprocessor.py
=============================================================================
Clase DataPreprocessor — construcción del panel maestro.

Responsabilidades (en orden de ejecución):
  1. Filtrar el panel JST por país y período de análisis.
  2. Construir la variable dependiente binaria con la cronología LV.
  3. Construir las variables forward para horizontes h = 1, 2, 3.
  4. Imputar valores ausentes (forward/backward fill dentro de país).
  5. Calcular variables derivadas: brechas HP, spread de tipos, growth.
  6. Seleccionar y exportar el panel maestro final.
  7. Exportar el diccionario de variables a CSV.
=============================================================================
"""

import numpy as np
import pandas as pd

from config.settings import (
    YEAR_MIN, YEAR_MAX,
    CRISIS_DURATION, FORECAST_HORIZONS,
    HP_LAMBDA,
    JST_COUNTRIES, FEATURE_GROUPS,
    IMPUTE_FFILL,
    JST_VARIABLE_DICT, DERIVED_VARIABLE_DICT,
    DATA_OUT,
)
from utils.filters import hp_filter_gap


class DataPreprocessor:
    """
    Transforma los DataFrames brutos (JST + LV) en el panel maestro listo
    para modelización.

    Parámetros
    ----------
    jst_raw : pd.DataFrame
        Panel JST completo tal como lo entrega DataLoader.
    lv_long : pd.DataFrame
        Episodios de crisis LV en formato largo (DataLoader.lv_long).

    Uso típico
    ----------
    >>> prep = DataPreprocessor(loader.jst_raw, loader.lv_long)
    >>> prep.build()
    >>> panel = prep.panel_master
    >>> feature_list = prep.feature_list
    """

    def __init__(self, jst_raw: pd.DataFrame, lv_long: pd.DataFrame) -> None:
        self._jst      = jst_raw.copy()
        self._lv_long  = lv_long.copy()
        self.panel_master: pd.DataFrame | None = None
        self.feature_list: list[str] = []
        self._built = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Ejecuta el pipeline completo de construcción del panel."""
        print("[DataPreprocessor] Construyendo panel maestro…")
        panel = self._filter_panel()
        panel = self._build_crisis_variable(panel)
        panel = self._build_forward_targets(panel)
        panel = self._impute(panel)
        panel = self._add_derived_features(panel)
        panel, features = self._select_columns(panel)

        self.panel_master = panel
        self.feature_list = features
        self._built = True

        self._print_summary()
        self._export()
        print("[DataPreprocessor] Panel maestro listo.")

    # ------------------------------------------------------------------
    # Pasos del pipeline (privados)
    # ------------------------------------------------------------------

    def _filter_panel(self) -> pd.DataFrame:
        """Filtra por período y lista de países JST."""
        panel = self._jst.copy()
        panel = panel[
            (panel["year"] >= YEAR_MIN) &
            (panel["year"] <= YEAR_MAX) &
            (panel["country"].isin(JST_COUNTRIES))
        ].copy()
        print(
            f"  Filtro temporal + países → "
            f"{len(panel):,} obs · {panel['country'].nunique()} países"
        )
        return panel

    def _build_crisis_variable(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Construye la variable dependiente principal: crisis_bin.

        crisis_bin = 1 si el año t del país i está dentro de un episodio
        de crisis sistémica según LV (duración asumida = CRISIS_DURATION años
        a partir del año de inicio).

        También construye:
          - crisis_lv       : misma lógica que crisis_bin (alias semántico)
          - crisisJST_fill  : crisisJST original con NaN → 0
          - crisis_union    : 1 si crisis_lv OR crisisJST
        """
        panel["crisis_lv"] = 0

        for _, row in self._lv_long.iterrows():
            country = row["country"]
            start   = row["crisis_start_lv"]
            end     = start + CRISIS_DURATION - 1
            mask = (
                (panel["country"] == country) &
                (panel["year"] >= start) &
                (panel["year"] <= end)
            )
            panel.loc[mask, "crisis_lv"] = 1

        panel["crisisJST_fill"] = panel["crisisJST"].fillna(0).astype(int)
        panel["crisis_union"]   = (
            (panel["crisis_lv"] == 1) | (panel["crisisJST_fill"] == 1)
        ).astype(int)
        panel["crisis_bin"] = panel["crisis_lv"].astype(int)

        n_pos = panel["crisis_bin"].sum()
        pct   = 100 * n_pos / len(panel)
        print(
            f"  Variable crisis_bin → {n_pos} obs positivas "
            f"({pct:.1f}%)  |  ratio desbalance 1:{int(len(panel)/n_pos)}"
        )
        return panel

    def _build_forward_targets(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Construye variables adelantadas crisis_h{h} para h en FORECAST_HORIZONS.

        crisis_h{h} = 1 para la observación (i, t) si el país i INICIA
        una crisis sistémica en el intervalo (t, t+h], es decir, en los
        próximos h años. Solo marca el INICIO de la crisis (no los años
        de crisis activa), lo que permite al modelo aprender a anticipar
        el evento antes de que ocurra.

        Nota: la exclusión de la ventana de crisis activa durante el
        entrenamiento ("crisis window exclusion") se aplica en el módulo
        de modelización, no aquí.
        """
        for h in FORECAST_HORIZONS:
            col = f"crisis_h{h}"
            panel[col] = 0
            for _, row in self._lv_long.iterrows():
                country = row["country"]
                start   = row["crisis_start_lv"]
                # Observaciones para las que el inicio está en (t, t+h]
                # equivalente a: t ∈ [start-h, start-1]
                mask = (
                    (panel["country"] == country) &
                    (panel["year"] >= start - h) &
                    (panel["year"] <  start)
                )
                panel.loc[mask, col] = 1

            n_pos = panel[col].sum()
            pct   = 100 * n_pos / len(panel)
            print(f"  {col} → {n_pos:3d} obs positivas ({pct:.1f}%)")

        return panel

    def _impute(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores ausentes con forward-fill seguido de backward-fill
        dentro de cada país, para variables de stock con datos anuales.

        Solo se imputan las variables listadas en IMPUTE_FFILL.
        Las variables con muchos NaN estructurales (p. ej. thh, tbus)
        se dejan sin imputar y el modelo de modelización las gestiona.
        """
        available = [v for v in IMPUTE_FFILL if v in panel.columns]
        panel = panel.sort_values(["country", "year"])
        for v in available:
            panel[v] = (
                panel.groupby("country")[v]
                     .transform(lambda s: s.ffill().bfill())
            )
        print(f"  Imputación ffill/bfill → {len(available)} variables")
        return panel

    def _add_derived_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula variables derivadas que complementan los predictores base:

          - tloans_gap    : brecha crédito/PIB (HP gap, λ=HP_LAMBDA)
          - hpnom_gap     : brecha precios nominales de vivienda (HP gap)
          - term_spread   : diferencial ltrate − stir (pendiente de la curva)
          - tloans_growth : tasa de crecimiento anual de tloans
        """
        panel = panel.sort_values(["country", "year"])

        # Brechas HP
        for gap_var, src_var in [("tloans_gap", "tloans"),
                                  ("hpnom_gap",  "hpnom")]:
            if src_var not in panel.columns:
                continue
            panel[gap_var] = np.nan
            for country, grp in panel.groupby("country"):
                gap = hp_filter_gap(grp[src_var], lamb=HP_LAMBDA)
                panel.loc[grp.index, gap_var] = gap
            n_ok = panel[gap_var].notna().sum()
            print(f"  {gap_var} (HP gap) → {n_ok} obs calculadas")

        # Pendiente de la curva de tipos
        if {"ltrate", "stir"}.issubset(panel.columns):
            panel["term_spread"] = panel["ltrate"] - panel["stir"]
            n_ok = panel["term_spread"].notna().sum()
            print(f"  term_spread (ltrate−stir) → {n_ok} obs calculadas")

        # Crecimiento del crédito bancario
        if "tloans" in panel.columns:
            panel["tloans_growth"] = (
                panel.groupby("country")["tloans"].pct_change()
            )
            n_ok = panel["tloans_growth"].notna().sum()
            print(f"  tloans_growth → {n_ok} obs calculadas")

        return panel

    def _select_columns(
        self, panel: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Construye la lista de features final (base + derivadas) y
        selecciona las columnas del panel maestro exportado.
        """
        # Features base de cada grupo
        base_features = [
            v for grp in FEATURE_GROUPS.values()
            for v in grp["vars"]
        ]
        # Añadir features derivadas si existen
        derived = ["tloans_gap", "hpnom_gap", "term_spread", "tloans_growth"]
        all_features = list(dict.fromkeys(
            base_features + [d for d in derived if d in panel.columns]
        ))

        # Columnas identificadoras + targets + features
        id_cols = ["year", "country", "iso"]
        target_cols = (
            ["crisis_bin", "crisis_lv", "crisisJST", "crisis_union"]
            + [f"crisis_h{h}" for h in FORECAST_HORIZONS]
        )
        keep = id_cols + target_cols + [
            f for f in all_features if f in panel.columns
        ]
        # Eliminar duplicados preservando orden
        keep = list(dict.fromkeys(keep))

        panel_out = panel[keep].copy()
        features_out = [f for f in all_features if f in panel_out.columns]

        return panel_out, features_out

    # ------------------------------------------------------------------
    # Exportación
    # ------------------------------------------------------------------

    def _export(self) -> None:
        """Guarda el panel maestro y el diccionario de variables en disco."""
        if self.panel_master is None:
            return

        csv_path  = DATA_OUT / "panel_maestro.csv"
        xlsx_path = DATA_OUT / "panel_maestro.xlsx"
        dict_path = DATA_OUT / "diccionario_variables.csv"

        self.panel_master.to_csv(csv_path,  index=False)
        self.panel_master.to_excel(xlsx_path, index=False)
        print(f"  Panel guardado → {csv_path.name} / {xlsx_path.name}")

        self._export_variable_dict(dict_path)

    def _export_variable_dict(self, path) -> None:
        """Construye y guarda el diccionario de variables en CSV."""
        rows = []
        derived_vars = {"tloans_gap", "hpnom_gap", "term_spread", "tloans_growth"}

        for grp_name, grp_info in FEATURE_GROUPS.items():
            group_vars = list(grp_info["vars"])
            # Añadir derivadas que pertenecen conceptualmente a este grupo
            if grp_name == "CRÉDITO_BANCARIO":
                group_vars += ["tloans_gap", "tloans_growth"]
            elif grp_name == "PRECIOS_ACTIVOS":
                group_vars += ["hpnom_gap"]
            elif grp_name == "TIPOS_INTERÉS":
                group_vars += ["term_spread"]

            for v in group_vars:
                if v in DERIVED_VARIABLE_DICT:
                    desc, unit, rel = DERIVED_VARIABLE_DICT[v]
                elif v in JST_VARIABLE_DICT:
                    desc, unit, rel = JST_VARIABLE_DICT[v]
                else:
                    desc, unit, rel = "Variable derivada", "—", "MEDIA"

                rows.append({
                    "grupo"        : grp_name,
                    "variable"     : v,
                    "descripcion"  : desc,
                    "unidad"       : unit,
                    "relevancia"   : rel,
                    "justificacion": grp_info["doc"][:300],
                })

        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Diccionario guardado → {path.name}")

    # ------------------------------------------------------------------
    # Resumen en consola
    # ------------------------------------------------------------------

    def _print_summary(self) -> None:
        p = self.panel_master
        print(
            f"\n  ── PANEL MAESTRO ──────────────────────────────\n"
            f"  Observaciones  : {len(p):,}\n"
            f"  Variables      : {p.shape[1]}\n"
            f"  Países         : {p['country'].nunique()}\n"
            f"  Período        : {p['year'].min()} – {p['year'].max()}\n"
            f"  Features       : {len(self.feature_list)}\n"
            f"  Crisis activas : {p['crisis_bin'].sum()} obs "
            f"({100*p['crisis_bin'].mean():.1f}%)\n"
            f"  ───────────────────────────────────────────────"
        )
