"""
=============================================================================
data/loader.py
=============================================================================
Clase DataLoader — carga y validación de las fuentes de datos brutas.

Responsabilidades:
  1. Leer el panel JST desde Excel y realizar validaciones básicas.
  2. Leer y parsear la base Laeven & Valencia (LV) desde Excel.
  3. Exponer los datos en DataFrames limpios para su procesamiento posterior.

No aplica transformaciones de negocio: eso es responsabilidad de
DataPreprocessor (data/preprocessor.py).
=============================================================================
"""

import pandas as pd
from pathlib import Path

from config.settings import (
    JST_FILE, LV_FILE, LV_SHEET,
    LV_TO_JST, JST_COUNTRIES,
)


class DataLoader:
    """
    Carga las fuentes de datos brutas y las expone como DataFrames validados.

    Uso típico
    ----------
    >>> loader = DataLoader()
    >>> loader.load_all()
    >>> jst = loader.jst_raw
    >>> lv  = loader.lv_long
    """

    def __init__(self) -> None:
        self.jst_raw:  pd.DataFrame | None = None   # Panel JST completo
        self.lv_long:  pd.DataFrame | None = None   # LV en formato largo
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Carga JST y LV. Llamar antes de acceder a los atributos."""
        print("[DataLoader] Cargando fuentes de datos…")
        self.jst_raw = self._load_jst()
        self.lv_long = self._load_lv()
        self._loaded = True
        print("[DataLoader] Carga completada.")

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _load_jst(self) -> pd.DataFrame:
        """
        Lee el panel JST desde Excel y realiza validaciones básicas.

        Retorna
        -------
        pd.DataFrame
            Panel JST con todas las columnas originales.

        Raises
        ------
        FileNotFoundError
            Si el archivo no existe en la ruta configurada.
        ValueError
            Si faltan columnas mínimas obligatorias.
        """
        if not JST_FILE.exists():
            raise FileNotFoundError(
                f"Archivo JST no encontrado: {JST_FILE}\n"
                "Asegúrate de que 'JSTdatasetR6.xlsx' está en DATA_DIR."
            )

        df = pd.read_excel(JST_FILE)

        # Validar columnas mínimas obligatorias
        required = {"year", "country", "iso", "crisisJST"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Columnas obligatorias ausentes en JST: {missing}"
            )

        print(
            f"  JST → {len(df):,} obs · {df.shape[1]} vars · "
            f"{df['country'].nunique()} países · "
            f"{df['year'].min()}–{df['year'].max()}"
        )
        return df

    def _load_lv(self) -> pd.DataFrame:
        """
        Lee la hoja 'Crisis Years' de Laeven & Valencia y devuelve
        un DataFrame largo con una fila por (país_JST, año_inicio_crisis).

        La columna de años puede contener varios valores separados por coma
        (p. ej. '1977, 2008'), que se separan en filas individuales.

        Retorna
        -------
        pd.DataFrame
            Columnas: ['country', 'crisis_start_lv']
        """
        if not LV_FILE.exists():
            raise FileNotFoundError(
                f"Archivo LV no encontrado: {LV_FILE}\n"
                "Asegúrate de que 'SYSTEMIC_BANKING_CRISES_DATABASE_2018.xlsx' "
                "está en DATA_DIR."
            )

        raw = pd.read_excel(LV_FILE, sheet_name=LV_SHEET)
        raw.columns = ["country_lv", "crisis_years_str"]
        raw = raw.dropna(subset=["country_lv"])

        records: list[dict] = []
        for _, row in raw.iterrows():
            lv_name = str(row["country_lv"]).strip()
            if lv_name not in LV_TO_JST:
                continue
            jst_name = LV_TO_JST[lv_name]
            for year in self._parse_year_cell(row["crisis_years_str"]):
                records.append({"country": jst_name, "crisis_start_lv": year})

        lv_long = pd.DataFrame(records)
        print(
            f"  LV  → {len(lv_long)} episodios en países JST "
            f"({lv_long['country'].nunique()} países con ≥1 crisis)"
        )
        return lv_long

    @staticmethod
    def _parse_year_cell(value) -> list[int]:
        """
        Convierte el contenido de una celda de año(s) en lista de enteros.

        Ejemplos
        --------
        '1991'        → [1991]
        '1977, 2008'  → [1977, 2008]
        NaN           → []
        """
        if pd.isna(value):
            return []
        return [
            int(token.strip())
            for token in str(value).split(",")
            if token.strip().isdigit()
        ]
