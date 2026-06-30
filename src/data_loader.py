from pathlib import Path

import pandas as pd
import pyreadstat


FEATURE_COLS = [
    "Día",
    "Mes",
    "Año",
    "Estación",
    "País",
    "Ciudad",
    "CalleLugar",
    "NumeroPiso",
    "Miguel2",
    "González2",
    "Avenida2",
    "Imperial2",
    "A682",
    "Caldera2",
    "Copiapo2",
]

TARGETS = ["GDS", "GDS_R1", "GDS_R2", "GDS_R3", "GDS_R4", "GDS_R5"]


def load_sav_dataset(dataset_path: str | Path) -> pd.DataFrame:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset configurado: {path}")

    df, _ = pyreadstat.read_sav(str(path))
    _validate_columns(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    required_cols = ["ID", *FEATURE_COLS, *TARGETS]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas en el .sav: {missing_cols}")


def prepare_xy(df: pd.DataFrame, target_name: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_name not in TARGETS:
        raise ValueError(f"Objetivo no reconocido: {target_name}")

    X = df[FEATURE_COLS].astype(float)
    y = df[target_name].astype(int)
    return X, y
