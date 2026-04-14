import pandas as pd
import logging

def load_data(path: str) -> pd.DataFrame:
    """Carga datos desde un archivo CSV."""
    logging.info(f"Cargando datos desde {path}")
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica de datos: elimina duplicados y nulos."""
    logging.info("Eliminando duplicados y nulos")
    df = df.drop_duplicates()
    df = df.dropna()
    return df
