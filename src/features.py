import pandas as pd
import logging

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Genera nuevas variables relevantes para el negocio."""
    logging.info("Generando features")
    if 'precio_venta' in df.columns and 'precio_base' in df.columns:
        df['descuento_pct'] = ((df['precio_venta'] - df['precio_base']) / df['precio_base']) * 100
    # Añade más features según el negocio
    return df
