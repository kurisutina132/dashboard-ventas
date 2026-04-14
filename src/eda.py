import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

def plot_and_save(fig, name, output_dir):
    path = os.path.join(output_dir, name)
    fig.savefig(path)
    logging.info(f"Figura guardada en {path}")

def eda(df: pd.DataFrame, output_dir: str):
    """Análisis exploratorio automatizado."""
    logging.info("Iniciando EDA")
    # Histograma de unidades vendidas
    fig, ax = plt.subplots()
    sns.histplot(df['unidades_vendidas'], ax=ax)
    ax.set_title('Histograma de Unidades Vendidas')
    plot_and_save(fig, 'hist_unidades.png', output_dir)
    plt.close(fig)

    # Boxplot de precios
    if 'precio_venta' in df.columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df['precio_venta'], ax=ax)
        ax.set_title('Boxplot de Precio de Venta')
        plot_and_save(fig, 'box_precio.png', output_dir)
        plt.close(fig)

    # Correlación
    fig, ax = plt.subplots()
    sns.heatmap(df.select_dtypes('number').corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Heatmap de Correlaciones')
    plot_and_save(fig, 'heatmap_corr.png', output_dir)
    plt.close(fig)
