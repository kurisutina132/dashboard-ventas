import argparse
import os
from src.utils import setup_logging, load_config
from src.data import load_data, clean_data
from src.features import feature_engineering
from src.eda import eda
from src.report import generate_report

def main(args):
    config = load_config(args.config)
    setup_logging(config['log_path'])

    df = load_data(args.input)
    df = clean_data(df)
    df = feature_engineering(df)
    eda(df, config['output_figures'])
    generate_report()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline profesional de análisis de datos')
    parser.add_argument('--input', type=str, help='Ruta al archivo de datos', default='data/raw/retail.csv')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración', default='config.yaml')
    args = parser.parse_args()
    main(args)
