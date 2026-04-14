import logging
import yaml

def setup_logging(log_path):
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
