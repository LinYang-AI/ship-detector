import ruamel.yaml as yaml
from typing import Dict, Any


def load_config(config_path: str) ->  Dict[str, Any]:
    with open(config_path, 'r') as f:
        config = yaml.YAML(typ='rt').load(f)
    return config