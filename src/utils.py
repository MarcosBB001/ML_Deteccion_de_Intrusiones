import yaml
import numpy as np

def load_config(path):
    """
        Load a YAML configuration file.

        Args:
            path (str): Path to the YAML file.

        Returns:
            dict: Configuration dictionary loaded from the YAML file.
    """
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config

def aggregate(values):
        return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}

def make_range(value, options, n=3):
    """Create a small grid around the best value found by random search."""
    idx = options.index(value) if value in options else 0
    return options[max(0, idx - 1): idx + 2]