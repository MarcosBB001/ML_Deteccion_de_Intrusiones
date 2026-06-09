import yaml

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
