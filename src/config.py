import toml
import os

class Config:
    _instance = None

    def __new__(cls, config_path="settings.toml"):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            self._config = toml.load(f)

    def get(self, key: str, default=None):
        """
        Retrieves a configuration value using a dot-separated key (e.g., "llm.model_name").
        """
        parts = key.split('.')
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

# Initialize the config instance
config = Config()
