import yaml
import os

def load_env_file(path):
    with open(path, 'r') as stream:
        try:
            env = yaml.safe_load(stream)
            for key, value in env.items():
                os.environ[key] = value
        except yaml.YAMLError as exc:
            print(exc)