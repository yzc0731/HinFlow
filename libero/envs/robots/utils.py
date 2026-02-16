import os

assets_dir = os.path.join(os.path.dirname(__file__), '../../assets')

def xml_path_completion(path):
    return os.path.join(assets_dir, path)

