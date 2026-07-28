from pathlib import Path
import os

def get_root_dir():
    return Path(__file__).parent.parent
# root_dir = Path(__file__).parent.parent
prefix = os.path.join('datasets', 'processed')