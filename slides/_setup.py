import os
import sys


def _setup_notebook():
    module_path = os.path.abspath(os.path.join(".."))
    if module_path not in sys.path:
        sys.path.append(module_path)
