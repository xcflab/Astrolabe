import importlib
import pkgutil
from pathlib import Path
import logging


current_dir = Path(__file__).parent
for module_info in pkgutil.iter_modules([str(current_dir)]):
    if module_info.name not in ["__init__", "base", "utils"]:
        try:
            logging.debug("Importing loss module: %s", module_info.name)
            importlib.import_module(f".{module_info.name}", package=__name__)
        except Exception as e:
            # Many loss modules depend on optional heavy deps (pyrosetta, biopython, rdkit, ...).
            # Keep import best-effort so lightweight runs (e.g., container smoke tests) can proceed.
            logging.debug("Skipping loss module %s: %s", module_info.name, e)
