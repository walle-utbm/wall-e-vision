import sys
import pprint

try:
    import pysnpe_utils
except ImportError as e:
    print(f"Erreur d'importation de pysnpe_utils : {e}")
    sys.exit(1)

print("=== Attributs publics via dir() ===")
pprint.pprint([a for a in dir(pysnpe_utils) if not a.startswith("_")])

print("\n=== Attributs internes (commençant par __) ===")
pprint.pprint([a for a in dir(pysnpe_utils) if a.startswith("__")])

print("\n=== Documentation native générée par Pybind ===")
# help() va lire les docstrings C++ injectées par Pybind11
help(pysnpe_utils)