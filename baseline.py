from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_mod_path = Path(__file__).resolve().parent / 'lightgbm' / 'baseline.py'
_spec = spec_from_file_location('_local_lightgbm_baseline', _mod_path)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_mod, _name)
