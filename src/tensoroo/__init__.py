# tensoroo/__init__.py
"""Top-level API for the tensoroo package."""

import pkgutil
import importlib
import inspect

__all__ = []

for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f'.{module_name}', __name__)
    for name, func in inspect.getmembers(module, inspect.isfunction):
        globals()[name] = func
        __all__.append(name)