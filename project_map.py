"""
project_map.py
Utility to inspect project structure, imports, and required paths.
Run: python project_map.py
"""

import os
import pkgutil
import importlib
import inspect

# Import your config directly
from app import config


def walk_modules(package_name):
	"""Yield all modules in a package without executing them fully."""
	package = importlib.import_module(package_name)
	prefix = package.__name__ + "."
	for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, prefix):
		yield modname


def check_paths():
	"""Verify that checkpoint and log directories exist, create if needed."""
	paths = {
		"log_dir": config.CONFIG.get("log_dir"),
		"checkpoint_dir": config.CONFIG.get("checkpoint_dir"),
	}

	print("\n📂 Path Check")
	# Handle dirs
	for name, path in paths.items():
		if path:
			if not os.path.isdir(path):
				print(f"  ⚠️ {name}: {path} missing → creating...")
				os.makedirs(path, exist_ok=True)
			else:
				print(f"  ✅ {name}: {path} exists")

	# Handle checkpoint files
	checkpoints = {
		"resume_from_checkpoint_A": config.CONFIG.get("resume_from_checkpoint_A"),
		"resume_from_checkpoint_B": config.CONFIG.get("resume_from_checkpoint_B"),
	}

	for name, path in checkpoints.items():
		if path:
			if os.path.isfile(path):
				print(f"  ✅ {name}: {path} exists")
			else:
				print(f"  ⚠️ {name}: {path} is missing (expected file)")


def inspect_imports(module_name):
	"""Show top-level imports for a module."""
	try:
		mod = importlib.import_module(module_name)
		print(f"\n🔹 {module_name}")
		for name, obj in inspect.getmembers(mod):
			if inspect.ismodule(obj) and obj.__package__:
				if obj.__package__.startswith("app"):
					print(f"   ↳ imports {obj.__package__}")
	except Exception as e:
		print(f"  ⚠️ Could not import {module_name}: {e}")


if __name__ == "__main__":
	print("🔍 Project Map Inspector")
	print("========================")

	# 1. Walk and print all modules under app/
	print("\n📦 Modules under app/:")
	for mod in walk_modules("app"):
		print(f" - {mod}")
		inspect_imports(mod)

	# 2. Check config paths and auto-create dirs
	check_paths()

	print("\n✅ Inspection complete.")
