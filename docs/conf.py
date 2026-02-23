import os, sys

sys.path.insert(0, os.path.abspath(".."))  # points to the pyport_sim root

project = "PyPort Sim"
copyright = "2026, Toni Garcês"
author = "Toni Garcês"
release = "1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.apidoc",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

apidoc_modules = [
    {
        "path": "..",
        "destination": "api",
        "exclude_patterns": [
            "**/venv/*",
            "**/tests/*",
            "**/assets/*",
            "**/streamlit_app.py",
        ],
        "separate_modules": True,
    },
]
