"""Sphinx configuration for qiskit-qm-provider docs."""

from __future__ import annotations

from datetime import datetime

project = "Qiskit QM Provider"
author = "Arthur Strauss"
copyright = f"{datetime.now().year}, {author}"
release = "0.1.1"
version = release

extensions = [
    "myst_parser",
    "qiskit_sphinx_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "sidebar_qiskit_ecosystem_member": True,
}

