"""Sphinx configuration for qiskit-qm-provider docs."""

from __future__ import annotations

import importlib
import inspect
import os
import re
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    release = version("qiskit-qm-provider")
except PackageNotFoundError:
    release = "0.1.1"

version = ".".join(release.split(".")[:2])

project = "Qiskit QM Provider"
author = "Arthur Strauss"
copyright = f"{datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.viewcode",
    "qiskit_sphinx_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

root_doc = "index"
exclude_patterns = ["_build", "README.md", "Thumbs.db", ".DS_Store"]

suppress_warnings = ["myst.xref_missing"]

templates_path = ["_templates"]

html_theme = "qiskit-ecosystem"
html_title = f"{project} {release}"
html_theme_options = {
    "sidebar_qiskit_ecosystem_member": True,
}

html_last_updated_fmt = "%Y/%m/%d"
language = "en"

add_module_names = False
modindex_common_prefix = ["qiskit_qm_provider."]

autosummary_generate = True
autosummary_generate_overwrite = True

autoclass_content = "both"
autodoc_typehints = "signature"
autodoc_use_type_comments = False
autodoc_default_options = {
    "show-inheritance": True,
    "members": True,
    "member-order": "bysource",
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_preprocess_types = True

autodoc_mock_imports = [
    "qiskit.pulse",
    "qiskit.pulse.transforms",
    "qiskit.pulse.library.pulse",
    "qiskit.pulse.channels",
    "symengine",
]

intersphinx_mapping = {
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "quam": ("https://qua-platform.github.io/quam/", None),
    "python": ("https://docs.python.org/3/", None),
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links."""
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return (
        f"stable/{version_without_patch.group()}"
        if version_without_patch
        else ref_name
    )


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    module_name = info["module"]
    if "qiskit_qm_provider" not in module_name:
        return None

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    while hasattr(obj, "__wrapped__"):
        obj = getattr(obj, "__wrapped__")

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None:
        return None

    try:
        relative_file_name = Path(full_file_name).resolve().relative_to(REPO_ROOT)
        file_name = re.sub(
            r"\.tox/.+/site-packages/", "", relative_file_name.as_posix()
        )
    except ValueError:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"

    return (
        f"https://github.com/arthurostrauss/qiskit-qm-provider/tree/"
        f"{GITHUB_BRANCH}/{file_name}{linespec}"
    )
