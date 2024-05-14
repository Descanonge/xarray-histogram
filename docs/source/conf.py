# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import xarray_histogram

## Project information

project = "Xarray-Histogram"
copyright = "2021, Clément Haëck"
author = "Clément Haëck"

release = xarray_histogram.__version__
version = xarray_histogram.__version__
print(f"xarray-histogram: {version}")

## General configuration

templates_path = ["_templates"]
# exclude_patterns = []
pygments_style = "default"

nitpicky = True

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_type = False

add_module_names = False

# Autosummary config
autosummary_generate = ["api.rst"]

# Autodoc config
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Intersphinx config
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "boost-histogram": ("https://boost-histogram.readthedocs.io/en/latest", None),
    "dask-histogram": ("https://dask-histogram.readthedocs.io/en/stable", None),
}

## HTML Output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "Xarray-Histogram"
html_theme_options = dict(
    collapse_navigation=False,
    use_download_button=True,
    use_fullscreen_button=False,
    # TOC
    show_toc_level=2,
    # Link to source in repo
    repository_url="https://github.com/Descanonge/xarray-histogram",
    use_source_button=True,
    repository_branch="master",
    path_to_docs="doc",
    # Social icons
    icon_links=[
        dict(
            name="Repository",
            url="https://github.com/Descanonge/xarray-histogram",
            icon="fa-brands fa-github",
        ),
        dict(
            name="PyPI",
            url="https://pypi.org/xarray-histogram",
            icon="fa-brands fa-python",
        ),
    ],
    # Footer
    article_footer_items=["prev-next"],
    content_footer_items=[],
    footer_start=["footer-left"],
    footer_end=["footer-right"],
)
html_last_updated_fmt = "%Y-%m-%d"

html_sidebars = {"**": ["navbar-logo.html", "sbt-sidebar-nav.html", "icon-links.html"]}
