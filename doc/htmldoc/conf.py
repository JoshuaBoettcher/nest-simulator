# -*- coding: utf-8 -*-
#
# conf.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


import sys
import os
import json
import subprocess

from urllib.request import urlretrieve

from pathlib import Path
from shutil import copyfile

# Add the extension modules to the path
extension_module_dir = os.path.abspath("./_ext")
sys.path.append(extension_module_dir)

from extractor_userdocs import ExtractUserDocs, relative_glob  # noqa

repo_root_dir = os.path.abspath("../..")
pynest_dir = os.path.join(repo_root_dir, "pynest")
# Add the NEST Python module to the path (just the py files, the binaries are mocked)
sys.path.append(pynest_dir)

# -- General configuration ------------------------------------------------

source_suffix = '.rst'
master_doc = 'index'
extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting',
    'nbsphinx',
    'sphinx_design',
    'HoverXTooltip',
    'VersionSyncRole',
]

autodoc_mock_imports = ["nest.pynestkernel", "nest.ll_api"]
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
panels_add_bootstrap_css = False
# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': '../../pynest/examples',
    # path where to save gallery generated examples
    'gallery_dirs': 'auto_examples',
    'plot_gallery': 'False',
    'download_all_examples': False,
}

# General information about the project.
project = u'NEST Simulator user documentation'
copyright = u'2004, nest-simulator'
author = u'nest-simulator'


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = [
    '**.ipynb_checkpoints',
    '.DS_Store',
    'README.md',
    'Thumbs.db',
    'auto_examples/**.ipynb',
    'auto_examples/index.rst',
    'nest_by_example',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'manni'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# add numbered figure link
numfig = True

numfig_secnum_depth = (2)
numfig_format = {'figure': 'Figure %s', 'table': 'Table %s',
                 'code-block': 'Code Block %s'}

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'
html_title = 'NEST Simulator Documentation'
html_logo = 'static/img/nest_logo.png'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    # Set the name of the project to appear in the navigation.
    # Set you GA account ID to enable tracking
    # 'google_analytics_account': 'UA-XXXXX',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://nest-simulator.readthedocs.io/en/latest/',
    'html_minify': False,
    'html_prettify': False,
    'css_minify': True,
    # Set the color and the accent color
    'color_primary': 'orange',
    'color_accent': 'white',
    'theme_color': 'ff6633',
    'master_doc': True,
    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/nest/nest-simulator/',
    'repo_name': 'NEST Simulator',
    # "nav_links": [
    #     {"href": "index", "internal": True, "title": "NEST docs home"}
    #     ],
    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 1,
    # If False, expand all TOC entries
    'globaltoc_collapse': True,
    # If True, show hidden TOC entries
    'globaltoc_includehidden': True,
    }

html_static_path = ['static']
html_additional_pages = {'index': 'index.html'}
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'NESTsimulatordoc'

html_show_sphinx = False
html_show_copyright = False

# This way works for ReadTheDocs
# With this local 'make html' is broken!
github_doc_root = ''

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'nestml': ('https://nestml.readthedocs.io/en/latest/', None),
    'pynn': ('https://neuralensemble.org/docs/PyNN/', None),
    'elephant': ('https://elephant.readthedocs.io/en/latest/', None),
    'desktop': ('https://nest-desktop.readthedocs.io/en/latest/', None),
    'gpu': ('https://nest-gpu.readthedocs.io/en/latest/', None),
    'neuromorph': ('https://electronicvisions.github.io/hbp-sp9-guidebook/', None),
    'arbor': ('https://docs.arbor-sim.org/en/latest/', None),
    'tvb': ('https://docs.thevirtualbrain.org/', None),
    'extmod': ('https://nest-extension-module.readthedocs.io/en/latest/', None),
}


def config_inited_handler(app, config):
    models_rst_dir = os.path.abspath("models")
    ExtractUserDocs(
        listoffiles=relative_glob("models/*.h", "nestkernel/*.h", basedir=repo_root_dir),
        basedir=repo_root_dir,
        outdir=models_rst_dir,
    )


def add_button_to_examples(app, env, docnames):
    """Find all examples and include a link to launch notebook.

     Function finds all restructured text files in auto_examples
     and injects the multistring prolog, which is rendered
     as a button link in HTML. The target is set to a Jupyter notebook of
     the same name and a service to run it.
     The nameholder in the string is replaced with the file name.

     The rst files are generated at build time by Sphinx_gallery.
     The notebooks that the target points to are linked with
     services (like EBRAINS JupyterHub) that runs notebooks using nbgitpuller.
     See https://hub.jupyter.org/nbgitpuller/link.html
     The notebooks are located in the repository nest/nest-simulator-examples/.
     The notebooks are generated from the CI workflow of NEST
     on GitHub, which converts the source Python files to .ipynb.

     The link to run the notebook is rendered in an image within a card directive.
    """
    example_prolog = """
.. only:: html

----

 Run this example as a Jupyter notebook:

  .. card::
    :width: 25%
    :margin: 2
    :text-align: center
    :link: https://lab.ebrains.eu/hub/user-redirect/\
git-pull?repo=https%3A%2F%2Fgithub.com%2Fnest%2Fnest-simulator-examples\&urlpath=lab\
%2Ftree%2Fnest-simulator-examples%2Fnotebooks%2Fnotebooks%2Ffilepath.ipynb&branch=main
    :link-alt: JupyterHub service

    .. image:: https://nest-simulator.org/TryItOnEBRAINS.png


.. grid:: 1 1 1 1
   :padding: 0 0 2 0

   .. grid-item::
     :class: sd-text-muted
     :margin: 0 0 3 0
     :padding: 0 0 3 0
     :columns: 4

     See :ref:`our guide <run_jupyter>` for more information and troubleshooting.

----
"""
    # Find all relevant files
    # Inject prolog into Python example
    files = list(Path("auto_examples/").rglob("*.rst"))
    for file in files:

        # Skip index files and benchmark file. These files do not have notebooks that can run
        # on the service.
        if file.stem == "index" or file.stem == "hpc_benchmark":
            continue

        with open(file, "r") as f:
            parent = Path("auto_examples/")
            path2example = os.path.relpath(file, parent)
            path2example = os.path.splitext(path2example)[0]
            path2example = path2example.replace("/", "%2F")
            prolog = example_prolog.replace("filepath", path2example)

            lines = f.readlines()

        # find the first heading of the file.
        for i, item in enumerate(lines):
            if item.startswith("-----"):
                break

        # insert prolog into rst file after heading
        lines.insert(i + 1, prolog + '\n')

        with open(file, 'w') as f:
            lines = "".join(lines)
            f.write(lines)


def toc_customizer(app, docname, source):
    if docname == "models/models-toc":
        models_toc = json.load(open("models/toc-tree.json"))
        html_context = {"nest_models": models_toc}
        models_source = source[0]
        rendered = app.builder.templates.render_string(models_source, html_context)
        source[0] = rendered


def setup(app):
    # for events see
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    app.connect("source-read", toc_customizer)
    app.add_css_file('css/custom.css')
    app.add_css_file('css/pygments.css')
    app.add_js_file("js/custom.js")
    app.connect('env-before-read-docs', add_button_to_examples)
    app.connect('config-inited', config_inited_handler)


nitpick_ignore = [('py:class', 'None'),
                  ('py:class', 'optional'),
                  ('py:class', 's'),
                  ('cpp:identifier', 'CommonSynapseProperties'),
                  ('cpp:identifier', 'Connection<targetidentifierT>'),
                  ('cpp:identifier', 'ArchivingNode'),
                  ('cpp:identifier', 'DeviceNode'),
                  ('cpp:identifier', 'Node'),
                  ('cpp:identifier', 'ClopathArchivingNode'),
                  ('cpp:identifier', 'MessageHandler'),
                  ('cpp:identifer', 'CommonPropertiesHomW')]

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'NESTsimulator.tex', u'NEST Simulator Documentation',
     u'NEST Developer Community', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'nestsimulator', u'NEST Simulator Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'NESTsimulator', u'NEST Simulator Documentation',
     author, 'NESTsimulator', 'One line description of project.',
     'Miscellaneous'),
]


def copy_example_file(src):
    copyfile(os.path.join(pynest_dir, src), Path("examples") / Path(src).parts[-1])


# -- Copy documentation for Microcircuit Model ----------------------------
copy_example_file("examples/Potjans_2014/box_plot.png")
copy_example_file("examples/Potjans_2014/raster_plot.png")
copy_example_file("examples/Potjans_2014/microcircuit.png")
copy_example_file("examples/hpc_benchmark_connectivity.svg")
copyfile(
    os.path.join(pynest_dir, "examples/Potjans_2014/README.rst"),
    "examples/README.rst",
)


def patch_documentation(patch_url):
    """Apply a hot-fix patch to the documentation before building it.

    This function is useful in situations where the online documentation should
    be modified, but the reason for a new documentation build does not justify
    a new release of NEST. Example situations are the discovery of broken links
    spelling errors, or styling issues. Moreover, this mechanism can be used to
    customize to the look and feel of the NEST documentation in individual
    installations.

    In order to make use of this function, the environment variable ``patch_url``
    has to be set to the URL where documentation patch files are located. The
    environment variable must either be set locally or via the admin panel of
    Read the Docs.

    Patch files under the ``patch_url`` are expected to have names in the format
    ``{git_hash}_doc.patch``, where ``{git_hash}`` is the full hash of the version the
    patch applies to.

    The basic algorithm implemented by this function is the following:
      1. obtain the Git hash of the version currently checked out
      2. log the hash by printing it to the console
      3. retrieve the patch

    """

    print("Preparing patch...")
    try:
        git_dir = repo_root_dir / ".git"
        git_hash = subprocess.check_output(
            f"GIT_DIR='{git_dir}' git rev-parse HEAD",
            shell=True,
            encoding='utf8').strip()
        print(f"  current git hash: {git_hash}")
        patch_file = f'{git_hash}_doc.patch'
        patch_url = f'{patch_url}/{patch_file}'
        print(f"  retrieving {patch_url}")
        urlretrieve(patch_url, patch_file)
        print(f"  applying {patch_file}")
        result = subprocess.check_output('patch -p3', stdin=open(patch_file, 'r'), stderr=subprocess.STDOUT, shell=True)
        print(f"Patch result: {result}")
    except Exception as exc:
        print(f"Error while applying patch: {exc}")


patch_url = os.getenv("patch_url")
if patch_url is not None:
    patch_documentation(patch_url)
