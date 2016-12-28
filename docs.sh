# Documentation standard: http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

pip3 install Sphinx sphinx_rtd_theme
cd docs
sphinx-apidoc -f -o . ../g3py
make clean
make html