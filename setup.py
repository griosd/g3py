from setuptools import setup, find_packages
exec(open('g3py/version.py').read())

setup(name='g3py',
      version=__version__,
      description='Generalized Graphical Gaussian Processes',
      url='https://github.com/griosd/g3py',
      author='Gonzalo Rios',
      author_email='grios@dim.uchile.cl',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'theano>=0.9', 'pymc3', 'emcee', 'scikit-learn>=0.18', 'statsmodels', 'ipywidgets'
      ],
      extras_require={
              'load_data':  ['statsmodels']
      },
      tests_require=[
            'pytest', 'pytest-mpl'
      ],
      zip_safe=False)
