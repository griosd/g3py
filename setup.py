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
          'numpy>=1.13', 'scipy>=1.0', 'pandas>=0.21', 'matplotlib>=2.1', 'seaborn>=0.8.1', 'theano>=1.0', 'pymc3>=3.2', 'emcee>=2.2', 'scikit-learn>=0.19', 'statsmodels', 'ipywidgets'
      ],
      extras_require={
              'load_data':  ['statsmodels']
      },
      tests_require=[
            'pytest', 'pytest-mpl'
      ],
      zip_safe=False)
