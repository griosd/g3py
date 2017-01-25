from setuptools import setup, find_packages

setup(name='g3py',
      version='0.2.4.9',
      description='Generalized Graphical Gaussian Processes',
      url='https://github.com/griosd/g3py',
      author='Gonzalo Rios',
      author_email='grios@dim.uchile.cl',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'theano', 'pymc3', 'scikit-learn', 'statsmodels'
      ],
      tests_require=[
            'pytest', 'pytest-mpl'
      ],
      zip_safe=False)
