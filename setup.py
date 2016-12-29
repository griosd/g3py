from setuptools import setup

setup(name='g3py',
      version='0.2.4',
      description='Generalized Graphical Gaussian Processes',
      url='https://github.com/griosd/g3py',
      author='Gonzalo Rios',
      author_email='grios@dim.uchile.cl',
      license='MIT',
      packages=['g3py'],
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'theano', 'pymc3'
      ],
      tests_require=[
            'pytest', 'pytest-mpl'
      ],
      zip_safe=False)
