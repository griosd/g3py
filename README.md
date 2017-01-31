# g3py
Generalized Graphical Gaussian Processes with Theano and PyMC3 Backend

g3py es una librería para modelar procesos estocásticos. Sus características son:
* Definir Gaussian Processes de forma simple e intuitiva.
* Definir Transformed Gaussian Processes de forma natural.
* Gran eficiencia gracias al backend de Theano al optimizar el grafo de computación y luego compilarlo para CPU y/o GPU.
* Modelos bayesianos gracias al backend de PyMC3, el cual permite definir distribuciones a priori sobre los hiperparámetros.
* Entrenamiento basado en hiperparámetros iniciales automáticos, widgets para manipular sus valores,
búsqueda de óptimos locales y recorrer el espacio en búsqueda de óptimos globales utilizando técnicas de MCMC.
* Métodos con derivadas (BFGS para la optimización, HamiltonianMC para el sampling) de forma gratuita, gracias al motor de diferenciación simbólica de Theano

# Installation
[https://pypi.python.org/pypi/g3py](https://pypi.python.org/pypi/g3py)
```
pip install git+https://github.com/griosd/g3py.git
```

# Tutorials
0. [Introduction](https://github.com/griosd/g3py/blob/master/notebooks/00-Introduction.ipynb)
1. [Gaussian Processes](https://github.com/griosd/g3py/blob/master/notebooks/01-Gaussian-Processes.ipynb)
2. [Hyperparameters](https://github.com/griosd/g3py/blob/master/notebooks/02-Hyperparameters.ipynb)
3. [Kernels](https://github.com/griosd/g3py/blob/master/notebooks/03-Kernels.ipynb)
4. [Random Fields](https://github.com/griosd/g3py/blob/master/notebooks/04-Random-Fields.ipynb)
5. [Pushforward](https://github.com/griosd/g3py/blob/master/notebooks/05-Pushforward.ipynb)
6. [Copulas](https://github.com/griosd/g3py/blob/master/notebooks/06-Copulas.ipynb)
7. [Multi Output](https://github.com/griosd/g3py/blob/master/notebooks/07-Multi-Output.ipynb)
8. [Graphical Models](https://github.com/griosd/g3py/blob/master/notebooks/08-Graphical-Models.ipynb)

# Documentation
