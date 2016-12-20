# g3py
g3py: Generalized Graphical Gaussian Processes with Theano and PyMC3 Backend

g3py es una librería para modelar procesos estocásticos. Sus características son:
* Definir Gaussian Processes de forma simple e intuitiva.
* Definir Transformed Gaussian Processes de forma natural.
* Gran eficiencia gracias al backend de Theano al optimizar el grafo de computación para luego compilarlo utilizando C (CPU) o CUDA (GPU).
* Modelos bayesianos gracias al backend de PyMC3, el cual permite definir distribuciones a priori sobre los hiperparámetros.
* Entrenamiento basado en hiperparámetros iniciales automáticos, widgets para manipular sus valores,
búsqueda de óptimos locales y recorrer el espacio en búsqueda de óptimos globales utilizando técnicas de MCMC.
* Métodos con derivadas de forma gratuita gracias al motor de diferenciación simbólica de Theano.

# Installation
pip install g3py