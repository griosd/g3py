import numpy as np
import g3py as g3


# C02 Concentration
name_data = 'MaunaLoaC02'
x, y = g3.data_co2()
y = y[~np.isnan(y)]
y = y-y[0]
x = np.arange(len(y))[:,None]
obs_j, x_obs, y_obs, test_j, x_test, y_test = g3.random_obs(x, y, 0.3, 0.7)


kernel = g3.WN(x)
gp = g3.GP(space=x, location=g3.Zero(), kernel=kernel)#, file='models/02-'+name_data+'-'+kernel.name+'.g3')
gp.describe(name_data,'month','CO2')
gp.set_space(x,y)
print(gp.compiles)

_ = gp.predict()
print(gp.compiles)

gp.plot(samples=5, data=False, prior=True, title='Prior')
print(gp.compiles)

gp.observed(x_obs,y_obs)
_ = gp.predict()
print(gp.compiles)

gp.plot(samples=5, title='Posterior')
print(gp.compiles)