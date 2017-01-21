import numpy as np
import pandas as pd
import statsmodels.api as sm


def load_sunspots():
    data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    print(sm.datasets.sunspots.NOTE)
    x = data.index.values[:]
    y = data.values[:]
    return x, y


def load_co2():
    data = sm.datasets.get_rdataset('co2', cache=True)
    print(data.__doc__)
    x = data.data.time.values[:]
    y = data.data.co2.values[:]
    return x, y


def save_csv(df, file, index_col=0):
    return df.to_csv(file, index_col=index_col)


def load_csv(file, index_col=0):
    return pd.read_csv(file, index_col=index_col)


def random_obs(x, y, p=0.2, s=1.0, include_min=False):
    n = int(len(x)*s)
    obs_j = np.unique(np.sort(np.random.choice(range(n), int(n*p), replace=False)))
    if include_min:
        id_min = y.argmin()
        if id_min not in obs_j:
            obs_j[np.random.choice(range(int(n*p)), 1)] = id_min
            obs_j = np.sort(obs_j)
    test_j = np.array([v for v in np.arange(len(x)) if v not in obs_j])
    x_obs = x[obs_j]
    y_obs = y[obs_j]
    x_test = x[test_j]
    y_test = y[test_j]
    print('Total: '+str(len(x)) +' | '+'Obs: '+str(len(obs_j)) + ' ('+str(100*len(obs_j)/len(x))+'%)')
    return obs_j, x_obs, y_obs, test_j, x_test, y_test


def uniform_obs(x, y, p=0.2, s=1.0):
    n = int(len(x) * s)
    obs_j = np.arange(0, n, int(1/p)).astype(np.int)
    test_j = np.array([v for v in np.arange(len(x)) if v not in obs_j])
    x_obs = x[obs_j]
    y_obs = y[obs_j]
    x_test = x[test_j]
    y_test = y[test_j]
    print('Total: '+str(len(x)) +' | '+'Obs: '+str(len(obs_j)) + ' ('+str(100*len(obs_j)/len(x))+'%)')
    return obs_j, x_obs, y_obs, test_j, x_test, y_test
