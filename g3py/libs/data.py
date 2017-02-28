import g3py as g3
import numpy as np
import pandas as pd
import statsmodels.api as sm


def data_sunspots():
    data = sm.datasets.sunspots.load_pandas().data['SUNACTIVITY']
    print(sm.datasets.sunspots.NOTE)
    x = data.index.values[:]
    y = data.values[:]
    return x, y


def data_co2():
    data = sm.datasets.co2.load_pandas().data
    print(sm.datasets.co2.NOTE)
    x = data.index.values[:]
    y = data.values[:]
    return x, y


def data_engel():
    data = sm.datasets.engel.load_pandas().data
    print(sm.datasets.engel.NOTE)
    return data


def data_heart():
    hr = pd.read_csv(g3.__path__[0] + '/data/hr2.txt', names=['hr'], dtype=np.float32)
    return hr.index.values, hr.values


def data_eurusd():
    hr = pd.read_csv(g3.__path__[0] + '/data/EURUSD-1401-1510.txt', names=['EURUSD'], dtype=np.float32)
    return hr.index.values, hr.values


def data_abalone(raw=False):
    names = ['Sex', 'Length', 'Diam', 'Height', 'Whole', 'Shucked', 'Viscera', 'Shell', 'Rings']
    print('abalone')
    abalone = pd.read_csv(g3.__path__[0] + '/data/abalone.data', names=names)
    if not raw:
        abalone['Sex'] = (abalone['Sex'] == 'M') * -1.0 + (abalone['Sex'] == 'F') * 1.0 + 0.0
        abalone = abalone.drop('Rings', axis=1)
    return abalone
    x = abalone.drop('Rings', axis=1).values.astype(dtype=np.float32)
    y = abalone['Rings'].values.astype(dtype=np.float32)
    return x, y


def data_creep(raw=False):
    names = ['Lifetime', 'Rupture_stress', 'Temperature', 'Carbon', 'Silicon', 'Manganese', \
             'Phosphorus', 'Sulphur', 'Chromium', 'Molybdenum', 'Tungsten', 'Nickel', 'Copper', \
             'Vanadium', 'Niobium', 'Nitrogen', 'Aluminium', 'Boron', 'Cobalt', 'Tantalum', 'Oxygen', \
             'Normalising_temperature', 'Normalising_time', 'Cooling_rate', 'Tempering_temperature', \
             'Tempering_time', 'Cooling_rate_tempering', 'Annealing_temperature', 'Annealing_time', \
             'Cooling_rate_annealing', 'Rhenium']
    print('creep')
    creep = pd.read_table(g3.__path__[0] + '/data/creep', names=names).astype('float32')
    if not raw:
        creep = creep.drop(['Rupture_stress', 'Tantalum', 'Cooling_rate_annealing', 'Rhenium'], axis=1)
    return creep


def data_ailerons(raw=False):
    names = ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', \
             'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', \
             'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', \
             'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', \
             'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', \
             'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal']
    print('ailerons')
    ailerons = pd.concat([pd.read_csv(g3.__path__[0] + '/data/ailerons.data', names=names),
                          pd.read_csv(g3.__path__[0] + '/data/ailerons.test', names=names)]).astype('float32')
    if not raw:
        ailerons['goal'] = ailerons['goal']*1000
        ailerons = ailerons.drop(['goal', 'diffSeTime2', 'diffSeTime4', 'diffSeTime6', 'diffSeTime8', 'diffSeTime10',
                                  'diffSeTime12', 'diffSeTime14'], axis=1)

    return ailerons


def data_rivers():
    r1 = np.exp(pd.read_csv(g3.__path__[0] + '/data/logbmau.csv', names=['bmau'], dtype=np.float32, skiprows=1))
    r2 = np.exp(pd.read_csv(g3.__path__[0] + '/data/logbmis.csv', names=['bmis'], dtype=np.float32, skiprows=1))
    r3 = np.exp(pd.read_csv(g3.__path__[0] + '/data/logcip.csv', names=['cip'], dtype=np.float32, skiprows=1))
    r4 = np.exp(pd.read_csv(g3.__path__[0] + '/data/logcol.csv', names=['col'], dtype=np.float32, skiprows=1))
    r5 = np.exp(pd.read_csv(g3.__path__[0] + '/data/logmau.csv', names=['mau'], dtype=np.float32, skiprows=1))
    return pd.concat([r1, r2, r3, r4, r5], axis=1)




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
