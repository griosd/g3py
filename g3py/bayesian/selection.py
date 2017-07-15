import time
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from tqdm import tqdm
from inspect import signature
from ..libs import random_obs, uniform_obs, save_pkl, load_pkl, nan_to_high, MaxTime
from .average import marginal_datatrace


def optimize(logp, start, dlogp=None, fmin=None, max_time=None, *args, **kwargs):
    if fmin in [None, 'bfgs', 'BFGS']:
        fmin = sp.optimize.fmin_bfgs
    else:
        fmin = sp.optimize.fmin_powell
    if max_time is None:
        callback = None
    else:
        callback = MaxTime(max_time)
    if ('fprime' in signature(fmin).parameters) and (dlogp is not None):
        r = fmin(lambda x: nan_to_high(-logp(x)), start, #process.model.bijection.map(start)[process.sampling_dims],
                 fprime=lambda x: np.nan_to_num(-dlogp(x)), callback=callback, *args, **kwargs)
    else:
        r = fmin(lambda x: nan_to_high(-logp(x)), start, #process.model.bijection.map(start)[process.sampling_dims],
                 callback=callback, *args, **kwargs)
    return r


#TODO: Probar
class CrossValidation:
    def __init__(self, models=None, file=None, load=True):
        self.file = file
        if self.file is not None and load:
            try:
                exp = self.load()
                self.__dict__.update(exp.__dict__)
                self.load_simulations()
                self.load_results()
                if models is not None:
                    self.models = models
                return
            except:
                pass
        self.models = models

        self.data_x = None
        self.data_y = None
        self.data_p = None
        self.data_limit = 1
        self.data_min = True
        self.data_method = random_obs

        self.scores_mean = True
        self.scores_median = True
        self.scores_variance = True
        self.scores_logp = True
        self.scores_time = True

        self.find_MAP = None
        self.starts = None
        self.master = None
        self.points = None
        self.powell = None
        self.max_time = None
        self.holdout = None
        self.holdout_p = 0

        try:
            self.simulations_raw = self.load_simulations()
            self.results_raw = self.load_results()
        except:
            self.simulations_raw = pd.DataFrame(columns=['obs', 'valid', 'test', 'datetime'], index=None)
            self.results_raw = pd.DataFrame(columns=['n_sim', 'model', 'selected', 'start', 'params', 'scores_obs', 'scores_valid', 'scores_test',
                                                     'time_params', 'time_obs', 'time_valid', 'time_test', 'datetime'])

    def save(self, file=None):
        if file is not None:
            self.file = file
        if self.file is not None:
            save_pkl(self, self.file)

    def save_simulations(self):
        if self.file is not None:
            save_pkl(self.simulations_raw, self.file + '.s')

    def save_results(self):
        if self.file is not None:
            save_pkl(self.results_raw, self.file + '.r')

    def load(self):
        return load_pkl(self.file)

    def load_simulations(self):
        self.simulations_raw = load_pkl(self.file + '.s')
        return self.simulations_raw

    def load_results(self):
        self.results_raw = load_pkl(self.file + '.r')
        return self.results_raw

    def add_simulation(self, index, obs, valid, test):
        self.simulations_raw.loc[index] = {'obs': obs, 'valid': valid, 'test': test, 'datetime': str(dt.now())}
        self.save_simulations()

    def add_result(self, n_sim, model, selected, start, params, scores_obs, scores_valid, scores_test, time_params,
                   time_scores_obs, time_scores_valid, time_scores_test):
        self.results_raw.loc[len(self.results_raw)] = {'n_sim': n_sim, 'model': model, 'selected': selected,
                                                       'start': start, 'params': params, 'scores_obs': scores_obs,
                                                       'scores_valid': scores_valid, 'scores_test': scores_test,
                                                       'time_params': time_params, 'time_obs': time_scores_obs,
                                                       'time_valid': time_scores_valid, 'time_test': time_scores_test,
                                                       'datetime': str(dt.now())}
        self.save_results()

    def data(self, x, y, p, limit=1.0, method='random', include_min=False):
        self.data_x = x
        self.data_y = y
        self.data_p = p
        self.data_limit = limit
        if method is 'random':
            self.data_method = random_obs
        elif method is 'uniform':
            self.data_method = uniform_obs
        self.data_min = include_min

    def new_data(self):
        obs_j, x_obs, y_obs, test_j, x_test, y_test = self.data_method(x=self.data_x, y=self.data_y, p=self.data_p,
                                                                       s=self.data_limit, include_min=self.data_min)
        if self.holdout_p > 0:
            valid_j, x_valid, y_valid, sub_obs_j, sub_x_obs, sub_y_obs = self.data_method(x=obs_j, y=obs_j, p=self.holdout_p, include_min=self.data_min)
            obs_j, valid_j = obs_j[sub_obs_j], obs_j[valid_j]
            x_obs, y_obs = self.data_x[obs_j], self.data_y[obs_j]
            x_valid, y_valid = self.data_x[valid_j], self.data_y[valid_j]
        else:
            valid_j, x_valid, y_valid = None, None, None
        return obs_j, x_obs, y_obs, valid_j, x_valid, y_valid, test_j, x_test, y_test

    def scores(self, logp=True, mean=True, median=False, variance=False):
        self.scores_mean = mean
        self.scores_median = median
        self.scores_variance = variance
        self.scores_logp = logp

    def calc_scores(self, sp, params):
        return sp.scores(params, logp=self.scores_logp, bias=self.scores_mean, median=self.scores_median,
                         variance=self.scores_variance)

    def model_selection(self, find_MAP=True, points=2, powell=True, starts='default', master=None, holdout=None, holdout_p=0, max_time=None):
        self.find_MAP = find_MAP
        self.points = points
        self.powell = powell
        self.starts = starts
        self.master = master
        self.holdout = holdout
        self.holdout_p = holdout_p
        self.max_time = max_time

    def select_model(self, sp, x_valid=None, y_valid=None):
        if self.find_MAP:
            if self.starts is 'default':
                start = [sp.get_params_test(),
                         sp.get_params_default(),
                         sp.get_params_random(mean=sp.get_params_test(), sigma=0.1, prop=False),
                         sp.get_params_random(mean=sp.get_params_default(), sigma=0.1, prop=True),
                         sp.get_params_random(mean=sp.get_params_test(), sigma=0.2, prop=False),
                         sp.get_params_random(mean=sp.get_params_default(), sigma=0.2, prop=True)]
            elif self.starts is 'mcmc':
                lnp, chain = sp.ensemble_hypers(start=sp.get_params_default(), samples=25)
                step = -1
                start = list()
                for k in range(min(chain.shape[0], self.points)):
                    start.append(sp.model.bijection.rmap(chain[k, step, :]))
            else:
                start = sp.get_params_default()
            if self.master is not None and self.master != sp:
                start = [sp.get_params_process(self.master, params=self.master.get_params_test()),
                         sp.get_params_process(self.master, params=self.master.get_params_default())] + start

            params, points_list = sp.find_MAP(start=start, points=self.points, powell=self.powell, max_time=self.max_time, return_points=True)
            selected = 'find_MAP'
            if (self.holdout_p > 0) and (x_valid is not None) and (y_valid is not None):
                sp.set_space(x_valid, y_valid)
                params_scores = self.calc_scores(sp, params)
                params_scores['_nll'] = -sp.logp_dict(params).max()
                for (n, l, p) in points_list:
                    try:
                        scores = self.calc_scores(sp, p)
                        scores['_nll'] = l.min()
                        print(n, l, scores[self.holdout])
                        if scores[self.holdout] < params_scores[self.holdout]:
                            selected = n
                            params_scores = scores
                            params = p
                    except:
                        pass
                print('Selected: '+selected, params_scores)
        else:
            params = sp.get_params_default()
        return selected, start, params

    def run(self, n_simulations=1, repeat=[], plot=False):
        total_sims = len(self.simulations_raw)
        if type(repeat) is int:
            repeat = list(range(repeat))
        print('Simulations: n=', n_simulations, ' repeat=', repeat)
        for n_sim in tqdm(list(range(total_sims, total_sims+n_simulations)) + repeat, total=n_simulations+len(repeat)):
            if n_sim not in self.simulations_raw.index:
                print('\n' * 2 + '*' * 70+'\n' + '*' * 70 + '\nSimulation #'+str(n_sim) )
                obs_j, x_obs, y_obs, valid_j, x_valid, y_valid, test_j, x_test, y_test = self.new_data()
                self.add_simulation(n_sim, obs_j, valid_j, test_j)
                print('*' * 70)
            else:
                obs_j, valid_j, test_j = self.simulations_raw.loc[n_sim]['obs'], self.simulations_raw.loc[n_sim]['valid'], self.simulations_raw.loc[n_sim]['test']
                x_obs, y_obs, x_valid, y_valid, x_test, y_test = self.data_x[obs_j], self.data_y[obs_j], \
                                                                 self.data_x[valid_j], self.data_y[valid_j], \
                                                                 self.data_x[test_j], self.data_y[test_j]
                print('\n' * 2 + '*' * 60 + '\n' + '*' * 60 + '\nRepetition #' + str(n_sim) + '\n' + '*' * 60)

            for sp in tqdm(self.models, total = len(self.models)):
                print('\n'*2+'*'*50 + '\n' + sp.name+' #'+str(n_sim) + '\n'+'*'*50)
                sp.observed(x_obs, y_obs)
                if plot:
                    print('\n' + sp.name)

                tictoc = time.time()
                selected, start, params = self.select_model(sp, x_valid, y_valid)

                time_params, tictoc = time.time() - tictoc, time.time()
                if plot:
                    sp.plot(params)
                    sp.plot_model(params)
                sp.set_params(params)
                sp.set_space(x_obs, y_obs, obs_j)
                scores_obs = self.calc_scores(sp, params)
                time_scores_obs, tictoc = time.time() - tictoc, time.time()
                if valid_j is not None:
                    sp.set_space(x_valid, y_valid, valid_j)
                    scores_valid = self.calc_scores(sp, params)
                else:
                    scores_valid = {}
                time_scores_valid, tictoc = time.time() - tictoc, time.time()

                sp.observed(np.concatenate([x_obs, x_valid]), np.concatenate([y_obs, y_valid]))
                sp.set_space(x_test, y_test, test_j)
                scores_test = self.calc_scores(sp, params)
                time_scores_test, tictoc = time.time() - tictoc, time.time()
                if plot:
                    print(scores_test, time_params, time_scores_obs, time_scores_test)
                self.add_result(n_sim, sp.name, selected, start, params, scores_obs, scores_valid, scores_test, time_params,
                                time_scores_obs, time_scores_valid, time_scores_test)

    def describe(self):
        return {k: v for k, v in self.__dict__.items() if k not in ['results_raw', 'simulations_raw']}

    def results(self, model=None, scores_columns=True, params_columns=False, raw=False, like=None):
        if raw or self.results_raw.empty:
            return self.results_raw
        if model is not None:
            if type(model) is not list:
                model = [model]

        df = pd.DataFrame(columns=['n_sim', 'model', 'selected', 'time_params', 'time_obs', 'time_valid', 'time_test', 'datetime'])
        for i in range(len(self.results_raw)):
            row = self.results_raw.iloc[i]
            if model is not None and row.model not in model:
                continue
            sim = {'n_sim': row.n_sim, 'model': row.model, 'selected': row.selected, 'time_params': row.time_params, 'time_obs': row.time_obs,
                   'time_valid': row.time_valid, 'time_test': row.time_test, 'start': row.start, 'datetime': row.datetime}

            if scores_columns:
                sim.update({'obs'+k: v for k, v in row.scores_obs.items()})
                sim.update({'valid' + k: v for k, v in row.scores_valid.items()})
                sim.update({'test' + k: v for k, v in row.scores_test.items()})
            else:
                sim.update({'obs': row.scores_obs, 'valid': row.scores_valid, 'test': row.scores_test})

            if params_columns:
                sim.update({'p_' + k: v for k, v in row.params.items()})
            else:
                sim.update({'params': row.params})
            df = df.append(sim, ignore_index=True)
        return marginal_datatrace(df, like=like)

    def simulations(self):
        return self.simulations_raw

    def plot(self, score=None, data=None, bw=0.15, jitter=0.05):
        if score is None:
            return
        if data is None:
            data = self.results()
        sb.lvplot(y='model', x=score, data=data)
        sb.violinplot(y='model', x=score, data=data, inner='quartile', bw=bw)
        sb.swarmplot(y='model', x=score, data=data, color='w', alpha=0.5)
        sb.stripplot(y='model', x=score, data=data, color='w', alpha=0.5, jitter=jitter)
        sb.pointplot(y='model', x=score, data=data, color='k')
        plt.title(score)

    def drop_duplicates(self):
        self.results_raw = self.results_raw.drop_duplicates(subset=['n_sim', 'model'], keep='last').reset_index(drop=True)
        self.save_results()
