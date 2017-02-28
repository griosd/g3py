import os, sys, time
import pandas as pd
import pymc3 as pm
import multiprocessing as mp
from datetime import datetime as dt


from ..libs import random_obs, uniform_obs, likelihood_datatrace, save_pkl, load_pkl


class Experiment:

    def __init__(self, models, file=None):
        if file is not None:
            try:
                load = load_pkl(file)
                self.__dict__.update(load.__dict__)
                return
            except:
                print('Not found experiment in '+str(file))
        self.models = models
        self.file = file
        self.results = Results()

        self.data_x = None
        self.data_y = None
        self.data_p = None
        self.data_limit = 1
        self.data_min = True
        self.data_method = random_obs
        self.data_simulation = dict()

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

        self.save()

    def save(self, file=None):
        if file is not None:
            self.file = file
        if self.file is not None:
            save_pkl(self, self.file)

    def save_results(self):
        if self.file is not None:
            save_pkl(self, self.file+'.r')

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
        return obs_j, x_obs, y_obs, test_j, x_test, y_test

    def scores(self, logp=True, mean=True, median=False, variance=False):
        self.scores_mean = mean
        self.scores_median = median
        self.scores_variance = variance
        self.scores_logp = logp

    def calc_scores(self, sp, params):
        return sp.scores(params, logp=self.scores_logp, bias=self.scores_mean, median=self.scores_median,
                         variance=self.scores_variance)

    def model_selection(self, find_MAP=True, points=2, powell=True, starts='default', master=None):
        self.find_MAP = find_MAP
        self.points = points
        self.powell = powell
        self.starts = starts
        self.master = master

    def select_model(self, sp):
        if self.find_MAP:
            if self.starts is 'default':
                start = [sp.get_params_default(),
                         sp.get_params_random(mean=sp.get_params_current(), sigma=0.1),
                         sp.get_params_random(mean=sp.get_params_default(), sigma=0.2),
                         sp.get_params_random(mean=sp.get_params_current(), sigma=0.3),
                         sp.get_params_random(mean=sp.get_params_default(), sigma=0.4)]
            else:
                start = sp.get_params_default()
            if self.master:
                start = [sp.get_params_process(self.master, params=self.master.get_params_current())] + start
            params = sp.find_MAP(start=start, points=self.points, powell=self.powell)
        else:
            params = sp.get_params_default()
        return params

    def run(self, n_simulations=1):
        total_sims = len(self.data_simulation)
        for n_sim in range(total_sims, total_sims + n_simulations):
            obs_j, x_obs, y_obs, test_j, x_test, y_test = self.new_data()
            self.data_simulation[n_sim] = (obs_j, test_j)
            for sp in self.models:
                sp.observed(x_obs, y_obs)
                tictoc = time.time()
                params = self.select_model(sp)
                time_params, tictoc = time.time() - tictoc, time.time()

                sp.set_space(x_obs, y_obs)
                scores_obs = self.calc_scores(sp, params)
                time_scores_obs, tictoc = time.time() - tictoc, time.time()

                sp.set_space(x_test, y_test)
                scores_test = self.calc_scores(sp, params)
                time_scores_test, tictoc = time.time() - tictoc, time.time()

                self.results.add(n_sim, sp, params, scores_obs, scores_test, time_params, time_scores_obs, time_scores_test)
                self.save_results()

    def describe(self):
        return self.__dict__


class Results:
    def __init__(self):
        self.n_sim = list()
        self.model = list()
        self.params = list()
        self.obs = list()
        self.test = list()
        self.time_params = list()
        self.time_obs = list()
        self.time_test = list()

    def add(self, n_sim, model, params, obs, test, time_params, time_obs, time_test):
        self.n_sim.append(n_sim)
        self.model.append(model)
        self.params.append(params)
        self.obs.append(obs)
        self.test.append(test)
        self.time_params.append(time_params)
        self.time_obs.append(time_obs)
        self.time_test.append(time_test)

    def __call__(self, model=None):

        df = pd.DataFrame(columns=['n_sim', 'model', 'params', 'time_params', 'time_obs', 'time_test']
                                  + ['obs'+k for k in self.obs[0].keys()]+['test'+k for k in self.test[0].keys()])
        for i in range(len(self.n_sim)):
            sim = {'n_sim': self.n_sim[i], 'model': self.model[i].name, 'params': (self.params[i]),
                   'time_params': self.time_params[i], 'time_obs': self.time_obs[i], 'time_test': self.time_test[i]}
            sim.update({'obs'+k: v for k, v in self.obs[i].items()})
            sim.update({'test'+k: v for k, v in self.test[i].items()})
            df = df.append(sim, ignore_index=True)
        if model is not None:
            if type(model) is not list:
                model = [model]
            df = df[[m in model for m in df.model]]
        return df


def likelihood_datatrace_mp(sp, traces, index):
    strace_mp = traces._straces[index]
    strace_mp.chain = 0
    trace_mp = pm.backends.base.MultiTrace([strace_mp])
    datatraces = pm.trace_to_dataframe(trace_mp, hide_transformed_vars=False)
    likelihood_datatrace(sp, datatraces, trace_mp)
    return datatraces


def training():
    if len(sys.argv) != 5:
        print("Usage: <modelo.pkl> <n_samples> <chains (n_jobs)> <output path>")
        exit()

    model_pkl = sys.argv[1]
    n_samplings = int(sys.argv[2])
    n_chains = int(sys.argv[3])
    output_path = sys.argv[4]

    model_name = (model_pkl.split("/"))[-1]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + "/" + model_name):
        os.mkdir(output_path + "/" + model_name)

    output_dir = output_path + "/" + model_name + "/" + dt.now().strftime("%Y_%m_%d_%H_%M_%S") + "-" + str(
        randint(0, 1000))

    ngmodel = ngm.NGModel.load(model_pkl)

    print("Init Training: {}x{}".format(n_chains, n_samplings))
    traces = ngmodel.training(chain_number=n_chains, chain_length=n_samplings, path=output_dir)



    print("Calc Datatrace")
    ngmodel.predictor.time_init = 2
    if __name__ == '__main__':
        p = mp.Pool(n_chains)
        p.map(calc_datatrace_mp, list(range(n_chains)))
    print("End Training")


def training_auto():
    if len(sys.argv) != 5:
        print("Usage: <modelo.pkl> <n_samples> <chains (n_jobs)> <output path>")
        exit()

    model_pkl = sys.argv[1]
    n_samplings = int(sys.argv[2])
    n_chains = int(sys.argv[3])
    output_path = sys.argv[4]

    model_name = (model_pkl.split("/"))[-1]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path + "/" + model_name):
        os.mkdir(output_path + "/" + model_name)

    output_dir = output_path + "/" + model_name + "/" + dt.now().strftime("%Y_%m_%d_%H_%M_%S") + "-" + str(
        randint(0, 1000))

    ngmodel = ngm.NGModel.load(model_pkl)

    print("Test Training Convergence")
    step_size = max(n_samplings // 10, 1)
    max_iters = n_samplings // step_size
    traces = ngmodel.training(chain_number=n_chains, chain_length=1)
    for i in range(max_iters):
        print("Iter: {}".format(i))
        traces = ngmodel.training(chain_number=n_chains, chain_length=step_size, trace=traces)
        print("Length: {}".format(len(traces)))
        rhat = tr.max_rhat(traces, init=len(traces) // 2)
        print("R_Hat: {}".format(rhat))
        if rhat < 1.05:
            print("Chains Converged in sample {}".format((i + 1) * step_size))
            break

    print("Init Training: {}x{}".format(n_chains, n_samplings))
    traces = ngmodel.training(chain_number=n_chains, chain_length=n_samplings, trace=traces)
    print("Save Traces")
    pm.backends.text.dump(output_dir, traces)  # tr.dump_trace(output_dir, traces)

    print("Calc Datatrace")
    ngmodel.predictor.time_init = 2
    if __name__ == '__main__':
        p = mp.Pool(n_chains)
        p.map(calc_datatrace_mp, list(range(n_chains)))
    print("End Training")


def simple_sampling(path_data, path_model, data_name, time_end, dir_trace,
                        dir_datatrace, samples):
    dir_random = dt.now().strftime("%Y_%m_%d_%H_%M_%S") + "-" + str(randint(0, 1000))
    path_trace = dir_trace + data_name + '_' + str(time_end) + '/' + dir_random
    path_datatrace = dir_datatrace + data_name + '_' + str(time_end) + '/' + dir_random + '.pkl'

    os.makedirs(path_trace, exist_ok=True)
    os.makedirs(path_datatrace[:path_datatrace.rfind('/')], exist_ok=True)
    os.makedirs(path_model[:path_model.rfind('/')], exist_ok=True)


    df = pd.read_excel(path_data)
    df['Time'] = df.index
    data = df.filter(items=['Mes', 'Time', 'TRPS_CCR', 'TRPS_CCL', 'TRPS_CCZ', 'TRPS_CCLF'])

    Y = df[data_name]

    Y_dummy = data['TRPS_CCR']

    Y_dummy2 = data['TRPS_CCZ']

    Y_dummy3 = data['TRPS_CCL']

    Y_dummy4 = data['TRPS_CCLF']


    X = pd.DataFrame()
    X['id'] = 0
    X['Time'] = data['Time']
    X['Mes'] = data['Mes']
    X['id'] = 0

    X_dummy = X.copy()
    X_dummy['id'] = 1

    X_dummy2 = X.copy()
    X_dummy2['id'] = 2

    X_dummy3 = X.copy()
    X_dummy3['id'] = 3

    X_dummy4 = X.copy()
    X_dummy4['id'] = 4



    Y = np.concatenate([Y[:time_end],Y_dummy[:time_end],Y_dummy2[:time_end],Y_dummy3[:time_end],Y_dummy4[:time_end]])

    X = pd.concat([X[:time_end],X_dummy[:time_end],X_dummy2[:time_end],X_dummy3[:time_end],X_dummy4[:time_end]],ignore_index=True)


    try:
        gp = load_model(path_model+'.g3')
    except:
        path_model = path_model

    gp.hidden = Y
    gp.set_space(X)
    gp.observed(X, Y)
    gp.get_params_default()

    start = [gp.get_params_default(),
             gp.get_params_random(mean=gp.get_params_default(), sigma=0.1),
             gp.get_params_random(mean=gp.get_params_default(), sigma=0.2)]
    params, points_list = gp.find_MAP(start=start, points=3, powell=False, plot=False, return_points=True)
    with gp.model:
        traces = gp.sample_hypers(start=params, samples=samples, method='Slice', trace=pm.backends.Text(path_trace))
        # pm.backends.text.dump(path_trace, traces)
        datatraces = trace_to_datatrace(gp.model, traces)
        save_datatrace(datatraces, path_datatrace)
