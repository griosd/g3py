import os, sys
import pandas as pd
import pymc3 as pm
import multiprocessing as mp
from datetime import datetime as dt
from processes import StochasticProcess
from libs import show, random_obs, likelihood_datatrace
from time import time


class Experiment:

    def __init__(self, processes, x, y, p_obs, limit_obs, include_min, n_splits,
                   path_experiment, master = None):
        create_experiments(processes, path_experiment)

        for split in range(n_splits):
            obs_j, x_obs, y_obs, test_j, x_test, y_test = random_obs(x, y, p_obs, limit_obs, include_min=include_min)
            for sp in processes:
                init_time = time()
                params_gp, scores_gp = simulation(gp)
                time_simulation = time() - init_time
                scores_gp['_time'] = time_simulation
                print('GP' + str(i), time_simulation)
                gp_scores = gp_scores.append(scores_gp, ignore_index=True)
                gp_params += [params_gp]
                gp_scores.to_csv(gp_scores_file)
                save_pkl(gp_params, gp_params_file)
        return scores

    def data(self):
        obs_j, x_obs, y_obs, test_j, x_test, y_test = g3.random_obs(x, y, p_obs, include_min=True)

    def create_experiment(self, tgp, path_experiment):
        tgp.observed(x_obs, y_obs)
        tgp.set_space(x_obs, y_obs)
        try:
            tgp_params = load_pkl(tgp_params_file)
            tgp_scores = load_csv(tgp_scores_file)
            print('load tgp experiment: ' + tgp_scores_file)
        except:
            print('create tgp experiment: ' + tgp_scores_file)
            init_time = time()
            tgp_params = [tgp.get_params_default()]
            tgp_scores = pd.DataFrame(
                tgp.scores(params=tgp_params[0], space=x_test, hidden=y_test, variance=False, median=True), index=[0])
            tgp_params[0]['_time'] = time() - init_time

    def run(self, n_simulations, simulation=None):

        if simulation is None:
            simulation = self.simulation

        sp.observed(x_obs, y_obs)
        sp.set_space(x_obs[:2, :], y_obs[:2])
        print(params)
        sp.set_space(x_test, y_test)
        scores = sp.scores(params=params, variance=False, median=True)
        print(scores)
        params, scores

        for i in range(n_simulations):
            init_time = time.time()
            params_gp, scores_gp = simulation(gp)
            time_simulation = time.time() - init_time
            scores_gp['_time'] = time_simulation
            print('GP' + str(i), time_simulation)
            gp_scores = gp_scores.append(scores_gp, ignore_index=True)
            gp_params += [params_gp]
            gp_scores.to_csv(gp_scores_file)
            g3.save_pkl(gp_params, gp_params_file)

            # Initial point from GP
            # print(params_gp)
            # print(tgp.get_params_process(gp, params=params_gp))
            tgp.set_params(tgp.get_params_process(gp, params=params_gp))
            # print(tgp.get_params_current())
            init_time = time.time()
            params_tgp, scores_tgp = simulation(tgp)
            time_simulation = time.time() - init_time
            scores_tgp['_time'] = time_simulation
            print('TGP' + str(i), time_simulation)
            tgp_scores = tgp_scores.append(scores_tgp, ignore_index=True)
            tgp_params += [params_tgp]
            tgp_scores.to_csv(tgp_scores_file)
            g3.save_pkl(tgp_params, tgp_params_file)

    def simulation(self, sp: StochasticProcess, x_obs, y_obs, x_test, y_test, plot=False):
        sp.observed(x_obs,y_obs)
        sp.set_space(x_obs[:2,:],y_obs[:2])
        params = sp.find_MAP(start=[sp.get_params_current(),
                                    sp.get_params_default(),
                                    sp.get_params_random(mean=sp.get_params_current(), sigma=0.1),
                                    sp.get_params_random(mean=sp.get_params_default(), sigma=0.2),
                                    sp.get_params_random(mean=sp.get_params_current(), sigma=0.3),
                                    sp.get_params_random(mean=sp.get_params_default(), sigma=0.4)],
                             points=5, plot=False, powell=False)
        print(params)
        if plot:
            sp.set_space(x_obs,y_obs)
            sp.plot(params = params, mean=False, var=False)
            show()
            sp.plot_model(params = params, indexs=[len(x_obs)//2, len(y_obs)//2+2])
            show()
            sp.set_space(x, y)
            sp.plot(params = params, mean=False, var=False, quantiles=False, noise=False)
            show()
        sp.set_space(x_test,y_test)
        scores = sp.scores(params = params, variance=False, median=True)
        print(scores)
        return params, scores


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
