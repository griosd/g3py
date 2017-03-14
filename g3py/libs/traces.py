import os
import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn import cluster, mixture
from pymc3 import traceplot
from copy import copy


def save_pkl(to_pkl, path='file.pkl'):
    os.makedirs(path[:path.rfind('/')], exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(to_pkl, f, protocol=-1)


def load_pkl(path='file.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_datatrace(dt, path='datatrace.pkl'):
    os.makedirs(path[:path.rfind('/')], exist_ok=True)
    dt.to_pickle(path)


def load_datatrace(path='datatrace.pkl'):
    return pd.read_pickle(path)


def chains_to_datatrace(sp, chains, ll=None, transforms=True):
    columns = list()
    for v in sp.model.bijection.ordering.vmap:
        columns += pm.backends.tracetab.create_flat_names(v.var, v.shp)
    datatrace = pd.DataFrame()
    for nchain in range(len(chains)):
        pdchain = pd.DataFrame(chains[nchain, :, :], columns=columns)
        pdchain['_nchain'] = nchain
        pdchain['_niter'] = pdchain.index
        if ll is not None:
            pdchain['_ll'] = ll[nchain]
        datatrace = datatrace.append(pdchain, ignore_index=True)
    if transforms:
        ncolumn = 0
        varnames = sp.get_params_test().keys()
        for v in datatrace.columns:
            if '___' in v:
                name = v[:v.find('___')+1]
            else:
                name = v
            if name not in varnames:
                continue
            dist = sp.model[name].distribution
            if hasattr(dist, 'transform_used'):
                trans = dist.transform_used
                datatrace.insert(ncolumn, v.replace('_'+trans.name+'_', ''), trans.backward(datatrace[v]).eval())
                ncolumn += 1
    return datatrace


def datatraceplot(datatrace, varnames=None, transform=lambda x: x, figsize=None,
                  lines=None, combined=False, plot_transformed=True, grid=True,
                  alpha=0.35, priors=None, prior_alpha=1, prior_style='--',
                  ax=None):
    """Plot samples histograms and values

    Parameters
    ----------

    datatrace : result of MCMC run on DataFrame format
    varnames : list of variable names
        Variables to be plotted, if None all variable are plotted
    transform : callable
        Function to transform data (defaults to identity)
    figsize : figure size tuple
        If None, size is (12, num of variables * 2) inch
    lines : dict
        Dictionary of variable name / value  to be overplotted as vertical
        lines to the posteriors and horizontal lines on sample values
        e.g. mean of posteriors, true values of a simulation
    combined : bool
        Flag for combining multiple chains into a single chain. If False
        (default), chains will be plotted separately.
    plot_transformed : bool
        Flag for plotting automatically transformed variables in addition to
        original variables (defaults to False).
    grid : bool
        Flag for adding gridlines to histogram. Defaults to True.
    alpha : float
        Alpha value for plot line. Defaults to 0.35.
    priors : iterable of PyMC distributions
        PyMC prior distribution(s) to be plotted alongside poterior. Defaults
        to None (no prior plots).
    prior_alpha : float
        Alpha value for prior plot. Defaults to 1.
    prior_style : str
        Line style for prior plot. Defaults to '--' (dashed line).
    ax : axes
        Matplotlib axes. Accepts an array of axes, e.g.:

        >>> fig, axs = plt.subplots(3, 2) # 3 RVs
        >>> pymc3.traceplot(trace, ax=axs)

        Creates own axes by default.

    Returns
    -------

    ax : matplotlib axes

    """
    datatrace = datatrace.set_index(['_nchain'])
    if combined:
        datatrace.index = datatrace.index * 0
    else:
        datatrace = datatrace.drop(['_niter'], axis=1)
    if varnames is None:
        if plot_transformed:
            varnames = [name for name in datatrace.columns]
        else:
            varnames = [name for name in datatrace.columns if not name.endswith('_')]

    n = len(varnames)

    if figsize is None:
        figsize = (12, n * 2)

    if ax is None:
        fig, ax = plt.subplots(n, 2, squeeze=False, figsize=figsize)
    elif ax.shape != (n, 2):
        pm._log.warning('traceplot requires n*2 subplots')
        return None

    for i, v in enumerate(varnames):
        prior = None
        if priors is not None:
            prior = priors[i]

        for key in datatrace.index.unique():
            d = datatrace.loc[key][v]
            d = np.squeeze(transform(d))
            d = pm.plots.make_2d(d)
            if d.dtype.kind == 'i':
                pm.plots.histplot_op(ax[i, 0], d, alpha=alpha)
            else:
                pm.plots.kdeplot_op(ax[i, 0], d, prior, prior_alpha, prior_style)
            ax[i, 0].set_title(str(v))
            ax[i, 0].grid(grid)
            ax[i, 1].set_title(str(v))
            ax[i, 1].plot(d, alpha=alpha)

            ax[i, 0].set_ylabel("Frequency")
            ax[i, 1].set_ylabel("Sample value")

            if lines:
                try:
                    ax[i, 0].axvline(x=lines[v], color="r", lw=1.5)
                    ax[i, 1].axhline(y=lines[v], color="r",
                                     lw=1.5, alpha=alpha)
                except KeyError:
                    pass
            ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()


def save_traces(sp, traces, path):
    os.makedirs(path[:path.rfind('/')], exist_ok=True)
    with sp.model:
        pm.backends.text.dump(path, traces)


def load_traces(sp, path):
    with sp.model:
        return pm.backends.text.load(path)


def load_traces_dir(sp, dir_traces, last_samples=None):
    with sp.model:
        traces = []
        for subdir in [os.path.join(dir_traces, o) for o in os.listdir(dir_traces) if
                       os.path.isdir(os.path.join(dir_traces, o))]:
            try:
                if last_samples is None:
                    traces.append(pm.backends.text.load(subdir))
                else:
                    traces.append(pm.backends.text.load(subdir)[-int(last_samples):])
            except:
                pass
        return append_traces(traces)


def append_traces(mtraces):
    """Joins many MultiTrace objects into one.

    Args:
        mtraces (list): MultiTrace objects to join

    Returns:
        pm.backends.base.MultiTrace: MultiTrace object containing all the others joined

    """
    base_mtrace = copy(mtraces[0])
    i = base_mtrace.nchains
    for new_mtrace in mtraces[1:]:
        for new_chain, strace in new_mtrace._straces.items():
            base_mtrace._straces[i] = strace
            base_mtrace._straces[i].chain = i
            i += 1
    return base_mtrace


def trace_to_datatrace(sp, trace):
    dt = pm.trace_to_dataframe(trace, hide_transformed_vars=False)
    likelihood_datatrace(sp, dt, trace)
    return dt


def likelihood_datatrace(sp, datatrace, trace):
    ll = pd.Series(index=datatrace.index)
    adll = pd.Series(index=datatrace.index)
    niter = pd.Series(index=datatrace.index)

    #flogp = sp.model.logp
    #dflogp = sp.model.dlogp()

    # Pasar de diccionario a arreglo
    vars = pm.theanof.inputvars(sp.model.cont_vars)
    start = sp.model.test_point
    start = pm.model.Point(start, model=sp.model)
    OrdVars = pm.blocking.ArrayOrdering(vars)
    bij = pm.blocking.DictToArrayBijection(OrdVars, start)
    logp = bij.mapf(sp.model.fastlogp)
    dlogp = bij.mapf(sp.model.fastdlogp(vars))


    n_traces = len(trace._straces)
    for s in range(n_traces):
        lenght_trace = len(trace._straces[s])
        for i in range(lenght_trace):
            niter[s*lenght_trace + i] = i
            ll[s*lenght_trace + i] = logp(bij.map(trace._straces[s][i]))
            adll[s*lenght_trace + i] = np.sum(np.abs(dlogp(bij.map(trace._straces[s][i]))))

            #ll[s*lenght_trace + i] = flogp(trace._straces[s][i])
            #adll[s*lenght_trace + i] = np.sum(np.abs(dflogp((trace._straces[s][i]))))
    datatrace['_niter'] = niter
    datatrace['_ll'] = ll
    datatrace['_adll'] = adll


def cluster_datatrace(dt, n_components=10, n_init=1):
    excludes = '^((?!_nchain|_niter|_log_|_logodds_|_interval_|_lowerbound_|_upperbound_|_sumto1_|_stickbreaking_|_circular_).)*$'
    datatrace_filter = dt.filter(regex=excludes)
    gm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type='full', max_iter=1000, n_init=n_init).fit(datatrace_filter)
    cluster_gm = gm.predict(datatrace_filter)
    dt['_cluster'] = cluster_gm


def marginal(dt, items=None, like=None, regex=None, samples=None):
    if items is None and like is None and regex is None:
        df = dt
    else:
        df = dt.filter(items=items, like=like, regex=regex)
    if samples is None or samples > len(dt):
        return df
    else:
        return df.sample(samples)


def conditional(dt, lambda_df):
    conditional_traces = dt.loc[lambda_df, :]
    print('#' + str(len(conditional_traces)) + " (" + str(100 * len(conditional_traces) / len(dt)) + " %)")
    return conditional_traces


def find_candidates(dt, ll=1, adll=1, rand=1):
    # modes
    candidates = list()
    if '_ll' in dt:
        for index, row in dt.nlargest(ll, '_ll').iterrows():
            row.name = "ll[" + str(row.name) + "]"
            candidates.append(row)
    if '_adll' in dt:
        for index, row in dt.nsmallest(adll, '_adll').iterrows():
            row.name = "adll[" + str(row.name) + "]"
            candidates.append(row)
    mean = dt.mean()
    mean.name = 'mean'
    candidates.append(mean)
    median = dt.median()
    median.name = 'median'
    candidates.append(median)
    return pd.DataFrame(candidates).append(dt.sample(rand))


def hist_trace(datatrace, items=None, like=None, regex=None, samples=None, bins=200, layout=(4,4), figsize=(20,20)):
    marginal(datatrace, items=items, like=like, regex=regex, samples=samples).hist(bins=bins, layout=layout, figsize=figsize)


def scatter_datatrace(dt, items=None, like=None, regex=None, samples=None, bins=200, figsize=(15, 10), cluster=None, cmap=cm.rainbow):
    df = marginal(dt, items=items, like=like, regex=regex, samples=samples)
    if cluster is None:
        pd.scatter_matrix(df, grid=True, hist_kwds={'normed': True, 'bins': bins}, figsize=figsize)
    else:
        pd.scatter_matrix(df, grid=True, hist_kwds={'normed': True, 'bins': bins}, figsize=figsize, c=cluster[df.index],
                          cmap=cmap)


def kde_datatrace(dt, items=None, size=6, n_levels=20, cmap="Blues_d"):
    dt = marginal(dt, items)
    g = sb.PairGrid(dt, size=size)
    g.map_diag(sb.distplot, bins=200)
    g.map_offdiag(plt.scatter)
    g.map_offdiag(sb.kdeplot, n_levels=n_levels, cmap=cmap)
    return g