import os
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import pymc3 as pm
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import mixture, neighbors
from matplotlib import cm
from copy import copy
from numba import jit


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

@jit
def gelman_rubin(chains, method='multi-sum'):
    # This method return the abs(gelman_rubin-1), so near to 0 is best
    nwalkers, nsamples, ndim = chains.shape
    if nwalkers == 1:
        return 0
    if method in ['multi-sum', 'multi-max']:
        B = nsamples * np.cov(np.mean(chains, axis=1).T)
        W = B * 0
        for chain in range(nwalkers):
            W += np.cov(chains[chain, :, :].T)
        W /= nwalkers
        Vhat = W * (nsamples - 1) / nsamples + B / nsamples
        eigvalues = np.linalg.eigvals((1 / nsamples) * np.linalg.solve(W, Vhat))  # np.matmul(np.linalg.inv(W), Vhat))
        if method is 'multi-sum':
            return np.abs((nsamples - 1) / nsamples + ((nwalkers + 1) / nwalkers) * np.sum(eigvalues) - 1)
        else:
            return np.abs((nsamples - 1) / nsamples + ((nwalkers + 1) / nwalkers) * np.max(eigvalues) - 1)
    else:
        Rhat = np.zeros(ndim)
        for i in range(ndim):
            x = chains[:, :, i]
            # Calculate between-chain variance
            B = nsamples * np.var(np.mean(x, axis=1), axis=0, ddof=1)
            # Calculate within-chain variance
            W = np.mean(np.var(x, axis=1, ddof=1), axis=0)
            # Estimate of marginal posterior variance
            Vhat = W * (nsamples - 1) / nsamples + B / nsamples
            Rhat[i] = np.sqrt(Vhat / W)
        return np.max(np.abs(Rhat - 1))


def burn_in_samples(chains, tol=0.1, method='multi-sum'):
    try:
        score = gelman_rubin(chains, method)
    except:
        method = 'uni'
        try:
            score = gelman_rubin(chains, method)
        except:
            score = np.inf
    if score > tol:
        return chains.shape[1]
    lower = 0
    upper = chains.shape[1]
    burnin = upper
    while lower + 1 < upper:
        n = lower + (upper - lower) // 2
        if gelman_rubin(chains[:, :n, :], method) < tol:
            burnin = upper
            upper = n
        else:
            lower = n
    return burnin


def chains_to_datatrace(sp, chains, ll=None, transforms=True, burnin_tol=0.01, burnin_method='multi-sum', burnin_dims=None,
                        outlayer_percentile=0.001):
    columns = list()
    for v in sp.model.bijection.ordering.vmap:
        columns += pm.backends.tracetab.create_flat_names(v.var, v.shp)
    n_vars = len(columns)
    datatrace = pd.DataFrame()
    if len(chains.shape) == 2:
        chains = chains[None, :, :]
    if ll is not None and len(ll.shape) == 1:
        ll = ll[None, :]
    if burnin_tol is not None:
        if burnin_dims is None:
            chains_to_burnin = chains[:, :, sp.sampling_dims]
        else:
            chains_to_burnin = chains[:, :, burnin_dims]
        nburn = burn_in_samples(chains_to_burnin, tol=burnin_tol, method=burnin_method)
    for nchain in range(len(chains)):
        pdchain = pd.DataFrame(chains[nchain, :, :], columns=columns)
        pdchain['_nchain'] = nchain
        pdchain['_niter'] = pdchain.index
        if burnin_tol is not None:
            pdchain['_burnin'] = pdchain['_niter'] > nburn
        if ll is not None:
            pdchain['_ll'] = ll[nchain]
        datatrace = datatrace.append(pdchain, ignore_index=True)

    if outlayer_percentile is not None:
        if ll is not None:
            percentiles = datatrace[np.isfinite(datatrace['_ll'])].describe(percentiles=[outlayer_percentile, 1 - outlayer_percentile])
        else:
            percentiles = datatrace.describe(percentiles=[outlayer_percentile, 1 - outlayer_percentile])
        lower = percentiles.iloc[-4]
        upper = percentiles.iloc[-2]
        datatrace.insert(n_vars + 2 + (burnin_tol is not None), '_outlayer', ~(((datatrace.iloc[:, :sp.ndim] > upper.iloc[:sp.ndim]) |
                                         (datatrace.iloc[:, :sp.ndim] < lower.iloc[:sp.ndim])).any(axis=1) | (datatrace._ll > upper._ll) | (datatrace._ll < lower._ll)  ) )
        if ll is not None:
            datatrace['_outlayer'] &= np.isfinite(datatrace['_ll'])

    if transforms:
        ncolumn = n_vars
        varnames = sp.get_params_test().keys()
        for v in datatrace.columns:
            if '___' in v:
                name = v[:v.find('___')+1]
            else:
                name = v
            if name not in varnames:
                continue
            if name in sp.compiles_trans:
                datatrace.insert(ncolumn, v.replace('_'+sp.model[name].distribution.transform_used.name+'_', ''), sp.compiles_trans[name](datatrace[v]))
                ncolumn += 1
    return datatrace


def datatrace_to_chains(process, dt, flat=False, burnin=False):
    if burnin and hasattr(dt,'_burnin'):
        chain = dt[dt._burnin]
    else:
        chain = dt
    if flat:
        return chain.ix[:, :process.ndim].values
    else:
        levshape = chain.set_index([chain._nchain, chain._niter]).index.levshape
        return chain.ix[:, :process.ndim].values.reshape(levshape[0], levshape[1], process.ndim)


def datatrace_to_kde(process, dt, kernel='tophat', bandwidth=0.02, min_ll=-1e6):
    # con outlayers pero sin burn-in
    if hasattr(dt, '_ll'):
        dt = dt[np.isfinite(dt['_ll'])]
        dt = dt[dt._ll > min_ll]
    kde = neighbors.kde.KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(dt[dt._burnin].iloc[:, :process.ndim])
    kde.min_ll = dt[dt._burnin]._ll.min()
    return kde


def kde_to_datatrace(process, kde, nsamples=1000):
    samples = kde.sample(n_samples=1)
    ll = process.logp_chain(samples)
    samples, ll = samples[ll > kde.min_ll], ll[ll > kde.min_ll]

    while len(samples) < nsamples:
        new_samples = kde.sample(n_samples=nsamples-len(samples))
        new_ll = process.logp_chain(new_samples)
        new_samples, new_ll = new_samples[new_ll > kde.min_ll], new_ll[new_ll > kde.min_ll]
        samples = np.concatenate([samples, new_samples])
        ll = np.concatenate([ll, new_ll])
    kde_dt = chains_to_datatrace(process, samples, ll=ll)
    if hasattr(process, '_cluster'):
        process._cluster(kde_dt)
    kde_dt._burnin = True
    return kde_dt


def plot_datatrace(datatrace, burnin = False, outlayer = False, varnames=None, transform=lambda x: x, figsize=None,
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
    nburnin = (~datatrace._burnin).idxmin()
    if burnin and hasattr(datatrace, '_burnin'):
        datatrace = datatrace[datatrace._burnin]
    if outlayer and hasattr(datatrace, '_outlayer'):
        datatrace = datatrace[datatrace._outlayer]
    datatrace = datatrace.set_index(['_nchain']).drop(['_burnin', '_outlayer'], axis=1)
    if combined:
        datatrace.index = datatrace.index * 0
    #else:
    #    datatrace = datatrace.drop(['_niter'], axis=1)
    if varnames is None:
        if plot_transformed:
            varnames = [name for name in datatrace.columns if name != '_niter']
        else:
            varnames = [name for name in datatrace.columns if name != '_niter' and (not name.endswith('_'))]

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
            dk = datatrace.loc[key]
            dk = dk[np.isfinite(dk[v])]
            d = np.squeeze(transform(dk[v]))
            d = pm.plots.make_2d(d)
            try:
                if d.dtype.kind == 'i':
                    pm.plots.histplot_op(ax[i, 0], d, alpha=alpha)
                else:
                    pm.plots.kdeplot_op(ax[i, 0], d, prior, prior_alpha, prior_style)
            except:
                pass
            ax[i, 1].plot(dk._niter, d, alpha=alpha)
        ax[i, 1].axvline(x=nburnin, color="r", lw=1.5, alpha=alpha)
        if lines:
            try:
                ax[i, 0].axvline(x=lines[v], color="r", lw=1.5)
                ax[i, 1].axhline(y=lines[v], color="r",
                                 lw=1.5, alpha=alpha)
            except KeyError:
                pass
        #ax[i, 0].set_title(str(v))
        #ax[i, 1].set_title(str(v))
        name_var = str(v)
        ax[i, 0].set_ylabel(name_var[name_var.find('_')+1:])
        ax[i, 1].set_ylabel("Sample value")
        ax[i, 0].grid(grid)
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


def cluster_datatrace(process, dt, n_components=5, bayesian=True, burnin=True, n_init=1, max_iter=5000):
    #excludes = '^((?!_nchain|_niter|_burnin|_outlayer|_cluster|_log_|_logodds_|_interval_|_lowerbound_|_upperbound_|_sumto1_|_stickbreaking_|_circular_).)*$'
    #dt.filter(regex=excludes)
    if burnin:
        datatrace_filter = dt[dt._burnin].iloc[:, :process.ndim]
    else:
        datatrace_filter = dt.iloc[:, :process.ndim]
    if bayesian:
        cluster_method = mixture.BayesianGaussianMixture
    else:
        cluster_method = mixture.GaussianMixture
    gm = cluster_method(n_components=n_components, covariance_type='full', max_iter=max_iter, n_init=n_init).fit(datatrace_filter)
    cluster_gm = gm.predict(datatrace_filter)
    argsort = np.argsort(np.bincount(cluster_gm))
    argsorted = sorted(np.unique(cluster_gm), reverse=True)

    def _cluster(datatrace):
        datatrace['_cluster'] = (gm.predict(datatrace.iloc[:, :process.ndim]) == argsort[:, None]).T.dot(argsorted)
    _cluster(dt)
    process._cluster = _cluster
    return _cluster


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


def find_candidates(dt, ll=1, l1=0, l2=0, mean=False, median=False, cluster=False, rand=0):
    # modes
    dt_full = dt.drop_duplicates(subset=[k for k in dt.columns if not k.startswith('_')])
    candidates = list()
    for c in (dt_full._cluster.unique() if cluster else [0]):
        if cluster:
            dt = dt_full[dt_full._cluster == c]
        else:
            dt = dt_full
        if '_ll' in dt:
            for index, row in dt.nlargest(ll, '_ll').iterrows():
                row.name = "ll[" + str(row.name) + "]"
                candidates.append(row)
        if '_l1' in dt:
            for index, row in dt.nsmallest(l1, '_l1').iterrows():
                row.name = "l1[" + str(row.name) + "]"
                candidates.append(row)
        if '_l2' in dt:
            for index, row in dt.nsmallest(l2, '_l2').iterrows():
                row.name = "l2[" + str(row.name) + "]"
                candidates.append(row)
        if mean:
            m = dt.mean()
            m.name = 'mean'
            candidates.append(m)
        if median:
            m = dt.median()
            m.name = 'median'
            candidates.append(m)
    return pd.DataFrame(candidates).append(dt.sample(rand))


def hist_datatrace(dt, items=None, like=None, regex=None, samples=None, bins=200, layout=(5, 5), figsize=(20, 20), burnin=True, outlayer=True):
    if burnin and hasattr(dt, '_burnin'):
        dt = dt[dt._burnin]
    if outlayer and hasattr(dt, '_outlayer'):
        dt = dt[dt._outlayer]
    marginal(dt, items=items, like=like, regex=regex, samples=samples).hist(bins=bins, layout=layout, figsize=figsize)


def scatter_datatrace(dt, items=None, like=None, regex=None, samples=100000, bins=200, figsize=(15, 15), burnin=True, outlayer=True, cluster=None, cmap=cm.rainbow_r):
    if burnin and hasattr(dt, '_burnin'):
        dt = dt[dt._burnin]
    if outlayer and hasattr(dt, '_outlayer'):
        dt = dt[dt._outlayer]
    df = marginal(dt, items=items, like=like, regex=regex, samples=samples)
    if cluster is None and hasattr(dt, '_cluster'):
        cluster = dt._cluster
    if cluster is None:
        pd.scatter_matrix(df, grid=True, hist_kwds={'normed': True, 'bins': bins}, figsize=figsize)
    else:
        pd.scatter_matrix(df, grid=True, hist_kwds={'normed': True, 'bins': bins}, figsize=figsize, c=cluster[df.index],
                          cmap=cmap)
    #ax = plt.gca()
    #ax.spines['top'].set_visible(True)
    #ax.spines['bottom'].set_visible(True)
    #ax.spines['left'].set_visible(True)
    #ax.spines['right'].set_visible(True)


def kde_datatrace(dt, items=None, size=6, n_levels=20, cmap="Blues_d"):
    dt = marginal(dt, items)
    g = sb.PairGrid(dt, size=size)
    g.map_diag(sb.distplot, bins=200)
    g.map_offdiag(plt.scatter)
    g.map_offdiag(sb.kdeplot, n_levels=n_levels, cmap=cmap)
    return g


def effective_sample_min(process, alpha=0.05, error=0.05, p=None):
    """Calculate number of minimum effective samples needed for good enough estimates.

    Args:
        process (StochasticProcess): StochasticProcess with model
        alpha (float): Wanted error precision (probability of satisfying "real-error" < error)
        error (float): Wanted precision (max. Monte Carlo error)
        p (int): Number of variables in the model
    Returns:
        float: Number of samples needed

    .. _Multivariate Output Analysis for MCMC:
        https://arxiv.org/pdf/1512.07713v2.pdf

    """
    if p is None:
        p = process.ndim
    return np.pi*(2**(2/p))*(sp.stats.chi2.ppf(1-alpha, p)) / (((p*sp.special.gamma(p/2))**(2/p))*(error**2))


def effective_sample_size(process, dt, method='mIS', batch_size=None, fixed=True, flat=False, reshape=False, burnin=True):
    chains = datatrace_to_chains(process, dt, flat=flat, burnin=burnin)
    if fixed:
        if flat:
            chains = chains[:, process.sampling_dims]
        else:
            chains = chains[:, :, process.sampling_dims]
    #print(chains.shape)
    dim_sample = 1
    #flat samples
    if flat:
        chains = chains[None, :, :]
    elif reshape:
        nwalkers, nsamples, ndim = chains.shape
        chains = np.transpose(chains, axes=[1, 0, 2]).reshape(1, nsamples, nwalkers * ndim)
        dim_sample = nwalkers
    #print(chains.shape)
    #flat dimension
    nwalkers, nsamples, ndim = chains.shape
    chains_mESS = np.zeros(nwalkers)
    for nchain in range(nwalkers):
        chains_mESS[nchain] = _mESS(chains[nchain, :, :], method, batch_size)
    return np.floor(dim_sample*np.sum(chains_mESS))


def _mESS(chain, method='mIS', batch_size=None):
    nsamples, ndim = chain.shape
    cov_chain = np.cov(chain.T)
    det_cov = np.abs(np.linalg.det(cov_chain))
    if det_cov == 0:
        return 1
    if method == 'batch' or batch_size is not None:
        if batch_size is None:
            batch_size = 1
        sigma_cov = _sigma_batch(chain, batch_size)
    elif method == 'adjusted':
        sigma_cov = _sigma_mIS_adj(chain)
    else: #mIS
        sigma_cov = _sigma_mIS(chain)
    det_sigma = np.abs(np.linalg.det(sigma_cov))
    if det_sigma == 0:
        return 1
    #print(det_cov, det_sigma, det_cov/det_sigma, (det_cov/det_sigma)**(1/ndim), nsamples*(det_cov/det_sigma)**(1/ndim))
    return nsamples*(det_cov/det_sigma)**(1/ndim)


def _is_positive_definite(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


def _autocov_matrix(chain, lag):
    n = chain.shape[0]
    x = chain - np.mean(chain, axis=0)
    #print(x[:(n - lag), :].T.shape, x[lag:, :].shape)
    #return (1 / n) * np.matmul(x[:(n - lag), :].T, x[lag:, :])
    return (1/n)*(x[:(n-lag), :].T.dot(x[lag:, :]))


def _autocov_matrix_2(chain, i):
    return _autocov_matrix(chain, lag =2 * i) + _autocov_matrix(chain, lag =2 * i + 1)


def _sigma_batch(chain, batch_size):
    # Estimador consistente (bajo ciertas hipotesis) de matriz de covarianza para el CLT de Markov
    # https://arxiv.org/pdf/1512.07713v1.pdf
    nsamples, ndim = chain.shape
    a = np.floor(nsamples / batch_size).astype(np.int)
    mu = np.mean(chain)
    block_means = np.zeros((a, ndim))
    k = np.arange(a) * batch_size
    for i in range(batch_size):
        block_means += chain[k, :]
        k += 1
    block_means /= batch_size
    A = block_means - mu
    return (batch_size / (a - 1)) * np.matmul(A.T, A)


def _sigma_mIS(chain):
    # estimador mIS de sigma_cov de clt de Markov
    # http://users.stat.umn.edu/~galin/DaiJones.pdf
    n = chain.shape[0]
    k = np.floor(n / 2 - 1).astype(np.int)
    sn = 0
    sigma_cov = _autocov_matrix(chain, lag = 0) + 2 * _autocov_matrix(chain, lag = 1) #Sigma_m(Trace, m = sn)
    while sn < k and not _is_positive_definite(sigma_cov):
        #print(sn, k)
        sigma_cov += 2 * _autocov_matrix_2(chain, sn + 1)
        sn += 1
    sn -= 1 # sigma_cov = Sigma_{n,s_n}
    m = sn + 1
    sigma_cov_init = sigma_cov
    sigma_cov += 2 * _autocov_matrix_2(chain, sn + 1) # sigma_cov = Sigma_{n,m}
    while np.linalg.det(sigma_cov_init) < np.linalg.det(sigma_cov) and m < k: #checkear <=
        sigma_cov_init = sigma_cov
        sigma_cov += 2 * _autocov_matrix_2(chain, m + 1)
        m += 1
    return sigma_cov


def _sigma_mIS_adj(chain):
    # estimador mISadj de sigma_cov de clt de Markov
    # http://users.stat.umn.edu/~galin/DaiJones.pdf
    n = chain.shape[0]
    k = np.floor(n / 2 - 1).astype(np.int)
    sn = 0
    sigma_cov = _autocov_matrix(chain, lag = 0) + 2 * _autocov_matrix(chain, lag = 1) #Sigma_m(Trace, m = sn)
    while sn < k and not _is_positive_definite(sigma_cov):
        sigma_cov += 2 * _autocov_matrix_2(chain, sn + 1)
        sn += 1
    sn -= 1 # sigma_cov = Sigma_{n,s_n}
    m = sn + 1
    sigma_cov_adj = sigma_cov
    sigma_cov_init = sigma_cov
    sigma_cov += 2 * _autocov_matrix_2(chain, sn + 1) # sigma_cov = Sigma_{n,m}
    while np.linalg.det(sigma_cov_init) < np.linalg.det(sigma_cov) and m < k:
        sigma_cov_init = sigma_cov
        sigma_cov_update = 2 * _autocov_matrix_2(chain, m + 1)
        if not _is_positive_definite(sigma_cov_update):
            val, vec = np.linalg.eigh(sigma_cov_update)
            val_pos = np.diag(np.max(val, 0))
            sigma_cov_update_adj = vec.dot((np.linalg.solve(vec.T,val_pos.T)).T)
        else:
            sigma_cov_update_adj = sigma_cov_update
        sigma_cov += sigma_cov_update
        sigma_cov_adj += sigma_cov_update_adj
        m += 1
    return sigma_cov_adj
