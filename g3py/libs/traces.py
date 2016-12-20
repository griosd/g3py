import os

import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sb
import matplotlib.pyplot as plt


def marginal(datatrace, items=None, like=None, regex=None, samples=None):
    if items is None and like is None and regex is None:
        df = datatrace
    else:
        df = datatrace.filter(items=items, like=like, regex=regex)
    if samples is None or samples > len(datatrace):
        return df
    else:
        return df.sample(samples)


def conditional(datatrace, lambda_df):
    conditional_traces = datatrace.loc[lambda_df, :]
    print('#'+str(len(conditional_traces)) + " (" + str(100 * len(conditional_traces) / len(datatrace)) + " %)")
    return conditional_traces


def datatrace(model, trace):
    dt = trace_to_dataframe(trace, hide_transformed_vars=True)
    add_likelihood_to_dataframe(model, dt, trace)
    return dt


def trace_to_dataframe(trace, chains=None, flat_names=None, hide_transformed_vars=True):
    # TODO: mientras pymc3 se arregla
    var_shapes = trace._straces[0].var_shapes
    if flat_names is None:
        flat_names = {v: pm.backends.tracetab.create_flat_names(v, shape)
                      for v, shape in var_shapes.items()
                      if not (hide_transformed_vars and v.endswith('_'))}

    var_dfs = []
    for varname, shape in var_shapes.items():
        if not hide_transformed_vars or not varname.endswith('_'):
            vals = trace.get_values(varname, combine=True, chains=chains)
            flat_vals = vals.reshape(vals.shape[0], -1)
            var_dfs.append(pd.DataFrame(flat_vals, columns=flat_names[varname]))
    return pd.concat(var_dfs, axis=1)


def add_likelihood_to_dataframe(model, datatrace, trace):
    ll = pd.Series(index=datatrace.index)
    adll = pd.Series(index=datatrace.index)
    niter = pd.Series(index=datatrace.index)

    flogp = model.logp
    dflogp = model.dlogp()
    for i in range(len(trace)):
        niter[i] = i
        ll[i] = flogp(trace[i])
        adll[i] = np.sum(np.abs(dflogp((trace[i]))))
    datatrace['niter'] = niter
    datatrace['ll'] = ll
    datatrace['adll'] = adll


def find_candidates(datatrace, ll=1, adll=1, rand=1):
    # modes
    candidates = list()
    if 'll' in datatrace:
        for index, row in datatrace.nlargest(ll, 'll').iterrows():
            row.name = "ll[" + str(row.name) + "]"
            candidates.append(row)
    if 'adll' in datatrace:
        for index, row in datatrace.nsmallest(adll, 'adll').iterrows():
            row.name = "adll[" + str(row.name) + "]"
            candidates.append(row)
    mean = datatrace.mean()
    mean.name = 'mean'
    candidates.append(mean)
    median = datatrace.median()
    median.name = 'median'
    candidates.append(median)
    return pd.DataFrame(candidates).append(datatrace.sample(rand))


def dump_trace(name, trace, chains=None):
    # TODO: mientras pymc3 se arregla
    if not os.path.exists(name):
        os.mkdir(name)
    if chains is None:
        chains = trace.chains

    var_shapes = trace._straces[chains[0]].var_shapes
    flat_names = {v: pm.backends.tracetab.create_flat_names(v, shape)
                  for v, shape in var_shapes.items()}

    for chain in chains:
        filename = os.path.join(name, 'chain-{}.csv'.format(chain))
        df = trace_to_dataframe(trace, chains=chain, flat_names=flat_names, hide_transformed_vars=False)
        df.to_csv(filename, index=False)


def load_traces(dir_traces, last_samples=None):
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
    base_mtrace = mtraces[0]
    i = base_mtrace.nchains
    for new_mtrace in mtraces[1:]:
        for new_chain, strace in new_mtrace._straces.items():
            base_mtrace._straces[i] = strace
            base_mtrace._straces[i].chain = i
            i += 1
    return base_mtrace


def traceplot(trace, plot_transformed=True):
    pm.traceplot(trace, plot_transformed=plot_transformed)


def plot_datatrace(df, items=None, size=6, n_levels=20, cmap="Blues_d"):
    df = marginal(df, items)
    g = sb.PairGrid(df, size=size)
    g.map_diag(sb.distplot, bins=200)
    g.map_offdiag(plt.scatter)
    g.map_offdiag(sb.kdeplot, n_levels=n_levels, cmap=cmap)
    return g


def scatter_trace(datatrace, items=None, like=None, regex=None, samples=2000, bins=200, figsize=(15, 10)):
    pd.scatter_matrix(marginal(datatrace, items=items, like=like, regex=regex, samples=min(samples, len(datatrace))),
                      grid=True, hist_kwds={'bins': bins}, figsize=figsize)

