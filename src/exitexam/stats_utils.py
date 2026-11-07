"""Statistical helpers (numpy/scipy only, so the project runs on a bare Python install)."""
import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------------------------------ proportions
def wilson_ci(k, n, z=1.96):
    k = np.asarray(k, float); n = np.asarray(n, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = k / n
        den = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / den
        half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


def two_prop_test(k1, n1, k2, n2):
    """Difference p1-p2 with a Newcombe-style Wald CI and a two-sided z-test (pooled)."""
    p1, p2 = k1 / n1, k2 / n2
    diff = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pp = (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = diff / se0 if se0 > 0 else np.nan
    return dict(diff=diff, ci_low=diff - 1.96 * se, ci_high=diff + 1.96 * se, p=2 * stats.norm.sf(abs(z)))


def cramers_v(table):
    chi2, p, dof, _ = stats.chi2_contingency(table, correction=False)
    n = np.asarray(table).sum()
    return dict(chi2=chi2, dof=dof, p=p, cramers_v=np.sqrt(chi2 / (n * (min(np.shape(table)) - 1))))


def cliffs_delta(a, b):
    """P(a>b) - P(a<b), from the Mann-Whitney U statistic. Range -1..1."""
    u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
    return dict(cliffs_delta=2 * u / (len(a) * len(b)) - 1, mw_p=p)


def holm(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def benjamini_hochberg(p):
    p = np.asarray(p, float); m = len(p)
    order = np.argsort(p)[::-1]; adj = np.empty(m); prev = 1.0
    for rank, i in zip(range(m, 0, -1), order):
        prev = min(prev, p[i] * m / rank); adj[i] = prev
    return adj


def fisher_z_ci(r, n, z=1.96):
    zr = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    return np.tanh(zr - z * se), np.tanh(zr + z * se)


# ------------------------------------------------------------------ cluster bootstrap
def cluster_bootstrap_rate(df, by, cluster='person_id', success='passed', n_boot=1000, seed=1, alpha=0.05):
    """Rate with a CI that respects repeated rows of the same person (resamples PERSONS, not rows)."""
    by = [by] if isinstance(by, str) else list(by)
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in df.groupby(by, observed=True):
        agg = g.groupby(cluster).agg(n=(success, 'size'), k=(success, 'sum'))
        n = agg.n.to_numpy(float); k = agg.k.to_numpy(float); G = len(agg)
        est = k.sum() / n.sum()
        reps = np.empty(n_boot)
        for s in range(0, n_boot, 100):
            m = min(100, n_boot - s)
            w = rng.multinomial(G, np.full(G, 1 / G), size=m)
            reps[s:s + m] = (w @ k) / (w @ n)
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
        rec = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        rec.update(rows=int(n.sum()), persons=G, passes=int(k.sum()), rate=est, ci_low=lo, ci_high=hi)
        rows.append(rec)
    return pd.DataFrame(rows)


def cluster_bootstrap_stat(values_by_cluster, stat_fn, n_boot=1000, seed=1):
    """Generic person-cluster bootstrap. values_by_cluster: list of arrays-of-rows grouped by person."""
    rng = np.random.default_rng(seed)
    G = len(values_by_cluster)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        out[b] = stat_fn(np.vstack([values_by_cluster[i] for i in idx]))
    return np.quantile(out, [0.025, 0.975])


# ------------------------------------------------------------------ logistic regression (IRLS + cluster-robust SE)
def design_matrix(df, numeric=(), categorical=(), refs=None):
    """categorical: column names; refs: {column: reference level}. Returns (X DataFrame incl. const)."""
    refs = refs or {}
    parts = [pd.Series(1.0, index=df.index, name='const')]
    for c in numeric:
        parts.append(df[c].astype(float).rename(c))
    for c in categorical:
        levels = sorted(df[c].astype(str).unique())
        ref = refs.get(c, levels[0])
        if ref not in levels:            # reference level absent in this sample -> use the first level
            ref = levels[0]
        for lv in levels:
            if lv != ref:
                parts.append((df[c].astype(str) == lv).astype(float).rename(f'{c}[{lv}]'))
    return pd.concat(parts, axis=1)


def logit_fit(X, y, cluster=None, max_iter=100, tol=1e-10):
    names = list(X.columns); Xv = X.to_numpy(float); yv = np.asarray(y, float)
    n, k = Xv.shape
    beta = np.zeros(k)
    converged = False

    def loglik(b):
        e = np.clip(Xv @ b, -30, 30)
        return np.sum(yv * e - np.log1p(np.exp(e)))
    ll_old = loglik(beta)
    for _ in range(max_iter):
        eta = np.clip(Xv @ beta, -30, 30); p = 1 / (1 + np.exp(-eta)); W = p * (1 - p)
        z = eta + (yv - p) / np.maximum(W, 1e-12)
        A = Xv.T @ (Xv * W[:, None])
        new = np.linalg.solve(A, Xv.T @ (W * z))
        step = 1.0
        while loglik(beta + step * (new - beta)) < ll_old - 1e-9 and step > 1e-4:
            step /= 2                                    # step halving keeps the likelihood increasing
        new = beta + step * (new - beta); ll_new = loglik(new)
        if np.max(np.abs(new - beta)) < tol:
            beta = new; converged = True; break
        beta, ll_old = new, ll_new
    eta = np.clip(Xv @ beta, -30, 30); p = 1 / (1 + np.exp(-eta)); W = p * (1 - p)
    bread = np.linalg.inv(Xv.T @ (Xv * W[:, None]))
    if cluster is None:
        V = bread
    else:
        cl = pd.factorize(np.asarray(cluster))[0]; G = cl.max() + 1
        s = Xv * (yv - p)[:, None]
        S = np.zeros((G, k)); np.add.at(S, cl, s)
        meat = S.T @ S
        V = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V)); zst = beta / se
    ll = np.sum(yv * np.log(np.clip(p, 1e-12, 1)) + (1 - yv) * np.log(np.clip(1 - p, 1e-12, 1)))
    p0 = yv.mean(); ll0 = n * (p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    res = pd.DataFrame({'coef': beta, 'se': se, 'z': zst, 'p': 2 * stats.norm.sf(np.abs(zst)),
                        'odds_ratio': np.exp(beta), 'or_low': np.exp(beta - 1.96 * se),
                        'or_high': np.exp(beta + 1.96 * se)}, index=names)
    res.attrs['separation_warning'] = bool(np.any(np.abs(beta) > 12))
    res.attrs.update(n=n, clusters=(None if cluster is None else int(G)), loglik=ll,
                     mcfadden_r2=1 - ll / ll0, converged=converged, beta=beta, V=V, names=names)
    return res


def predict_logit(res, X):
    eta = X.to_numpy(float) @ res.attrs['beta']
    return 1 / (1 + np.exp(-eta))


def average_predictions(res, X, set_values):
    """Average predicted probability if everybody had the given dummy values (marginal standardisation)."""
    X2 = X.copy()
    for col, v in set_values.items():
        X2[col] = v
    return predict_logit(res, X2).mean()


# ------------------------------------------------------------------ institution comparison helpers
def eta_squared_adj(score, groups):
    d = pd.DataFrame({'s': np.asarray(score, float), 'g': np.asarray(groups)})
    gm = d.s.mean(); sst = ((d.s - gm) ** 2).sum()
    agg = d.groupby('g').s.agg(['count', 'mean'])
    ssb = (agg['count'] * (agg['mean'] - gm) ** 2).sum()
    eta2 = ssb / sst; n, k = len(d), len(agg)
    omega2 = (ssb - (k - 1) * (sst - ssb) / (n - k)) / (sst + (sst - ssb) / (n - k))
    return dict(eta2=eta2, omega2=max(omega2, 0.0), groups=k)


def beta_binomial_shrinkage(k, n):
    """Empirical-Bayes (method of moments) shrinkage of proportions towards the grand mean."""
    k = np.asarray(k, float); n = np.asarray(n, float)
    p = k / n; pbar = k.sum() / n.sum()
    s2 = np.sum(n * (p - pbar) ** 2) / n.sum()                 # weighted variance of observed rates
    sampling = np.sum(n * pbar * (1 - pbar) / n) / n.sum()     # expected variance from sampling alone
    tau2 = max(s2 - sampling, 1e-9)                            # true between-institution variance
    m = max(pbar * (1 - pbar) / tau2 - 1, 0.1)                 # prior sample size (alpha+beta)
    a, b = pbar * m, (1 - pbar) * m
    return (k + a) / (n + a + b), dict(prior_mean=pbar, tau=np.sqrt(tau2), prior_size=m, a=a, b=b)


def funnel_limits(p0, n_grid, z):
    se = np.sqrt(p0 * (1 - p0) / n_grid)
    return np.clip(p0 - z * se, 0, 1), np.clip(p0 + z * se, 0, 1)


def kitagawa(w0, r0, w1, r1):
    """Decompose change in a pooled rate between two periods into composition and rate effects.
    w*: group shares (sum to 1); r*: group rates. Returns (total, composition, rate)."""
    w0, r0, w1, r1 = map(np.asarray, (w0, r0, w1, r1))
    comp = np.sum((w1 - w0) * (r0 + r1) / 2)
    rate = np.sum((r1 - r0) * (w0 + w1) / 2)
    return comp + rate, comp, rate
