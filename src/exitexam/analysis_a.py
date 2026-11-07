"""PART A - the three rounds that say which institution a candidate graduated from (2025, 2026-R1, 2026-R2).

Institution information exists only in these rounds, so every institution / province comparison is restricted to
them. Candidate history (first attempt vs retaking) is taken from the five-round linkage (Part B), because a
2025 candidate may already have sat 2023 or 2024; the ROWS analysed are still only the three Part-A rounds.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from . import config as C, stats_utils as su, plotting as P, labels as L, analysis_b as B

A_ROUNDS = C.PART_A_EXAMS


def prepare(h):
    a = h[h.exam.isin(A_ROUNDS) & (h.inst_type != 'unknown')].copy()
    a['type'] = a.inst_type.map(L.TYPE_EN)
    a['field_en'] = a.field.map(L.FIELD_EN)
    a['label'] = [L.institute_label(t, p, i) for t, p, i in zip(a.inst_type, a.province, a.institute)]
    a['province_en'] = a.province.map(lambda x: L.PROVINCE_EN.get(x, x))
    a['round'] = a.exam
    return a


def coverage(h, a):
    t = pd.crosstab(a.exam, a.type).reindex(A_ROUNDS)
    t['total_valid'] = t.sum(axis=1)
    t['rows_without_institution'] = [(h[(h.exam == e)].shape[0] - (a.exam == e).sum()) for e in A_ROUNDS]
    t['institutes_seen'] = a.groupby('exam').label.nunique().reindex(A_ROUNDS)
    return t.reset_index()


# ------------------------------------------------------------------ type comparison
def type_summary(a):
    rows = []
    for (e, t), g in a.groupby(['exam', 'type'], sort=False):
        f = g[~g.passed]; n = len(g); k = int(g.passed.sum()); lo, hi = su.wilson_ci(k, n)
        fi = g[~g.returning]
        rows.append(dict(exam=e, type=t, rows=n, passes=k, pass_rate=k / n, ci_low=lo, ci_high=hi,
                         mean_score=g.score.mean(), median_score=g.score.median(), sd_score=g.score.std(),
                         near_miss_share_of_fails=(f.gap > -C.NEAR_MISS).mean() if len(f) else np.nan,
                         n_first_attempt=len(fi), pass_rate_first=fi.passed.mean() if len(fi) else np.nan,
                         n_retaking=n - len(fi), pass_rate_retaking=g[g.returning].passed.mean() if n > len(fi) else np.nan))
    out = pd.DataFrame(rows)
    out['exam'] = pd.Categorical(out.exam, A_ROUNDS, ordered=True)
    out['type'] = pd.Categorical(out.type, ['Public', 'Private', 'Abroad'], ordered=True)
    out = out.sort_values(['exam', 'type']).reset_index(drop=True)
    pooled = su.cluster_bootstrap_rate(a, 'type', n_boot=C.N_BOOT, seed=C.SEED)
    return out, pooled


def type_tests(a):
    rows = []
    for e in A_ROUNDS + ['pooled']:
        d = a if e == 'pooled' else a[a.exam == e]
        pub, pri = d[d.type == 'Public'], d[d.type == 'Private']
        t = su.two_prop_test(pri.passed.sum(), len(pri), pub.passed.sum(), len(pub))
        tab = pd.crosstab(d.type, d.passed)
        cv = su.cramers_v(tab)
        cd = su.cliffs_delta(pub.score, pri.score)
        a2, b2, c2, d2 = pub.passed.sum(), (~pub.passed).sum(), pri.passed.sum(), (~pri.passed).sum()
        orr = (a2 * d2) / (b2 * c2); se = np.sqrt(1 / a2 + 1 / b2 + 1 / c2 + 1 / d2)
        rows.append(dict(scope=e, public_pass=pub.passed.mean(), private_pass=pri.passed.mean(),
                         diff_private_minus_public=t['diff'], diff_ci_low=t['ci_low'], diff_ci_high=t['ci_high'],
                         odds_ratio_public_vs_private=orr, or_low=np.exp(np.log(orr) - 1.96 * se), or_high=np.exp(np.log(orr) + 1.96 * se),
                         chi2_p_3types=cv['p'], cramers_v=cv['cramers_v'], cliffs_delta_public_vs_private=cd['cliffs_delta'],
                         note='pooled rows repeat persons: treat p-values as descriptive; see model table (cluster-robust)' if e == 'pooled' else ''))
    return pd.DataFrame(rows)


def burden(a):
    rows = []
    for e in A_ROUNDS + ['pooled']:
        d = a if e == 'pooled' else a[a.exam == e]
        f = d[~d.passed]
        for t in ['Public', 'Private', 'Abroad']:
            rows.append(dict(scope=e, type=t, share_of_candidates=(d.type == t).mean(), share_of_failures=(f.type == t).mean()))
    return pd.DataFrame(rows)


def field_table(a):
    g = a.groupby(['exam', 'type', 'field_en'], observed=True).agg(rows=('passed', 'size'), passes=('passed', 'sum'),
                                                                   mean_score=('score', 'mean')).reset_index()
    g['pass_rate'] = g.passes / g.rows
    g['ci_low'], g['ci_high'] = su.wilson_ci(g.passes, g.rows)
    return g[g.field_en != 'Unknown']


# ------------------------------------------------------------------ institutes
def institute_tables(a):
    """Institute comparison on FIRST attempts only (one row per person), adjusted for round, field and type
    (indirect standardisation: observed vs expected passes), with false-discovery-rate control."""
    d = a[a.type.isin(['Public', 'Private'])].copy()
    first = d[~d.returning].copy()
    # expected pass probability from round + field + institute TYPE only (no institute information)
    X = su.design_matrix(first, categorical=['round', 'field_en', 'type'], refs={'round': '2025', 'field_en': 'Medicine', 'type': 'Public'})
    base = su.logit_fit(X, first.passed)
    first['p_exp'] = su.predict_logit(base, X)
    meta = first.groupby('label').agg(type=('type', 'first'), province=('province_en', 'first'), n_first=('passed', 'size'),
                                      k_first=('passed', 'sum'), mean_score_first=('score', 'mean'),
                                      expected=('p_exp', 'sum'), var=('p_exp', lambda p: np.sum(p * (1 - p))))
    ps = first.groupby('label')[['province', 'institute']].first().rename(columns={'province': 'province_ps', 'institute': 'institute_ps'})
    meta = meta.join(ps)
    meta['pass_rate_first'] = meta.k_first / meta.n_first
    meta['ci_low'], meta['ci_high'] = su.wilson_ci(meta.k_first, meta.n_first)
    meta['obs_over_expected'] = meta.k_first / meta.expected
    meta['z'] = (meta.k_first - meta.expected) / np.sqrt(meta['var'])
    meta['p_value'] = 2 * stats.norm.sf(np.abs(meta.z))
    ranked = meta.n_first >= C.MIN_INSTITUTE_N
    meta['ranked'] = ranked
    meta['p_bh'] = np.nan
    meta.loc[ranked, 'p_bh'] = su.benjamini_hochberg(meta.loc[ranked, 'p_value'])
    meta['flag'] = np.where(meta.p_bh < .05, np.where(meta.z > 0, 'better than expected', 'worse than expected'), '')
    # empirical-Bayes shrunken first-attempt rate within each type
    meta['pass_rate_first_shrunk'] = np.nan
    for t, g in meta[ranked].groupby('type'):
        sh, _ = su.beta_binomial_shrinkage(g.k_first.to_numpy(), g.n_first.to_numpy())
        meta.loc[g.index, 'pass_rate_first_shrunk'] = sh
    # all attempts, per-person cluster bootstrap CI
    allr = su.cluster_bootstrap_rate(d, 'label', n_boot=C.N_BOOT, seed=C.SEED).rename(
        columns={'rows': 'rows_all_attempts', 'persons': 'persons_all', 'rate': 'pass_rate_all_attempts',
                 'ci_low': 'all_ci_low', 'ci_high': 'all_ci_high'}).drop(columns='passes').set_index('label')
    meta = meta.drop(columns=['var']).join(allr)
    meta = meta.sort_values(['type', 'pass_rate_first'], ascending=[True, False]).reset_index()
    # homogeneity within type: are institutes different beyond chance?
    rows = []
    for t, g in meta[meta.ranked].groupby('type'):
        tab = np.vstack([g.k_first, g.n_first - g.k_first])
        chi2, p, dof, _ = stats.chi2_contingency(tab, correction=False)
        _, info = su.beta_binomial_shrinkage(g.k_first.to_numpy(), g.n_first.to_numpy())
        rows.append(dict(type=t, institutes=len(g), chi2=chi2, dof=dof, p_value=p,
                         sd_of_true_rates_between_institutes=info['tau'], mean_rate=info['prior_mean']))
    hetero = pd.DataFrame(rows)
    # how much of the score variation is explained by each grouping (first attempts, adjusted omega-squared)
    rows = []
    for name, grp in [('Institute type', first.type), ('Province', first.province_en), ('Institute', first.label),
                      ('Round', first['round']), ('Field', first.field_en)]:
        r = su.eta_squared_adj(first.score, grp)
        rows.append(dict(grouping=name, groups=r['groups'], eta_squared=r['eta2'], omega_squared_adjusted=r['omega2']))
    var = pd.DataFrame(rows)
    return meta, hetero, var, base, first


def province_table(a):
    first = a[~a.returning & a.type.isin(['Public', 'Private'])]
    g = first.groupby(['type', 'province_en']).agg(n_first=('passed', 'size'), passes=('passed', 'sum'),
                                                   mean_score=('score', 'mean')).reset_index()
    g['pass_rate_first'] = g.passes / g.n_first
    g['ci_low'], g['ci_high'] = su.wilson_ci(g.passes, g.n_first)
    return g[g.n_first >= 30].sort_values(['type', 'pass_rate_first'], ascending=[True, False]).reset_index(drop=True)


def institute_round_matrix(a):
    """Pass rate and number of candidates for every institute and round (all attempts). Nothing is hidden here;
    the figure marks small cells instead of dropping them."""
    d = a[a.type.isin(['Public', 'Private'])]
    g = d.groupby(['label', 'exam']).agg(n=('passed', 'size'), rate=('passed', 'mean')).reset_index()
    rate = g.pivot(index='label', columns='exam', values='rate').reindex(columns=A_ROUNDS)
    n = g.pivot(index='label', columns='exam', values='n').reindex(columns=A_ROUNDS).fillna(0).astype(int)
    return rate, n


# ------------------------------------------------------------------ models
def models(a):
    d = a.copy()
    d['prior_fails_cat'] = d.prior_fails.clip(upper=2).map({0: '0', 1: '1', 2: '2+'})
    d['field_c'] = d.field_en
    refs = {'type': 'Public', 'round': '2025', 'field_c': 'Medicine', 'prior_fails_cat': '0'}
    d = d[(d.prior_passes == 0) & (d.field != 'unknown')]
    out = {}
    X = su.design_matrix(d, categorical=['type', 'field_c', 'round'], refs=refs)
    out['MA1 type + field + round'] = su.logit_fit(X, d.passed, cluster=d.person_id)
    X2 = su.design_matrix(d, categorical=['type', 'field_c', 'round', 'prior_fails_cat'], refs=refs)
    out['MA2 + previous failures'] = su.logit_fit(X2, d.passed, cluster=d.person_id)
    f = d[~d.returning]
    X3 = su.design_matrix(f, categorical=['type', 'field_c', 'round'], refs=refs)
    out['MA3 first attempts only'] = su.logit_fit(X3, f.passed, cluster=f.person_id)
    return out, d, X2


def predicted_grid(res, X, d, n_draw=4000):
    """Model-based pass probabilities for typical candidate profiles, with parametric 95% intervals."""
    rng = np.random.default_rng(C.SEED)
    beta, V = res.attrs['beta'], res.attrs['V']; names = res.attrs['names']
    draws = rng.multivariate_normal(beta, V, size=n_draw)
    rows = []
    for t in ['Public', 'Private']:
        for fails in ['0', '1']:
            for r in A_ROUNDS:
                x = pd.Series(0.0, index=names); x['const'] = 1.0
                if t != 'Public': x[f'type[{t}]'] = 1.0
                if r != '2025': x[f'round[{r}]'] = 1.0
                if fails != '0': x[f'prior_fails_cat[{fails}]'] = 1.0
                p = 1 / (1 + np.exp(-(draws @ x.to_numpy())))
                rows.append(dict(type=t, history='First attempt' if fails == '0' else 'Failed once before', round=r,
                                 predicted_pass=1 / (1 + np.exp(-(beta @ x.to_numpy()))),
                                 ci_low=np.quantile(p, .025), ci_high=np.quantile(p, .975)))
    return pd.DataFrame(rows)


def type_transitions(h, a):
    F = B._failed_pairs(h)
    F = F[F.transition.isin(['2025 > 2026-R1', '2026-R1 > 2026-R2'])]
    F = F.merge(a[['person_id', 'exam', 'type']].drop_duplicates(['person_id', 'exam']), on=['person_id', 'exam'], how='inner')
    rows = []
    for (t, ty), g in F.groupby(['transition', 'type'], sort=False):
        n, r, pr = len(g), int(g.returned.sum()), int(g.pass_next.sum())
        l1, h1 = su.wilson_ci(r, n); l2, h2 = su.wilson_ci(pr, r) if r else (np.nan, np.nan)
        rows.append(dict(transition=t, type=ty, failed=n, returned=r, return_rate=r / n, ret_low=l1, ret_high=h1,
                         pass_given_return=pr / r if r else np.nan, pass_low=l2, pass_high=h2))
    out = pd.DataFrame(rows)
    out['type'] = pd.Categorical(out.type, ['Public', 'Private', 'Abroad'], ordered=True)
    return out.sort_values(['transition', 'type']).reset_index(drop=True)


def odds_by_round(a):
    """Private-vs-public odds ratio inside each round (no modelling, 2x2 tables)."""
    rows = []
    for e in A_ROUNDS:
        for scope, d in [('all candidates', a[a.exam == e]), ('first attempts', a[(a.exam == e) & ~a.returning])]:
            pub, pri = d[d.type == 'Public'], d[d.type == 'Private']
            a1, b1, c1, d1 = pub.passed.sum(), (~pub.passed).sum(), pri.passed.sum(), (~pri.passed).sum()
            orr = (a1 * d1) / (b1 * c1); se = np.sqrt(1 / a1 + 1 / b1 + 1 / c1 + 1 / d1)
            rows.append(dict(round=e, scope=scope, public_pass=pub.passed.mean(), private_pass=pri.passed.mean(),
                             or_public_vs_private=orr, or_low=np.exp(np.log(orr) - 1.96 * se), or_high=np.exp(np.log(orr) + 1.96 * se)))
    return pd.DataFrame(rows)


# ====================================================================== figures
def fig_type_by_round(ts):
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); types = ['Public', 'Private', 'Abroad']; w = .26; x = np.arange(len(A_ROUNDS))
    for i, t in enumerate(types):
        g = ts[ts.type == t].set_index('exam').reindex(A_ROUNDS)
        err = np.vstack([g.pass_rate - g.ci_low, g.ci_high - g.pass_rate])
        ax.bar(x + (i - 1) * w, g.pass_rate, w, yerr=err, capsize=3, color=P.TYPE_COLORS[t], label=f'{t} (n={g.rows.sum():,})')
        for xi, v in zip(x + (i - 1) * w, g.pass_rate): ax.text(xi, v + .04, f'{v:.0%}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([P.ROUND_LABEL[e] for e in A_ROUNDS]); P.pct_axis(ax); ax.set_ylabel('Pass rate (all candidates)')
    ax.set_ylim(0, 1.0); ax.set_title('Public-institute graduates pass far more often, in every round'); ax.legend()
    return fig


def fig_type_ecdf(a):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, e in zip(axes, A_ROUNDS):
        for t in ['Public', 'Private', 'Abroad']:
            s = np.sort(a.loc[(a.exam == e) & (a.type == t), 'score'].to_numpy())
            ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=P.TYPE_COLORS[t], lw=2, label=t)
        ax.axvline(C.CUTOFF, color='k', ls='--', lw=1); ax.set_title(P.ROUND_LABEL[e]); ax.set_xlabel('Score')
    axes[0].set_ylabel('Share scoring at or below'); P.pct_axis(axes[0]); axes[0].legend()
    fig.suptitle('Score distributions by institute type (dashed = pass mark)', fontweight='bold', y=1.03)
    return fig


def fig_institutes(meta):
    m = meta[meta.ranked].sort_values('pass_rate_first')
    fig, ax = plt.subplots(figsize=(8.6, 0.27 * len(m) + 1.8)); y = np.arange(len(m))
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.errorbar(r.pass_rate_first, yi, xerr=[[r.pass_rate_first - r.ci_low], [r.ci_high - r.pass_rate_first]], fmt='o',
                    color=P.TYPE_COLORS[r.type], capsize=2, ms=5)
    ax.set_yticks(y); ax.set_yticklabels(m.label, fontsize=8); P.pct_axis(ax, 'x'); ax.set_xlim(0, 1)
    ax.set_xlabel('Pass rate at FIRST attempt (95% CI), rounds 2025 + 2026 R1 + 2026 R2 pooled')
    ax.grid(axis='x', alpha=.6); ax.grid(axis='y', visible=False)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker='o', color=P.TYPE_COLORS[t], ls='') for t in ['Public', 'Private']], ['Public', 'Private'], loc='lower right')
    ax.set_title(f'Institutes differ widely (n >= {C.MIN_INSTITUTE_N} first attempts)')
    return fig


def fig_funnel(meta, first):
    m = meta[meta.ranked]; pbar = first.passed.mean()
    fig, ax = plt.subplots(figsize=(8.6, 4.8)); ns = np.linspace(max(m.n_first.min() * .8, 10), m.n_first.max() * 1.05, 200)
    for z, ls in [(1.96, '--'), (3.09, ':')]:
        se = np.sqrt(first.p_exp.mean() * (1 - first.p_exp.mean()) / ns) / first.p_exp.mean()
        ax.plot(ns, 1 + z * se, 'k' + ls, lw=1); ax.plot(ns, 1 - z * se, 'k' + ls, lw=1)
    ax.axhline(1, color='k', lw=1)
    for _, r in m.iterrows():
        ax.scatter(r.n_first, r.obs_over_expected, color=P.TYPE_COLORS[r.type], s=40, zorder=3, edgecolor='white')
        if abs(r.z) > 3.09: ax.annotate(r.label, (r.n_first, r.obs_over_expected), fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel('First-attempt candidates (log scale)'); ax.set_ylabel('Observed / expected passes')
    ax.set_title('Institutes compared with what their type, round and field predict\n(dashed = 95%, dotted = 99.8% limits; labelled = beyond 99.8%)')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([0], [0], marker='o', color=P.TYPE_COLORS[t], ls='') for t in ['Public', 'Private']], ['Public', 'Private'])
    return fig


def fig_heatmap(rate, n, min_show=10, solid=30):
    """Every institute and round with at least `min_show` candidates is drawn and labelled with its pass rate and n.
    Cells with fewer than `solid` candidates are shown faded: read them with caution, but they are not hidden."""
    import seaborn as sns
    from matplotlib.patches import Rectangle
    shown = rate.where(n >= min_show)
    keep = shown.notna().sum(axis=1) >= 2
    shown, n2 = shown[keep], n[keep]
    order = shown.mean(axis=1).sort_values(ascending=False).index
    shown, n2 = shown.loc[order], n2.loc[order]
    annot = shown.copy().astype(object)
    for i in shown.index:
        for c in shown.columns:
            annot.loc[i, c] = '' if pd.isna(shown.loc[i, c]) else f'{shown.loc[i, c]:.0%}\nn={n2.loc[i, c]}'
    fig, ax = plt.subplots(figsize=(6.8, 0.46 * len(shown) + 2.0))
    sns.heatmap(shown.rename(columns=P.ROUND_LABEL), annot=annot.rename(columns=P.ROUND_LABEL).to_numpy(), fmt='', cmap='YlGnBu',
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Pass rate'}, annot_kws={'fontsize': 7},
                linewidths=.5, linecolor='white')
    for ri, i in enumerate(shown.index):
        for ci, c in enumerate(shown.columns):
            if not pd.isna(shown.loc[i, c]) and n2.loc[i, c] < solid:
                ax.add_patch(Rectangle((ci, ri), 1, 1, fill=True, color='white', alpha=.6, lw=0))
    ax.set_ylabel(''); ax.set_xlabel(''); ax.grid(False)
    ax.set_title(f'Pass rate by institute and round (all attempts)\nFaded cells: fewer than {solid} candidates; blank: fewer than {min_show}')
    return fig


def fig_province(pt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={'width_ratios': [1.2, 1]})
    for ax, t in zip(axes, ['Private', 'Public']):
        g = pt[pt.type == t].sort_values('pass_rate_first')
        y = np.arange(len(g)); err = np.vstack([g.pass_rate_first - g.ci_low, g.ci_high - g.pass_rate_first])
        ax.barh(y, g.pass_rate_first, xerr=err, color=P.TYPE_COLORS[t], capsize=2); ax.set_yticks(y)
        ax.set_yticklabels([f'{p} (n={n})' for p, n in zip(g.province_en, g.n_first)], fontsize=8); P.pct_axis(ax, 'x'); ax.set_xlim(0, 1)
        ax.set_title(f'{t} institutes'); ax.set_xlabel('First-attempt pass rate'); ax.grid(axis='y', visible=False); ax.grid(axis='x')
    fig.suptitle('Province differences within each type (provinces with n >= 30)', fontweight='bold', y=1.02)
    return fig


def fig_predicted(grid):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, hist in zip(axes, ['First attempt', 'Failed once before']):
        for t, off in [('Public', -.1), ('Private', .1)]:
            g = grid[(grid.history == hist) & (grid.type == t)].set_index('round').reindex(A_ROUNDS)
            x = np.arange(len(A_ROUNDS)) + off
            err = np.vstack([g.predicted_pass - g.ci_low, g.ci_high - g.predicted_pass])
            ax.errorbar(x, g.predicted_pass, yerr=err, fmt='o-', color=P.TYPE_COLORS[t], capsize=3, lw=2, label=t)
        ax.set_xticks(np.arange(len(A_ROUNDS))); ax.set_xticklabels([P.ROUND_LABEL[e] for e in A_ROUNDS]); ax.set_title(hist); P.pct_axis(ax)
    axes[0].set_ylabel('Model-predicted pass probability (medicine)'); axes[0].legend()
    fig.suptitle('Predicted chance of passing: institute type x history x round', fontweight='bold', y=1.03)
    return fig


def fig_burden(bd):
    d = bd[bd.scope == 'pooled'].set_index('type').reindex(['Public', 'Private', 'Abroad'])
    fig, ax = plt.subplots(figsize=(6.6, 4)); x = np.arange(3); w = .36
    ax.bar(x - w / 2, d.share_of_candidates, w, color='#b8c4d0', label='Share of candidates')
    ax.bar(x + w / 2, d.share_of_failures, w, color='#c0392b', label='Share of all failures')
    for xi, a1, b1 in zip(x, d.share_of_candidates, d.share_of_failures):
        ax.text(xi - w / 2, a1 + .01, f'{a1:.0%}', ha='center', fontsize=8); ax.text(xi + w / 2, b1 + .01, f'{b1:.0%}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(d.index); P.pct_axis(ax); ax.set_title('Private-institute graduates are most candidates and most failures'); ax.legend()
    return fig


def fig_type_transitions(tt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); transitions = list(tt.transition.unique()); types = ['Public', 'Private']; w = .35
    for ax, (col, lo, hi, ttl) in zip(axes, [('return_rate', 'ret_low', 'ret_high', 'Came back after failing'),
                                           ('pass_given_return', 'pass_low', 'pass_high', 'Passed, among those who came back')]):
        x = np.arange(len(transitions))
        for i, t in enumerate(types):
            g = tt[tt.type == t].set_index('transition').reindex(transitions)
            err = np.vstack([g[col] - g[lo], g[hi] - g[col]])
            ax.bar(x + (i - .5) * w, g[col], w, yerr=err, capsize=3, color=P.TYPE_COLORS[t], label=t)
        ax.set_xticks(x); ax.set_xticklabels([t.replace('2026-R1', '2026 R1').replace('2026-R2', '2026 R2').replace('>', 'to') for t in transitions]); P.pct_axis(ax); ax.set_title(ttl)
    axes[0].legend()
    return fig
