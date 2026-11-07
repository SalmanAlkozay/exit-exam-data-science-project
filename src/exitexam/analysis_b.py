"""PART B - five linked rounds (2023, 2024, 2025, 2026-R1, 2026-R2).

Questions: how does the pass probability change with the round, with the number of previous failures,
how many candidates come back after failing, and how likely are they to pass when they do?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import config as C, linking as lk, stats_utils as su, plotting as P, labels as L

BANDS = [-400, -60, -30, -10, 0]
BAND_LABELS = ['60+ below', '30-60 below', '10-30 below', '0-10 below']


# ====================================================================== tables
def exam_summary(h):
    rows = []
    for e in C.EXAMS:
        d = h[h.exam == e]; n = len(d); k = int(d.passed.sum()); lo, hi = su.wilson_ci(k, n)
        f = d[~d.passed]; new, ret = d[~d.returning], d[d.returning]
        r = dict(exam=e, rows=n, passes=k, pass_rate=k / n, ci_low=lo, ci_high=hi,
                 mean_score=d.score.mean(), median_score=d.score.median(), sd_score=d.score.std(),
                 q25=d.score.quantile(.25), q75=d.score.quantile(.75),
                 median_points_below_cutoff_among_fails=-f.gap.median(),
                 near_miss_share_of_fails=(f.gap > -C.NEAR_MISS).mean(),
                 far_fail_share_of_fails=(f.gap <= -60).mean(),
                 pct_medicine=(d.field == 'medicine').mean(),
                 n_new=len(new), pass_rate_new=new.passed.mean(),
                 n_returning=len(ret), pct_returning=len(ret) / n,
                 pass_rate_returning=ret.passed.mean() if len(ret) else np.nan)
        r['new_ci_low'], r['new_ci_high'] = su.wilson_ci(new.passed.sum(), len(new))
        r['ret_ci_low'], r['ret_ci_high'] = su.wilson_ci(ret.passed.sum(), len(ret)) if len(ret) else (np.nan, np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def exam_pairwise(h):
    rows = []
    for i in range(1, 5):
        a, b = h[h.exam_idx == i - 1], h[h.exam_idx == i]
        t = su.two_prop_test(b.passed.sum(), len(b), a.passed.sum(), len(a))
        rows.append(dict(comparison=f'{C.EXAMS[i]} vs {C.EXAMS[i-1]}', diff_pass_rate=t['diff'],
                         ci_low=t['ci_low'], ci_high=t['ci_high'], p_value=t['p']))
    out = pd.DataFrame(rows)
    out['p_holm'] = su.holm(out.p_value)
    return out


def field_by_round(h):
    g = h.groupby(['exam', 'field']).agg(rows=('passed', 'size'), passes=('passed', 'sum'),
                                          mean_score=('score', 'mean')).reset_index()
    g['pass_rate'] = g.passes / g.rows
    g['ci_low'], g['ci_high'] = su.wilson_ci(g.passes, g.rows)
    return g[g.field != 'unknown']


def participation(h):
    pt = lk.person_table(h)
    dist = pt.n_attempts.value_counts().sort_index().rename('persons').to_frame()
    dist['share_of_persons'] = dist.persons / dist.persons.sum()
    dist['ever_passed'] = pt.groupby('n_attempts').ever_passed.mean()
    dist['mean_first_score'] = pt.groupby('n_attempts').first_score.mean()
    dist['median_first_score'] = pt.groupby('n_attempts').first_score.median()
    dist['share_first_attempt_near_miss'] = pt.assign(nm=(pt.first_score - C.CUTOFF).between(-C.NEAR_MISS, 0)) \
        .groupby('n_attempts').nm.mean()
    dist = dist.reset_index()
    kw = __import__('scipy.stats', fromlist=['stats']).kruskal(
        *[g.first_score.values for _, g in pt.assign(grp=pt.n_attempts.clip(upper=4)).groupby('grp')])
    rho = pt[['n_attempts', 'first_score']].corr(method='spearman').iloc[0, 1]
    tests = pd.DataFrame([dict(test='Kruskal-Wallis: first-attempt score differs by number of attempts (1,2,3,4+)',
                               statistic=kw.statistic, p_value=kw.pvalue),
                          dict(test='Spearman rho: number of attempts vs first-attempt score',
                               statistic=rho, p_value=np.nan)])
    return pt, dist, tests


def decomposition(h):
    rows = []
    for i in range(2, 5):                           # 2023 has no identifiable returners (start of data)
        a, b = h[h.exam_idx == i - 1], h[h.exam_idx == i]

        def parts(d):
            w = d.returning.value_counts(normalize=True).reindex([False, True]).fillna(0).to_numpy()
            r = d.groupby('returning').passed.mean().reindex([False, True]).fillna(0).to_numpy()
            return w, r
        w0, r0 = parts(a); w1, r1 = parts(b)
        tot, comp, rate = su.kitagawa(w0, r0, w1, r1)
        rows.append(dict(from_exam=C.EXAMS[i - 1], to_exam=C.EXAMS[i], pass_rate_from=a.passed.mean(),
                         pass_rate_to=b.passed.mean(), total_change=tot, composition_effect=comp,
                         rate_effect=rate, share_returning_from=w0[1], share_returning_to=w1[1]))
    return pd.DataFrame(rows)


def _failed_pairs(h):
    """One row per FAILED attempt that could be followed up (not the last round)."""
    out = []
    for i in range(4):
        cur = h[(h.exam_idx == i) & (~h.passed)]
        nxt = h[h.exam_idx == i + 1][['person_id', 'passed', 'score']]
        later = set(h.loc[h.exam_idx > i, 'person_id'])
        m = cur.merge(nxt, on='person_id', how='left', suffixes=('', '_next'))
        m['returned'] = m.score_next.notna()
        m['pass_next'] = m.passed_next.eq(True)
        m['returned_any_later'] = m.person_id.isin(later)
        m['transition'] = f'{C.EXAMS[i]} > {C.EXAMS[i+1]}'
        m['band'] = pd.cut(m.gap, BANDS, labels=BAND_LABELS)
        out.append(m)
    return pd.concat(out, ignore_index=True)


def transitions(h):
    F = _failed_pairs(h)
    rows = []
    for t, g in F.groupby('transition', sort=False):
        n, r = len(g), int(g.returned.sum()); pr = int(g.pass_next.sum())
        lo1, hi1 = su.wilson_ci(r, n); lo2, hi2 = su.wilson_ci(pr, r)
        rows.append(dict(transition=t, failed=n, returned=r, return_rate=r / n, ret_low=lo1, ret_high=hi1,
                         pass_given_return=pr / r, pass_low=lo2, pass_high=hi2, pass_rate_among_all_failed=pr / n,
                         mean_score_change=(g.score_next - g.score).mean(),
                         returned_in_any_later_round=g.returned_any_later.mean()))
    # pooled over all consecutive transitions: person-cluster bootstrap (a person can appear in several)
    b1 = su.cluster_bootstrap_rate(F.assign(all=1), 'all', success='returned', n_boot=C.N_BOOT, seed=C.SEED)
    b2 = su.cluster_bootstrap_rate(F[F.returned].assign(all=1), 'all', success='pass_next', n_boot=C.N_BOOT, seed=C.SEED)
    pooled = dict(transition='ALL (pooled, cluster bootstrap)', failed=len(F), returned=int(F.returned.sum()),
                  return_rate=b1.rate[0], ret_low=b1.ci_low[0], ret_high=b1.ci_high[0],
                  pass_given_return=b2.rate[0], pass_low=b2.ci_low[0], pass_high=b2.ci_high[0],
                  pass_rate_among_all_failed=F.pass_next.mean(),
                  mean_score_change=(F.score_next - F.score).mean(),
                  returned_in_any_later_round=F.returned_any_later.mean())
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([pooled])], ignore_index=True), F


def gap_bands(F):
    """Return probability and success probability by how close the failed attempt was to the cut-off."""
    rows = []
    for (t, b), g in F.groupby(['transition', 'band'], observed=True, sort=False):
        n, r, pr = len(g), int(g.returned.sum()), int(g.pass_next.sum())
        lo1, hi1 = su.wilson_ci(r, n); lo2, hi2 = su.wilson_ci(pr, r) if r else (np.nan, np.nan)
        rows.append(dict(transition=t, band=b, failed=n, returned=r, return_rate=r / n, ret_low=lo1, ret_high=hi1,
                         pass_given_return=pr / r if r else np.nan, pass_low=lo2, pass_high=hi2))
    per = pd.DataFrame(rows)
    pooled_ret = su.cluster_bootstrap_rate(F, 'band', success='returned', n_boot=C.N_BOOT, seed=C.SEED)
    pooled_pass = su.cluster_bootstrap_rate(F[F.returned], 'band', success='pass_next', n_boot=C.N_BOOT, seed=C.SEED)
    pooled = pooled_ret.merge(pooled_pass, on='band', suffixes=('_return', '_pass'))
    return per, pooled


def hazard_models(h):
    """Logistic models of passing, restricted to candidates who have not passed yet. SEs clustered by person."""
    r = h[(h.prior_passes == 0) & (h.field != 'unknown')].copy()
    r['prior_fails_cat'] = r.prior_fails.clip(upper=3).map({0: '0', 1: '1', 2: '2', 3: '3+'})
    r['round'] = r.exam
    r['gap10'] = r.last_gap / 10.0
    r['rounds_since_last'] = np.where(r.exam_idx - r.last_exam_idx > 1, '2+', '1')
    refs = {'round': '2025', 'field': 'medicine', 'prior_fails_cat': '0', 'rounds_since_last': '1'}
    out = {}
    X = su.design_matrix(r, categorical=['round', 'field'], refs=refs)
    out['M1 round + field'] = su.logit_fit(X, r.passed, cluster=r.person_id)
    X = su.design_matrix(r, categorical=['round', 'field', 'prior_fails_cat'], refs=refs)
    out['M2 + number of previous failures'] = su.logit_fit(X, r.passed, cluster=r.person_id)
    ret = r[r.prior_fails >= 1].copy()
    ret['gap10_sq'] = ret.gap10 ** 2
    refs3 = dict(refs, prior_fails_cat='1')          # returners have at least 1 previous failure
    X = su.design_matrix(ret, numeric=['gap10', 'gap10_sq'],
                         categorical=['round', 'field', 'prior_fails_cat', 'rounds_since_last'], refs=refs3)
    out['M3 returners: + distance from cut-off at last attempt'] = su.logit_fit(X, ret.passed, cluster=ret.person_id)
    return out, r, ret


def models_to_table(models):
    rows = []
    for name, res in models.items():
        for term, rr in res.iterrows():
            rows.append(dict(model=name, term=term, odds_ratio=rr.odds_ratio, or_low=rr.or_low, or_high=rr.or_high,
                             p_value=rr.p, n_rows=res.attrs['n'], n_persons=res.attrs['clusters'],
                             mcfadden_r2=res.attrs['mcfadden_r2']))
    return pd.DataFrame(rows)


def prior_fails_table(h):
    """Pass probability of candidates who have not passed yet, by number of previous failures (descriptive)."""
    r = h[(h.prior_passes == 0) & (h.field != 'unknown')].copy()
    r['prior_failures'] = r.prior_fails.clip(upper=3).map({0: '0', 1: '1', 2: '2', 3: '3+'})
    pooled = su.cluster_bootstrap_rate(r, 'prior_failures', n_boot=C.N_BOOT, seed=C.SEED)
    by_round = r.groupby(['exam', 'prior_failures']).agg(rows=('passed', 'size'), passes=('passed', 'sum')).reset_index()
    by_round['pass_rate'] = by_round.passes / by_round.rows
    by_round['ci_low'], by_round['ci_high'] = su.wilson_ci(by_round.passes, by_round.rows)
    return pooled, by_round


def retest(h):
    """Do scores persist from one attempt to the next? (stable ability vs luck)"""
    rows = []
    for i in range(4):
        a = h[h.exam_idx == i][['person_id', 'score']]; b = h[h.exam_idx == i + 1][['person_id', 'score']]
        m = a.merge(b, on='person_id', suffixes=('_prev', '_next'))
        n = len(m); r = m.score_prev.corr(m.score_next); rs = m.score_prev.corr(m.score_next, method='spearman')
        lo, hi = su.fisher_z_ci(r, n)
        slope = np.polyfit(m.score_prev, m.score_next, 1)[0]
        d = m.score_next - m.score_prev
        rows.append(dict(transition=f'{C.EXAMS[i]} > {C.EXAMS[i+1]}', persons=n, pearson_r=r, r_low=lo, r_high=hi,
                         spearman_rho=rs, slope_next_on_prev=slope, mean_change=d.mean(),
                         mean_change_low=d.mean() - 1.96 * d.sem(), mean_change_high=d.mean() + 1.96 * d.sem(),
                         sd_prev=m.score_prev.std(), sd_next=m.score_next.std()))
    return pd.DataFrame(rows)


def cohorts(h):
    """Cumulative share of an entry cohort that has passed by each later round (no censoring bias)."""
    rows = []
    for e in C.EXAMS:
        ids = set(h.loc[(h.exam == e) & (h.attempt == 1), 'person_id'])
        coh = h[h.person_id.isin(ids)]
        for t in range(C.EXAM_INDEX[e], 5):
            sub = coh[coh.exam_idx <= t].groupby('person_id').passed.max()
            rows.append(dict(entry_round=e, by_round=C.EXAMS[t], cohort_size=len(ids), cumulative_pass=sub.mean(),
                             rounds_observed=t - C.EXAM_INDEX[e] + 1))
    cum = pd.DataFrame(rows)
    # what happened to those who failed their first attempt?
    rows = []
    for e in C.EXAMS[:-1]:
        ids = set(h.loc[(h.exam == e) & (h.attempt == 1) & (~h.passed), 'person_id'])
        later = h[h.person_id.isin(ids) & (h.exam_idx > C.EXAM_INDEX[e])]
        ever_back = later.person_id.nunique(); ever_pass = later[later.passed].person_id.nunique()
        lo, hi = su.wilson_ci(ever_pass, len(ids))
        rows.append(dict(entry_round=e, failed_first_attempt=len(ids), came_back=ever_back,
                         came_back_share=ever_back / len(ids), eventually_passed=ever_pass,
                         eventually_passed_share=ever_pass / len(ids), ep_low=lo, ep_high=hi,
                         never_came_back_and_not_passed=len(ids) - ever_back))
    return cum, pd.DataFrame(rows)


def attempts_for_2023_cohort(h):
    """The 2023 entry cohort has the longest follow-up (5 rounds): least truncated view of persistence."""
    ids = set(h.loc[(h.exam == '2023') & (h.attempt == 1), 'person_id'])
    pt = lk.person_table(h[h.person_id.isin(ids)])
    out = pt.groupby('n_attempts').agg(persons=('person_id', 'size'), ever_passed=('ever_passed', 'mean'),
                                       mean_first_score=('first_score', 'mean')).reset_index()
    out['share'] = out.persons / out.persons.sum()
    return out


def supplementary(h):
    out = {}
    g = h[(h.exam == '2023') & h.get('grad_year', pd.Series(index=h.index, dtype=object)).notna()] if 'grad_year' in h else None
    if g is not None and len(g):
        g = g.assign(grad_year=pd.to_numeric(g.grad_year, errors='coerce'))
        t = g.groupby('grad_year').agg(rows=('passed', 'size'), passes=('passed', 'sum'), mean_score=('score', 'mean')).reset_index()
        t = t[t.rows >= 30]; t['pass_rate'] = t.passes / t.rows; t['ci_low'], t['ci_high'] = su.wilson_ci(t.passes, t.rows)
        out['S2_graduation_year_2023'] = t
    return out


def linkage_sensitivity(clean, h, models):
    """Do the conclusions survive a looser linking rule (typo-tolerant grandfather name)?"""
    cf, extra = lk.fuzzy_person_ids(clean)
    c2 = cf.drop(columns=['person_id']).rename(columns={'person_id_fuzzy': 'person_id'})
    h2 = lk.add_history(c2)
    t2, _ = transitions(h2)
    m2, _, _ = hazard_models(h2)
    rows = []
    for label, hh, tt, mm in [('exact name+field (main)', h, transitions(h)[0], models),
                              ('typo-tolerant grandfather name', h2, t2, m2)]:
        pooled = tt[tt.transition.str.startswith('ALL')].iloc[0]
        res = mm['M2 + number of previous failures']
        rows.append(dict(linking_rule=label, persons=hh.person_id.nunique(), return_rate_after_fail=pooled.return_rate,
                         pass_given_return=pooled.pass_given_return,
                         OR_1_prev_fail=res.loc['prior_fails_cat[1]', 'odds_ratio'],
                         OR_2_prev_fails=res.loc['prior_fails_cat[2]', 'odds_ratio'],
                         OR_3plus_prev_fails=res.loc['prior_fails_cat[3+]', 'odds_ratio']))
    out = pd.DataFrame(rows)
    out.attrs['extra_links'] = extra
    return out


# ====================================================================== figures
def fig_overview(es):
    fig, ax = plt.subplots(figsize=(8, 4.4)); x = np.arange(len(es))
    ax.bar(x, es.rows, color='#dfe6ee', width=.62)
    ax.set_xticks(x); ax.set_xticklabels([P.ROUND_LABEL[e] for e in es.exam]); ax.set_ylabel('Candidates (valid scores)')
    ax.grid(False); ax2 = ax.twinx(); ax2.grid(True, axis='y', alpha=.5)
    err = np.vstack([es.pass_rate - es.ci_low, es.ci_high - es.pass_rate])
    ax2.errorbar(x, es.pass_rate, yerr=err, fmt='o-', color='#c0392b', capsize=4, lw=2, label='All candidates')
    ax2.plot(x, es.pass_rate_new, 's--', color='#2a9d8f', lw=1.5, label='First attempt')
    ax2.plot(x, es.pass_rate_returning, '^--', color='#6a4c93', lw=1.5, label='Retaking')
    for xi, v in zip(x, es.pass_rate):
        ax2.annotate(f'{v:.0%}', (xi, v), textcoords='offset points', xytext=(0, 9), ha='center', fontsize=9, color='#c0392b')
    ax2.set_ylim(0, 1); P.pct_axis(ax2); ax2.set_ylabel('Pass rate'); ax2.spines['right'].set_visible(True)
    ax2.legend(loc='upper left', frameon=True)
    ax.set_title('Pass rate fell sharply in 2026 R1, whoever the candidate was')
    return fig


def fig_distribution(h):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for e in C.EXAMS:
        s = np.sort(h.loc[h.exam == e, 'score'].to_numpy())
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), color=P.ROUND_COLORS[e], lw=2, label=P.ROUND_LABEL[e])
    ax.axvline(C.CUTOFF, color='k', ls='--', lw=1); ax.text(C.CUTOFF + 3, .04, f'pass mark {C.CUTOFF:.0f}', fontsize=9)
    ax.set_xlabel('Score'); ax.set_ylabel('Share of candidates scoring at or below'); P.pct_axis(ax)
    ax.set_title('Score distributions: the share below the pass mark differs by round'); ax.legend(title='Round')
    return fig


def fig_composition(es):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2)); x = np.arange(len(es)); lab = [P.ROUND_LABEL[e] for e in es.exam]
    axes[0].bar(x, 1 - es.pct_returning, color='#2a9d8f', label='First attempt')
    axes[0].bar(x, es.pct_returning, bottom=1 - es.pct_returning, color='#6a4c93', label='Retaking')
    axes[0].set_xticks(x); axes[0].set_xticklabels(lab); P.pct_axis(axes[0]); axes[0].set_ylabel('Share of candidates')
    axes[0].set_title('Who sits each round'); axes[0].legend(loc='lower right')
    axes[0].annotate('2023: no earlier\nrounds in the data', (0, .5), ha='center', color='white', fontsize=8)
    w = .38
    for off, col, ci, lbl in [(-w / 2, '#2a9d8f', ('new_ci_low', 'new_ci_high'), 'pass_rate_new'),
                              (w / 2, '#6a4c93', ('ret_ci_low', 'ret_ci_high'), 'pass_rate_returning')]:
        v = es[lbl].to_numpy(); lo = v - es[ci[0]].to_numpy(); hi = es[ci[1]].to_numpy() - v
        axes[1].bar(x + off, np.nan_to_num(v), w, color=col, yerr=np.vstack([np.nan_to_num(lo), np.nan_to_num(hi)]), capsize=3,
                    label='First attempt' if lbl == 'pass_rate_new' else 'Retaking')
    axes[1].set_xticks(x); axes[1].set_xticklabels(lab); P.pct_axis(axes[1]); axes[1].set_ylabel('Pass rate')
    axes[1].set_title('Retakers pass less often in every round'); axes[1].legend()
    return fig


def fig_transitions(F):
    t = F.groupby('transition', sort=False).apply(lambda g: pd.Series({
        'Did not return': (~g.returned).mean(), 'Returned, failed again': (g.returned & ~g.pass_next).mean(),
        'Returned and passed': g.pass_next.mean()}), include_groups=False)
    fig, ax = plt.subplots(figsize=(8.6, 4.4)); left = np.zeros(len(t))
    for col, c in zip(t.columns, ['#c9ced6', '#e45756', '#2a9d8f']):
        ax.barh(t.index, t[col], left=left, color=c, label=col)
        for yi, (l, v) in enumerate(zip(left, t[col])):
            if v > .05: ax.text(l + v / 2, yi, f'{v:.0%}', ha='center', va='center', color='white' if c != '#c9ced6' else 'k', fontsize=9)
        left += t[col].to_numpy()
    ax.invert_yaxis(); P.pct_axis(ax, 'x'); ax.set_xlabel('Share of candidates who failed in the first round')
    ax.set_title('What happens in the next round to those who failed'); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=3, frameon=False)
    return fig


def fig_gap(per):
    fig, ax = plt.subplots(figsize=(8.6, 4.6)); labs = BAND_LABELS
    for t, g in per.groupby('transition', sort=False):
        g = g.set_index('band').reindex(labs); x = np.arange(len(labs))
        e = t.split(' > ')[1]; col = P.ROUND_COLORS[e]
        err = np.vstack([g.pass_given_return - g.pass_low, g.pass_high - g.pass_given_return])
        ax.errorbar(x, g.pass_given_return, yerr=err, fmt='o-', color=col, capsize=3, lw=2, label=f'{P.ROUND_LABEL[t.split(" > ")[0]]} > {P.ROUND_LABEL[e]}')
    ax.set_xticks(np.arange(len(labs))); ax.set_xticklabels(labs); P.pct_axis(ax)
    ax.set_xlabel('Points below the pass mark at the failed attempt'); ax.set_ylabel('Chance of passing at the next attempt')
    ax.set_title('Closer to the pass mark helps, but the round matters even more'); ax.legend(title='Transition', fontsize=8)
    return fig


def fig_forest(models, title, terms=None, name_map=None):
    rows = []
    for name, res in models.items():
        for term, r in res.iterrows():
            if term == 'const' or (terms and not any(term.startswith(t) for t in terms)): continue
            rows.append((name, term, r.odds_ratio, r.or_low, r.or_high))
    d = pd.DataFrame(rows, columns=['model', 'term', 'or', 'lo', 'hi'])
    d['label'] = d.term.map(lambda t: (name_map or L.TERM_EN).get(t, t))
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(d) + 1.6))
    y = np.arange(len(d))[::-1]; cols = dict(zip(models.keys(), ['#457b9d', '#e45756', '#2a9d8f', '#6a4c93']))
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.errorbar(r['or'], yi, xerr=[[r['or'] - r.lo], [r.hi - r['or']]], fmt='o', color=cols[r.model], capsize=3)
    ax.axvline(1, color='k', lw=1); ax.set_xscale('log'); ax.set_yticks(y); ax.set_yticklabels(d.label)
    lo, hi = d.lo.min(), d.hi.max()
    ticks = [t for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20] if lo / 1.3 <= t <= hi * 1.3]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:g}' for t in ticks]); ax.minorticks_off()
    ax.set_xlabel('Odds ratio of passing (log scale, 95% CI; 1 = no difference)'); ax.set_title(title)
    from matplotlib.lines import Line2D
    if len(cols) > 1:
        ax.legend([Line2D([0], [0], marker='o', color=c, ls='') for c in cols.values()], list(cols.keys()),
                  loc='upper center', bbox_to_anchor=(.5, -.16), ncol=len(cols), fontsize=8, frameon=False)
    return fig


def fig_cohorts(cum):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for e, g in cum.groupby('entry_round', sort=False):
        ax.plot([P.ROUND_LABEL[b] for b in g.by_round], g.cumulative_pass, 'o-', lw=2, color=P.ROUND_COLORS[e],
                label=f'Entered {P.ROUND_LABEL[e]} (n={g.cohort_size.iloc[0]:,})')
    P.pct_axis(ax); ax.set_ylabel('Cumulative share who have passed'); ax.set_xlabel('Round')
    ax.set_title('Each entry cohort: how many eventually pass'); ax.legend(fontsize=8)
    return fig


def fig_attempts(pt, dist):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    g = pt.assign(grp=pt.n_attempts.clip(upper=4).map({1: '1', 2: '2', 3: '3', 4: '4-5'}))
    order = ['1', '2', '3', '4-5']
    data = [g.loc[g.grp == k, 'first_score'] for k in order]
    bp = axes[0].boxplot(data, tick_labels=order, patch_artist=True, showfliers=False, medianprops=dict(color='k'))
    for p, c in zip(bp['boxes'], ['#2a9d8f', '#72b7b2', '#e9a03b', '#e45756']): p.set_facecolor(c)
    axes[0].axhline(C.CUTOFF, color='k', ls='--', lw=1); axes[0].text(.55, C.CUTOFF + 4, 'pass mark', fontsize=8)
    axes[0].set_xlabel('Total number of attempts in the data'); axes[0].set_ylabel('Score at FIRST attempt')
    axes[0].set_title('Frequent re-sitters started further below')
    d = dist.assign(grp=dist.n_attempts.clip(upper=4)).groupby('grp').apply(
        lambda x: pd.Series({'persons': x.persons.sum(), 'ever': np.average(x.ever_passed, weights=x.persons)}), include_groups=False)
    axes[1].bar(['1', '2', '3', '4-5'], d.ever, color=['#2a9d8f', '#72b7b2', '#e9a03b', '#e45756'])
    for i, v in enumerate(d.ever): axes[1].text(i, v + .01, f'{v:.0%}', ha='center', fontsize=9)
    P.pct_axis(axes[1]); axes[1].set_xlabel('Total number of attempts in the data'); axes[1].set_ylabel('Share who passed at some point')
    axes[1].set_title('Pass share falls with the number of attempts\n(survivorship: passers stop sitting)')
    return fig


def fig_retest(h):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        a = h[h.exam_idx == i][['person_id', 'score']]; b = h[h.exam_idx == i + 1][['person_id', 'score']]
        m = a.merge(b, on='person_id', suffixes=('_prev', '_next'))
        ax.hexbin(m.score_prev, m.score_next, gridsize=28, cmap='Blues', mincnt=1, extent=(20, 350, 20, 350))
        ax.plot([20, 350], [20, 350], 'k--', lw=1); ax.axhline(C.CUTOFF, color='#c0392b', lw=1); ax.axvline(C.CUTOFF, color='#c0392b', lw=1)
        ax.set_title(f'{P.ROUND_LABEL[C.EXAMS[i]]} to {P.ROUND_LABEL[C.EXAMS[i+1]]}  (r = {m.score_prev.corr(m.score_next):.2f})', fontsize=10)
        ax.set_xlabel('Score in earlier round')
        if i == 0: ax.set_ylabel('Score in next round')
    fig.suptitle('Scores persist strongly between attempts, but each round shifts them up or down', fontweight='bold', y=1.03)
    return fig


def fig_decomposition(dec):
    fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(dec)); w = .36
    ax.bar(x - w / 2, dec.composition_effect * 100, w, color='#6a4c93', label='Composition effect (share of retakers)')
    ax.bar(x + w / 2, dec.rate_effect * 100, w, color='#e9a03b', label='Rate effect (pass rate within groups)')
    ax.plot(x, dec.total_change * 100, 'k D', label='Total change')
    ax.axhline(0, color='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{P.ROUND_LABEL[a]} > {P.ROUND_LABEL[b]}' for a, b in zip(dec.from_exam, dec.to_exam)])
    ax.set_ylabel('Change in overall pass rate (percentage points)')
    ax.set_title('Why the overall pass rate moved between rounds'); ax.legend(fontsize=8)
    return fig


def fig_prior_fails(pooled, by_round):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2)); cats = ['0', '1', '2', '3+']
    g = pooled.set_index('prior_failures').reindex(cats); x = np.arange(4)
    err = np.vstack([g.rate - g.ci_low, g.ci_high - g.rate])
    axes[0].bar(x, g.rate, yerr=err, capsize=4, color=['#2a9d8f', '#e9a03b', '#e45756', '#9b2d30'])
    for xi, v, n in zip(x, g.rate, g.rows): axes[0].text(xi, v + .035, f'{v:.0%}\n(n={n:,})', ha='center', fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels(cats); P.pct_axis(axes[0]); axes[0].set_ylim(0, .55)
    axes[0].set_xlabel('Number of previous failures'); axes[0].set_ylabel('Chance of passing at this attempt')
    axes[0].set_title('Candidates with earlier failures pass far less often (pooled)')
    for e in C.EXAMS:
        d = by_round[by_round.exam == e].set_index('prior_failures').reindex(cats)
        if d.pass_rate.notna().sum() >= 2:
            axes[1].plot(x, d.pass_rate, 'o-', lw=2, color=P.ROUND_COLORS[e], label=P.ROUND_LABEL[e])
    axes[1].set_xticks(x); axes[1].set_xticklabels(cats); P.pct_axis(axes[1]); axes[1].set_xlabel('Number of previous failures')
    axes[1].set_title('Inside each round the chance mostly keeps falling'); axes[1].legend(title='Round', fontsize=8)
    return fig
