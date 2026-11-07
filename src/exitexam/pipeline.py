"""Runs the whole analysis (used by run_all.py and by the notebooks)."""
import os
import numpy as np
import pandas as pd
from . import config as C, cleaning as cl, linking as lk, plotting as P, analysis_a as A, analysis_b as B, stats_utils as su


def load_all():
    clean, excluded, audit = cl.build_clean()
    h = lk.add_history(clean)
    return clean, excluded, audit, h


def export_clean(h):
    """Anonymised analysis file: no names, no candidate numbers. person_id is a keyed hash."""
    cols = ['exam', 'person_id', 'field', 'score', 'gap', 'passed', 'inst_type', 'province', 'institute', 'country',
            'grad_year', 'attempt', 'prior_fails', 'prior_passes', 'returning', 'entry_exam']
    os.makedirs(C.CLEAN_DIR, exist_ok=True)
    out = h[[c for c in cols if c in h]].copy()
    path = os.path.join(C.CLEAN_DIR, 'clean_rounds_anonymized.csv')
    out.to_csv(path, index=False, encoding='utf-8-sig')
    return path


def write_workbook(path, sheets):
    """sheets: {sheet_name: (DataFrame, description)} -> one xlsx with an INDEX sheet."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        idx = pd.DataFrame([(n, d, len(df)) for n, (df, d) in sheets.items()], columns=['sheet', 'what it contains', 'rows'])
        idx.to_excel(xw, sheet_name='INDEX', index=False)
        for n, (df, _) in sheets.items():
            df = df.copy()
            for c in df.select_dtypes('float').columns:
                df[c] = df[c].round(4)
            df.to_excel(xw, sheet_name=n[:31], index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = 'A2'
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)


# ------------------------------------------------------------------------------------------- Part B
def run_part_b(clean, h, audit, show=False):
    R = {}
    es = B.exam_summary(h); R['exam_summary'] = es
    R['exam_pairwise'] = B.exam_pairwise(h)
    R['field_by_round'] = B.field_by_round(h)
    pt, dist, tests = B.participation(h); R.update(persons=pt, attempts_dist=dist, attempts_tests=tests)
    R['decomposition'] = dec = B.decomposition(h)
    tr, F = B.transitions(h); R['transitions'] = tr
    per, pooled = B.gap_bands(F); R['gap_by_transition'] = per; R['gap_pooled'] = pooled
    R['prior_fails_pooled'], R['prior_fails_by_round'] = B.prior_fails_table(h)
    models, _, _ = B.hazard_models(h); R['models'] = models; R['models_table'] = B.models_to_table(models)
    R['retest'] = B.retest(h)
    cum, coh = B.cohorts(h); R['cohort_curve'] = cum; R['cohort_table'] = coh
    R['cohort_2023_attempts'] = B.attempts_for_2023_cohort(h)
    R['linkage_validity'] = pd.DataFrame([{**lk.linkage_validity(h), **{f'name_key_{k}': v for k, v in lk.field_agreement_by_key(clean).items()}}])
    R['linkage_sensitivity'] = B.linkage_sensitivity(clean, h, models); R['extra_fuzzy_links'] = R['linkage_sensitivity'].attrs['extra_links']
    R['supplementary'] = B.supplementary(h)
    R['audit'] = audit.reset_index()
    figs = [('B01_overview', B.fig_overview(es)), ('B02_score_distribution', B.fig_distribution(h)),
            ('B03_composition_new_vs_retaking', B.fig_composition(es)), ('B04_after_failure', B.fig_transitions(F)),
            ('B05_distance_to_pass_mark', B.fig_gap(per)), ('B06_previous_failures', B.fig_prior_fails(R['prior_fails_pooled'], R['prior_fails_by_round'])),
            ('B07_odds_ratios_all_candidates', B.fig_forest({k: models[k] for k in list(models)[1:2]}, 'Odds of passing: round, field and previous failures (model M2)')),
            ('B08_odds_ratios_returners', B.fig_forest({k: models[k] for k in list(models)[2:]}, 'Odds of passing among candidates retaking (model M3)')),
            ('B09_entry_cohorts', B.fig_cohorts(cum)), ('B10_number_of_attempts', B.fig_attempts(pt, dist)),
            ('B11_score_persistence', B.fig_retest(h)), ('B12_decomposition', B.fig_decomposition(dec))]
    for name, fig in figs:
        P.save(fig, name, show=show)
    return R


def partb_workbook(R):
    S = {
        'audit': (R['audit'], 'Rows kept / removed in each round and why (empty score, zero score, annulled result)'),
        'exam_summary': (R['exam_summary'], 'Per round: pass rate (Wilson CI), scores, near-miss share, share retaking, new vs retaking pass rates'),
        'exam_pairwise': (R['exam_pairwise'], 'Change in pass rate between consecutive rounds (two-proportion test, Holm-adjusted p)'),
        'field_by_round': (R['field_by_round'], 'Medicine vs stomatology per round (2026 R2 has no stomatology candidates)'),
        'attempts_dist': (R['attempts_dist'], 'Candidates by total number of attempts (1-5) with ever-passed share and first-attempt score'),
        'attempts_tests': (R['attempts_tests'], 'Tests: first-attempt score vs number of attempts'),
        'cohort_2023_attempts': (R['cohort_2023_attempts'], 'Attempts distribution for the 2023 entry cohort (longest follow-up)'),
        'decomposition': (R['decomposition'], 'Kitagawa decomposition of pass-rate change: composition (share retaking) vs rate effect'),
        'transitions': (R['transitions'], 'After failing round t: returned in t+1? passed? (Wilson CI; pooled row uses person-cluster bootstrap)'),
        'gap_by_transition': (R['gap_by_transition'], 'Return and success probability by distance below the pass mark, per transition'),
        'gap_pooled': (R['gap_pooled'], 'Same, pooled over transitions (cluster bootstrap CI)'),
        'prior_fails_pooled': (R['prior_fails_pooled'], 'Pass probability by number of previous failures (candidates not yet passed)'),
        'prior_fails_by_round': (R['prior_fails_by_round'], 'Same, inside each round'),
        'models': (R['models_table'], 'Logistic models M1-M3 (odds ratios, cluster-robust CI by person)'),
        'retest': (R['retest'], 'Score persistence between consecutive rounds: correlation, slope, mean change'),
        'cohort_curve': (R['cohort_curve'], 'Cumulative share of each entry cohort that has passed by each round'),
        'cohort_table': (R['cohort_table'], 'Of those who failed their first attempt: came back? eventually passed?'),
        'linkage_validity': (R['linkage_validity'], 'Evidence that linked records are the same person'),
        'linkage_sensitivity': (R['linkage_sensitivity'], 'Main results under exact vs typo-tolerant linking'),
    }
    for k, v in R['supplementary'].items():
        S[k] = (v, 'Supplementary descriptive table')
    return S


# ------------------------------------------------------------------------------------------- Part A
def run_part_a(clean, h, audit, show=False):
    R = {}
    a = A.prepare(h); R['a'] = a
    R['coverage'] = A.coverage(h, a)
    ts, tp = A.type_summary(a); R['type_summary'], R['type_pooled'] = ts, tp
    R['type_tests'] = A.type_tests(a); R['odds_by_round'] = A.odds_by_round(a)
    R['burden'] = bd = A.burden(a); R['field_table'] = A.field_table(a)
    meta, hetero, var, base, first = A.institute_tables(a)
    R.update(institutes=meta, hetero=hetero, variance=var)
    R['province'] = pt = A.province_table(a)
    models, d, X2 = A.models(a); R['models'] = models; R['models_table'] = B.models_to_table(models)
    R['grid'] = grid = A.predicted_grid(models['MA2 + previous failures'], X2, d)
    R['type_transitions'] = tt = A.type_transitions(h, a)
    rate_m, n_m = A.institute_round_matrix(a)
    R['institute_round'] = (rate_m.stack().rename('pass_rate').reset_index()
                            .merge(n_m.stack().rename('candidates').reset_index(), on=['label', 'exam'])
                            .sort_values(['label', 'exam']).reset_index(drop=True))
    figs = [('A01_pass_rate_by_type', A.fig_type_by_round(ts)), ('A02_scores_by_type', A.fig_type_ecdf(a)),
            ('A03_institutes_first_attempt', A.fig_institutes(meta)), ('A04_institutes_funnel', A.fig_funnel(meta, first)),
            ('A05_institute_by_round', A.fig_heatmap(rate_m, n_m)), ('A06_province', A.fig_province(pt)),
            ('A07_model_odds_ratios', B.fig_forest({k: models[k] for k in list(models)[1:]}, 'Odds of passing, rounds 2025-2026 (Part A models)', terms=['type', 'field_c', 'prior', 'round'])),
            ('A08_predicted_probabilities', A.fig_predicted(grid)), ('A09_share_of_failures', A.fig_burden(bd)),
            ('A10_after_failure_by_type', A.fig_type_transitions(tt))]
    for name, fig in figs:
        P.save(fig, name, show=show)
    return R


def parta_workbook(R, audit):
    ar = audit.reset_index().loc[lambda d: d.exam.isin(C.PART_A_EXAMS)]
    return {
        'audit': (ar, 'Rows kept / removed in the three Part A rounds'),
        'coverage': (R['coverage'], 'Candidates per institute type and round; rows without institution info'),
        'type_summary': (R['type_summary'], 'Pass rate by institute type and round (Wilson CI), first attempt vs retaking'),
        'type_pooled': (R['type_pooled'], 'Pass rate by type, rounds pooled (person-cluster bootstrap CI)'),
        'type_tests': (R['type_tests'], 'Public vs private: difference, odds ratio, chi-square, Cramer V, Cliff delta'),
        'odds_by_round': (R['odds_by_round'], 'Public-vs-private odds ratio inside each round (all candidates and first attempts)'),
        'share_of_failures': (R['burden'], 'Share of candidates vs share of failures by type'),
        'field_table': (R['field_table'], 'Medicine vs stomatology by type and round'),
        'institutes': (R['institutes'], 'Institute table: first-attempt rate, expected vs observed (type/round/field adjusted), FDR-adjusted flags, shrunk rate'),
        'institute_heterogeneity': (R['hetero'], 'Do institutes differ more than chance allows? (chi-square within type, SD of true rates)'),
        'variance_explained': (R['variance'], 'Share of score variance explained by type, province, institute, round, field'),
        'province': (R['province'], 'First-attempt pass rate by province, private and public'),
        'institute_round': (R['institute_round'], 'Pass rate and number of candidates for every institute and round (nothing hidden)'),
        'models': (R['models_table'], 'Logistic models MA1-MA3 (odds ratios, cluster-robust CI by person)'),
        'predicted_probabilities': (R['grid'], 'Model-based pass probabilities for typical candidate profiles'),
        'type_transitions': (R['type_transitions'], 'After failing: came back / passed, by institute type'),
    }


# ------------------------------------------------------------------------------------------- key numbers
def key_numbers(RA, RB, audit, h):
    pc = lambda v, d=1: f'{100 * v:.{d}f}%'
    es = RB['exam_summary'].set_index('exam')
    L = ['# Key numbers (auto-generated by run_all.py; every figure below comes from outputs/tables)', '']
    L += ['## Data', '']
    for e in C.EXAMS:
        r = audit.loc[e]
        L.append(f'- {e}: {int(r.rows_raw):,} rows in the file; removed {int(r.score_missing)} empty-score, {int(r.score_zero)} zero-score, '
                 f'{int(r.result_annulled) + int(r.result_unreadable)} other; kept {int(r.kept):,}.')
    pt = RB['persons']; multi = (pt.n_attempts > 1).mean()
    L += [f'- Valid candidate-attempts: {len(h):,}; distinct candidates after linking: {len(pt):,}; '
          f'{pc(multi)} appear in two or more rounds; maximum {int(pt.n_attempts.max())} rounds.', f'- Pass mark: {C.CUTOFF:.0f} in every round (no pass below it, no fail at or above it).', '']
    lv = RB['linkage_validity'].iloc[0]
    L += ['## Linking quality', '',
          f'- Matched records keep the same field of study in {pc(lv.name_key_observed_agreement, 2)} of cases (random linking would give about {pc(lv.name_key_expected_if_random, 0)}).',
          f'- In Part A rounds linked records keep the same institute in {pc(lv.institute_agreement, 2)} of {int(lv.institute_pairs):,} pairs.',
          f'- Rank correlation of consecutive scores: {lv.spearman_linked:.2f} for linked pairs vs {lv.spearman_shuffled_null:.2f} after shuffling (false-link baseline).',
          f'- A typo-tolerant rule adds only {RB["extra_fuzzy_links"]} extra links and leaves conclusions unchanged (see linkage_sensitivity).', '']
    L += ['## Part B - five linked rounds', '', '### Pass rate by round', '']
    for e in C.EXAMS:
        r = es.loc[e]
        extra = f'; first attempt {pc(r.pass_rate_new)}' + (f', retaking {pc(r.pass_rate_returning)} ({pc(r.pct_returning, 0)} of candidates retaking)' if r.n_returning else '')
        L.append(f'- {e}: {pc(r.pass_rate)} (95% CI {pc(r.ci_low)} to {pc(r.ci_high)}), n={int(r.rows):,}, median candidate who failed was {r.median_points_below_cutoff_among_fails:.0f} points below the pass mark{extra}.')
    tr = RB['transitions'].iloc[-1]
    L += ['', '### After a failure', '',
          f'- Pooled over all consecutive rounds, {pc(tr.return_rate)} (CI {pc(tr.ret_low)} to {pc(tr.ret_high)}) of failed candidates came back in the next round, and {pc(tr.pass_given_return)} '
          f'(CI {pc(tr.pass_low)} to {pc(tr.pass_high)}) of those who came back passed.']
    for _, r in RB['transitions'].iloc[:-1].iterrows():
        L.append(f'- {r.transition}: {pc(r.return_rate)} came back; {pc(r.pass_given_return)} of them passed; mean score change {r.mean_score_change:+.1f}.')
    pf = RB['prior_fails_pooled'].set_index('prior_failures')
    L.append('- Chance of passing by number of previous failures (0/1/2/3+): ' + ' / '.join(pc(pf.loc[k, 'rate']) for k in ['0', '1', '2', '3+']) + '.')
    m2 = RB['models']['M2 + number of previous failures']
    L.append('- Odds ratios of passing (M2, adjusted for round and field; cluster-robust CI): ' + '; '.join(
        f'{k}: {m2.loc[t, "odds_ratio"]:.2f} ({m2.loc[t, "or_low"]:.2f}-{m2.loc[t, "or_high"]:.2f})' for k, t in
        [('1 previous failure', 'prior_fails_cat[1]'), ('2', 'prior_fails_cat[2]'), ('3+', 'prior_fails_cat[3+]'), ('2026 R1 vs 2025', 'round[2026-R1]')]) + '.')
    rt = RB['retest']
    L.append(f'- Scores persist between consecutive rounds: Pearson r from {rt.pearson_r.min():.2f} to {rt.pearson_r.max():.2f}; '
             f'mean change from {rt.mean_change.min():+.1f} to {rt.mean_change.max():+.1f} points depending on the round pair.')
    dec = RB['decomposition']
    for _, r in dec.iterrows():
        L.append(f'- {r.from_exam} to {r.to_exam}: pass rate changed {100 * r.total_change:+.1f} points '
                 f'(composition {100 * r.composition_effect:+.1f}, rate {100 * r.rate_effect:+.1f}).')
    ct = RB['cohort_table'].set_index('entry_round')
    L += ['', '### Persistence and heavy re-sitting', '']
    for e in ct.index:
        r = ct.loc[e]
        L.append(f'- Failed first attempt in {e} ({int(r.failed_first_attempt):,} candidates): {pc(r.came_back_share, 0)} came back, {pc(r.eventually_passed_share)} eventually passed by 2026 R2.')
    ad = RB['attempts_dist']
    L.append('- Share who passed at some point, by total attempts 1/2/3/4/5: ' + ' / '.join(pc(v, 0) for v in ad.ever_passed) + ' (survivorship: passers stop sitting).')
    L.append('- Median first-attempt score by total attempts 1/2/3/4/5: ' + ' / '.join(f'{v:.0f}' for v in ad.median_first_score) + '.')
    L += ['', '## Part A - three rounds with institution information (2025, 2026 R1, 2026 R2)', '']
    tp = RA['type_pooled'].set_index('type'); tt = RA['type_tests'].set_index('scope').loc['pooled']
    for t in ['Public', 'Private', 'Abroad']:
        r = tp.loc[t]; L.append(f'- {t}: pass rate {pc(r.rate)} (CI {pc(r.ci_low)} to {pc(r.ci_high)}), {int(r.rows):,} attempts by {int(r.persons):,} candidates.')
    L.append(f'- Public vs private odds ratio {tt.odds_ratio_public_vs_private:.1f} (CI {tt.or_low:.1f}-{tt.or_high:.1f}), Cramer V {tt.cramers_v:.2f}, Cliff delta {tt.cliffs_delta_public_vs_private:.2f}.')
    bd = RA['burden'].query("scope == 'pooled' and type == 'Private'").iloc[0]
    L.append(f'- Private-institute graduates are {pc(bd.share_of_candidates, 0)} of candidates and {pc(bd.share_of_failures, 0)} of all failures.')
    obr = RA['odds_by_round']; f1 = obr[obr.scope == 'first attempts']
    L.append('- Public-vs-private odds ratio among FIRST attempts by round (2025/2026 R1/2026 R2): ' + ' / '.join(f'{v:.1f}' for v in f1.or_public_vs_private) + '.')
    mA = RA['models']['MA3 first attempts only']
    L.append(f'- First attempts only, adjusted for round and field: private vs public OR {mA.loc["type[Private]", "odds_ratio"]:.3f} '
             f'(CI {mA.loc["type[Private]", "or_low"]:.3f}-{mA.loc["type[Private]", "or_high"]:.3f}); round 2026 R1 vs 2025 OR {mA.loc["round[2026-R1]", "odds_ratio"]:.2f}.')
    inst = RA['institutes']; rk = inst[inst.ranked]
    L.append(f'- Of {len(rk)} institutes with enough candidates, {(rk.flag == "better than expected").sum()} did better and {(rk.flag == "worse than expected").sum()} did worse than their type, round and field predict (FDR 5%).')
    hs = RA['hetero'].set_index('type')
    L.append('- Between-institute spread inside private institutes: SD of true pass rates ' + f'{100 * hs.loc["Private", "sd_of_true_rates_between_institutes"]:.0f} percentage points (chi-square p={hs.loc["Private", "p_value"]:.1e}).')
    vv = RA['variance'].set_index('grouping').omega_squared_adjusted
    L.append('- Share of first-attempt score variance explained (adjusted omega-squared): ' + ', '.join(f'{k} {pc(vv[k], 0)}' for k in vv.index) + '.')
    g = RA['grid']
    def gp(t, hi, r): return g[(g.type == t) & (g.history == hi) & (g['round'] == r)].predicted_pass.iloc[0]
    L.append(f'- Model-predicted pass chance (medicine): private first attempt {pc(gp("Private", "First attempt", "2025"), 0)} (2025), {pc(gp("Private", "First attempt", "2026-R1"), 0)} (2026 R1), '
             f'{pc(gp("Private", "First attempt", "2026-R2"), 0)} (2026 R2); private after one failure {pc(gp("Private", "Failed once before", "2026-R1"), 0)} in 2026 R1; '
             f'public first attempt {pc(gp("Public", "First attempt", "2026-R1"), 0)} in 2026 R1.')
    return '\n'.join(L) + '\n'


def run_all(show=False):
    P.setup()
    clean, excluded, audit, h = load_all()
    print('Cleaning done:', len(clean), 'valid rows')
    export_clean(h)
    RB = run_part_b(clean, h, audit, show=show); print('Part B done')
    RA = run_part_a(clean, h, audit, show=show); print('Part A done')
    write_workbook(os.path.join(C.TAB_DIR, 'partB_five_rounds_tables.xlsx'), partb_workbook(RB))
    write_workbook(os.path.join(C.TAB_DIR, 'partA_three_rounds_tables.xlsx'), parta_workbook(RA, audit))
    kn = key_numbers(RA, RB, audit, h)
    with open(os.path.join(C.OUT_DIR, 'key_numbers.md'), 'w', encoding='utf-8') as f:
        f.write(kn)
    return dict(clean=clean, h=h, audit=audit, RA=RA, RB=RB, key_numbers=kn)
