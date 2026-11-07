"""Tracking candidates across rounds: history features, linkage validation and a fuzzy-link sensitivity check."""
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from . import config as C
from .cleaning import hash_id


def add_history(c):
    """Sort strictly by round and add per-person history features. A person appears once per round."""
    c = c.sort_values(['person_id', 'exam_idx']).copy()
    assert not c.duplicated(['person_id', 'exam']).any(), 'a person appears twice in one round'
    g = c.groupby('person_id', sort=False)
    c['attempt'] = g.cumcount() + 1
    c['prior_fails'] = g['passed'].transform(lambda s: (~s).cumsum().shift(fill_value=0))
    c['prior_passes'] = g['passed'].transform(lambda s: s.cumsum().shift(fill_value=0))
    c['last_score'] = g['score'].shift()
    c['last_gap'] = g['gap'].shift()
    c['last_exam_idx'] = g['exam_idx'].shift()
    c['last_institute'] = g['institute'].shift()
    c['entry_exam'] = g['exam'].transform('first')
    c['returning'] = c['attempt'] > 1
    return c.sort_values(['exam_idx', 'person_id']).reset_index(drop=True)


def person_table(h):
    g = h.sort_values('exam_idx').groupby('person_id')
    p = g.agg(n_attempts=('exam', 'size'), entry_exam=('exam', 'first'), last_exam=('exam', 'last'),
              first_score=('score', 'first'), last_score=('score', 'last'), best_score=('score', 'max'),
              ever_passed=('passed', 'max'), field=('field', 'first'))
    fp = h[h.passed].groupby('person_id').attempt.min()
    p['attempts_to_pass'] = fp.reindex(p.index)
    p['rounds'] = g['exam'].agg(lambda s: ','.join(s))
    return p.reset_index()


def linkage_validity(h, seed=C.SEED):
    """Evidence that linked records are the same person (not two people sharing three names)."""
    rng = np.random.default_rng(seed)
    out = {}
    # 1) field agreement: a real person keeps the same field of study
    raw = h.copy()
    out['linked_persons'] = int((h.groupby('person_id').size() > 1).sum())
    # 2) score persistence of consecutive attempts vs a shuffled (false-link) baseline
    pairs = h.dropna(subset=['last_score'])
    r_real = pairs[['score', 'last_score']].corr(method='spearman').iloc[0, 1]
    shuffled = []
    for e, grp in pairs.groupby('exam'):
        s = grp.score.to_numpy().copy(); rng.shuffle(s)
        shuffled.append(pd.DataFrame({'score': s, 'last_score': grp.last_score.to_numpy()}))
    r_null = pd.concat(shuffled).corr(method='spearman').iloc[0, 1]
    out.update(pairs=len(pairs), spearman_linked=r_real, spearman_shuffled_null=r_null)
    # 3) institution agreement (Part A rounds only): a graduate keeps the same institution
    a = pairs[pairs.institute.notna() & (pairs.institute != 'unknown') & pairs.last_institute.notna()
              & (pairs.last_institute != 'unknown')]
    if len(a):
        out['institute_agreement'] = float((a.institute == a.last_institute).mean())
        out['institute_pairs'] = len(a)
    return out


def field_agreement_by_key(clean):
    """Using the NAME triple only (ignoring field): how often do matched records keep the same field?"""
    c = clean[~clean.ambiguous & (clean.field != 'unknown')]
    g = c.groupby('person_key').field.agg(['nunique', 'size'])
    multi = g[g['size'] > 1]
    base = c.field.value_counts(normalize=True)
    expected_same = float((base ** 2).sum())
    return dict(keys_matched=len(multi), keys_with_conflicting_field=int((multi['nunique'] > 1).sum()),
                observed_agreement=float((multi['nunique'] == 1).mean()), expected_if_random=expected_same)


def fuzzy_person_ids(clean, threshold=0.80):
    """Sensitivity analysis: also join records whose name AND father name match exactly, field matches, and the
    grandfather name is similar (typo tolerant). Returns a new person_id column and the number of extra links."""
    c = clean[~clean.ambiguous].copy()
    c['nf'] = c.person_key.str.rsplit('|', n=1).str[0]
    c['gf'] = c.person_key.str.rsplit('|', n=1).str[1]
    parent = {p: p for p in c.person_id.unique()}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    extra = 0
    for (nf, fld), g in c.groupby(['nf', 'field']):
        ids = g.drop_duplicates('person_id')
        if len(ids) < 2:
            continue
        rows = ids[['person_id', 'gf']].to_records(index=False)
        ex = {pid: set(g.loc[g.person_id == pid, 'exam']) for pid, _ in rows}
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (pa, ga), (pb, gb) = rows[i], rows[j]
                if ex[pa] & ex[pb]:
                    continue                       # same round twice -> two different people
                if ga != gb and SequenceMatcher(None, ga, gb).ratio() >= threshold:
                    ra, rb = find(pa), find(pb)
                    if ra != rb:
                        parent[ra] = rb; extra += 1
    mapping = {p: find(p) for p in parent}
    out = clean.copy()
    out['person_id_fuzzy'] = out.person_id.map(mapping).fillna(out.person_id)
    return out, extra
