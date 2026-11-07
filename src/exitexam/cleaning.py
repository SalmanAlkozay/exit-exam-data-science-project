"""Loading, harmonising and cleaning the five raw files.

Rules (decided with the researcher):
  * rows with an EMPTY score or a ZERO score are removed (absent or cheating)
  * rows whose result is annulled (cheating / biometric / quarantine) are removed
  * rows with an unreadable result are removed
Each row gets exactly ONE exclusion reason; counts are written to an audit table.
"""
import hashlib, hmac, re, unicodedata
import numpy as np
import pandas as pd
from . import config as C

_MAP = str.maketrans({
    'ي': 'ی', 'ې': 'ی', 'ى': 'ی', 'ۍ': 'ی', 'ئ': 'ی',
    'ك': 'ک', 'ڭ': 'ګ', 'گ': 'ګ',
    'ە': 'ه', 'ۀ': 'ه', 'ہ': 'ه', 'ة': 'ه',
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',
    'ء': '', 'ـ': '', '\u200c': ' ', '\u200d': '', '\u200e': '', '\u200f': '',
})


def norm_text(s):
    """Unify Arabic/Persian/Pashto letter variants, drop diacritics, collapse spaces."""
    if pd.isna(s):
        return ''
    s = unicodedata.normalize('NFKC', str(s)).translate(_MAP)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    return re.sub(r'\s+', ' ', s).strip()


def name_key(s):
    """Name used for linking: normalised AND without any space (files disagree on spacing)."""
    return norm_text(s).replace(' ', '')


# ---------------------------------------------------------------- results
_PASS = {norm_text('بریالی'), norm_text('کامیاب')}
_FAIL = {norm_text('بې نتیجه'), norm_text('بی نتیجه')}


def classify_result(x):
    t = norm_text(x)
    if t in _PASS:
        return 'pass'
    if t in _FAIL:
        return 'fail'
    if 'باطله' in t or 'قرنطین' in t:
        return 'void'
    return 'unknown'


# ---------------------------------------------------------------- program / field
def parse_field(text):
    t = norm_text(text)
    if 'ستوماتولوژی' in t:
        return 'stomatology'
    if 'معالجوی' in t:
        return 'medicine'
    return 'unknown'


_TYPE_TOKENS = {norm_text('خصوصي'): 'private', norm_text('امارتي'): 'public', norm_text('خارج مرز'): 'abroad'}
_PROVINCES = [norm_text(p) for p in [
    'کابل', 'بلخ', 'هرات', 'کندهار', 'ننګرهار', 'هلمند', 'خوست', 'غزنی', 'پروان', 'بغلان',
    'جوزجان', 'پکتیا', 'کندز', 'بدخشان', 'تخار', 'فاریاب', 'سمنګان', 'سرپل', 'بامیان', 'لغمان']]
_PUBLIC_PROVINCE = {norm_text('البیرونی'): norm_text('کاپیسا')}   # Al-Biruni University is in Kapisa
_COUNTRIES = [norm_text(c) for c in [
    'پاکستان', 'چین', 'روسیه', 'ایران', 'ترکیه', 'قزاقستان', 'تاجیکستان', 'قرغزستان',
    'هندوستان', 'هند', 'مالیزیا', 'سودان', 'مصر']]
_FIELD_RE = re.compile(r'\(?\s*(' + re.escape(norm_text('معالجوی طب')) + '|' +
                       re.escape(norm_text('ستوماتولوژی')) + r')\s*\)?')


def parse_program(p):
    """Split a messy 'program' string into type / province / institute / field / country."""
    t = norm_text(p)
    out = dict(inst_type='unknown', province='unknown', institute='unknown', field='unknown', country=None)
    if t in ('', '-'):
        return out
    out['field'] = parse_field(t)
    t2 = _FIELD_RE.sub(' ', t)
    t2 = t2.replace(norm_text('د هیواد بهر'), ' ')
    t2 = re.sub(r'\s*/\s*|\s+-\s+', ' | ', t2)      # '/' and ' - ' are separators; an unspaced hyphen is not
    tokens = [x.strip(' -|') for x in t2.split('|')]
    tokens = [x for x in tokens if x]
    inst_type = next((_TYPE_TOKENS[x] for x in tokens if x in _TYPE_TOKENS), 'unknown')
    tokens = [x for x in tokens if x not in _TYPE_TOKENS]
    out['inst_type'] = inst_type
    if inst_type == 'private' and tokens:
        out['province'] = tokens[0]
        out['institute'] = tokens[1] if len(tokens) > 1 else 'unknown'
    elif inst_type == 'public' and tokens:
        name = tokens[0]
        out['institute'] = name
        out['province'] = _PUBLIC_PROVINCE.get(name) or next((pv for pv in _PROVINCES if pv in name), 'unknown')
    elif inst_type == 'abroad':
        out['province'] = 'abroad'
        out['institute'] = ' - '.join(tokens) if tokens else 'unknown'
        out['country'] = next((c for c in _COUNTRIES if c in t), 'unknown')
    return out


# ---------------------------------------------------------------- loading
def load_raw():
    """Read all five files into one long table with a strict `exam` label."""
    frames = []
    for exam in C.EXAMS:
        sch = C.SCHEMAS[exam]
        raw = pd.read_excel(f"{C.RAW_DIR}/{sch['file']}", sheet_name=0, dtype=object)
        assert raw.shape[1] == sch['n_cols'], f"{sch['file']}: expected {sch['n_cols']} columns, got {raw.shape[1]}"
        d = pd.DataFrame({k: raw.iloc[:, i] for k, i in sch['cols'].items()})
        d['exam'] = exam
        d['excel_row'] = np.arange(len(d)) + 2
        frames.append(d)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out['exam_idx'] = out['exam'].map(C.EXAM_INDEX)
    out['score'] = pd.to_numeric(out['score'], errors='coerce')
    return out


def hash_id(key):
    return hmac.new(C.HASH_SALT.encode(), key.encode('utf-8'), hashlib.sha256).hexdigest()[:12]


def build_clean():
    """Return (clean_rows, excluded_rows, audit_table)."""
    d = load_raw()
    d['status'] = d['result'].map(classify_result)

    # --- institution / field information
    in_a = d['exam'].isin(C.PART_A_EXAMS)
    prog = d['program'].fillna('').astype(str)
    uniq = pd.Series(prog.unique())
    parsed = pd.DataFrame([parse_program(p) for p in uniq], index=uniq)
    parsed_rows = parsed.reindex(prog.values).reset_index(drop=True)
    for col in ['inst_type', 'province', 'institute', 'country']:
        d[col] = np.where(in_a, parsed_rows[col], None)
    d['field'] = np.where(in_a, parsed_rows['field'], d['field'].map(parse_field))
    # 2023 / 2024: only the field is known (province column is always Kabul -> not informative)
    d.loc[~in_a, ['inst_type', 'province', 'institute']] = 'unknown'

    # --- exclusion rule: ONE reason per row, evaluated in this order
    d['exclusion_reason'] = np.select(
        [d.score.isna(), d.score == 0, d.status == 'void', d.status == 'unknown'],
        ['score_missing', 'score_zero', 'result_annulled', 'result_unreadable'], default='kept')

    # --- person key used for linking rounds (names only; IDs differ between rounds)
    d['person_key'] = (d['name'].map(name_key) + '|' + d['father'].map(name_key) + '|' +
                       d['grandfather'].map(name_key))
    d['key_incomplete'] = (d['name'].map(name_key) == '') | (d['father'].map(name_key) == '') | \
                          (d['grandfather'].map(name_key) == '')

    excluded = d[d.exclusion_reason != 'kept'].copy()
    clean = d[d.exclusion_reason == 'kept'].copy()

    # --- consistency check that must hold for the analysis to be valid
    clean['passed'] = clean['status'] == 'pass'
    bad = clean[(clean.passed & (clean.score < C.CUTOFF)) | (~clean.passed & (clean.score >= C.CUTOFF))]
    assert bad.empty, f'{len(bad)} rows contradict the cut-off rule'
    clean['gap'] = clean['score'] - C.CUTOFF

    # --- ambiguous names: same triple twice in the same round -> cannot be linked safely
    clean['ambiguous'] = clean.duplicated(['exam', 'person_key', 'field'], keep=False) | clean.key_incomplete
    clean = clean.reset_index(drop=True)
    clean['person_id'] = [f'amb-{i}' if a else hash_id(k + '|' + f)
                          for i, (k, f, a) in enumerate(zip(clean.person_key, clean.field, clean.ambiguous))]

    # --- audit table (rows kept / removed per round)
    audit = d.groupby('exam')['exclusion_reason'].value_counts().unstack(fill_value=0).reindex(C.EXAMS)
    for c in ['score_missing', 'score_zero', 'result_annulled', 'result_unreadable', 'kept']:
        if c not in audit:
            audit[c] = 0
    audit.insert(0, 'rows_raw', d.groupby('exam').size().reindex(C.EXAMS))
    audit['ambiguous_names_kept'] = clean.groupby('exam')['ambiguous'].sum().reindex(C.EXAMS)
    audit = audit[['rows_raw', 'score_missing', 'score_zero', 'result_annulled', 'result_unreadable',
                   'kept', 'ambiguous_names_kept']]
    assert (audit.rows_raw == audit[['score_missing', 'score_zero', 'result_annulled',
                                      'result_unreadable', 'kept']].sum(axis=1)).all()
    keep_cols = ['exam', 'exam_idx', 'person_id', 'person_key', 'field', 'score', 'gap', 'passed', 'status',
                 'inst_type', 'province', 'institute', 'country', 'gender', 'grad_year', 'ambiguous',
                 'cand_no', 'excel_row']
    return clean[[c for c in keep_cols if c in clean]], excluded, audit
