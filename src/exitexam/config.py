"""Single place for every rule that defines the analysis. Nothing is hard-coded elsewhere."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
OUT_DIR = os.path.join(ROOT, 'outputs')
TAB_DIR = os.path.join(OUT_DIR, 'tables')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
CLEAN_DIR = os.path.join(OUT_DIR, 'clean')

# The five exam rounds in strict chronological order. The label is the ONLY key that identifies a round.
EXAMS = ['2023', '2024', '2025', '2026-R1', '2026-R2']
EXAM_INDEX = {e: i for i, e in enumerate(EXAMS)}

# Part A = rounds whose files say which institution (public / private / abroad) a candidate belongs to.
PART_A_EXAMS = ['2025', '2026-R1', '2026-R2']
PART_B_EXAMS = EXAMS

# Column positions per file (files have different layouts; we never rely on header text).
SCHEMAS = {
    '2023': dict(file='ama-afghanistan-exit-2023-1.xlsx', n_cols=9,
                 cols=dict(cand_no=0, name=1, father=2, grandfather=3, field=4, grad_year=5,
                           score=6, result=7, province_reported=8)),
    '2024': dict(file='ama-afghanistan-exit-2024-1.xlsx', n_cols=9,
                 cols=dict(cand_no=0, name=1, father=2, grandfather=3, field=4, gender=5,
                           score=6, result=7, province_reported=8)),
    '2025': dict(file='ama-afghanistan-exit-2025-1.xlsx', n_cols=7,
                 cols=dict(cand_no=0, name=1, father=2, grandfather=3, program=4, score=5, result=6)),
    '2026-R1': dict(file='ama-afghanistan-exit-2026-1.xlsx', n_cols=7,
                    cols=dict(cand_no=0, name=1, father=2, grandfather=3, program=4, score=5, result=6)),
    '2026-R2': dict(file='ama-afghanistan-exit-2026-2.xlsx', n_cols=7,
                    cols=dict(cand_no=0, name=1, father=2, grandfather=3, program=4, score=5, result=6)),
}

CUTOFF = 216.0            # pass mark observed in every round (verified in the audit)
NEAR_MISS = 10.0          # "near miss" = failed by at most this many points
MIN_INSTITUTE_N = 50      # min first-attempt candidates (Part A rounds pooled) to rank an institute
SEED = 20260101           # every random procedure uses this seed
N_BOOT = 1000             # bootstrap replicates (person-level cluster bootstrap)
HASH_SALT = os.environ.get('EXIT_EXAM_SALT', 'change-me-before-publishing')
