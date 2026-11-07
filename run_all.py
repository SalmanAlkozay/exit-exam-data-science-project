"""Run the complete analysis: python run_all.py  (about 1-2 minutes). Writes everything into outputs/."""
import os, sys
import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from exitexam import pipeline, config as C

if __name__ == '__main__':
    if C.HASH_SALT == 'change-me-before-publishing':
        print('NOTE: set the environment variable EXIT_EXAM_SALT to a private secret before sharing outputs/clean/.')
    res = pipeline.run_all(show=False)
    print('\nDone. See outputs/key_numbers.md, outputs/tables/*.xlsx and outputs/figures/*.png')
