# Exit-exam analysis (medical graduates, rounds 2023 - 2026 R2)

Reproducible pipeline for the five exit-exam result files. It answers: **why is the chance of failing so high, who
retakes the exam, and how likely is a retaker to pass?**

```
exit_exam_project/
  data/raw/                five original Excel files (never modified)
  src/exitexam/            config, cleaning, linking, statistics, analysis_a, analysis_b, plotting, pipeline
  notebooks/               01_data_preparation, 02_part_A_three_rounds, 03_part_B_five_rounds
  outputs/                 key_numbers.md, tables/*.xlsx, figures/*.png, clean/clean_rounds_anonymized.csv
  docs/PAPER_GUIDE.md      how to turn the results into a research paper
  run_all.py               runs everything in about a minute
```

## Run

```
pip install -r requirements.txt
python run_all.py                      # rebuilds every table, figure and outputs/key_numbers.md
jupyter lab notebooks/                 # step-by-step version with explanations
```

## Two analyses, never mixed

| | Rounds | Why |
|---|---|---|
| **Part A** | 2025, 2026-R1, 2026-R2 | only these files say whether the graduate comes from a public / private / foreign institution, and which one |
| **Part B** | all five rounds | institution is unknown for 2023 and 2024 (province is always Kabul), but candidates can be followed across five rounds |

Every row carries one `exam` label. Rounds are identified only by that label, sorted strictly in time, and the code
asserts one row per person per round and the 216 pass mark in every round.

## Cleaning rules (decided with the researcher)

* rows with an **empty score** or a **zero score** are removed (absent or cheating)
* rows with an annulled or unreadable result are removed
* every removed row gets exactly one reason; counts are in `audit` (outputs/tables)

## Linking candidates across rounds

The candidate number changes between rounds, so persons are linked by **name + father name + grandfather name +
field of study** after normalising Arabic/Persian/Pashto letter variants and removing spaces. Records whose triple
is not unique inside one round are never linked. Validation: linked records keep the same field (99.9%) and the same
institute (99.8%), and their scores correlate; a typo-tolerant rule changes almost nothing (sensitivity table).

## Statistics used (and why)

* Wilson confidence intervals for proportions within one round; person-cluster bootstrap when rounds are pooled
* effect sizes next to p-values (odds ratio, Cramer V, Cliff delta)
* logistic regression with cluster-robust standard errors (one person can appear in several rounds)
* institute comparison on first attempts, standardised for type, round and field (observed vs expected),
  Benjamini-Hochberg FDR control, empirical-Bayes shrinkage, funnel plot
* Kitagawa decomposition: change in pass rate = composition effect + rate effect
* cohort curves and a hazard-style view (only candidates who have not passed yet)
* sensitivity analysis for the linking rule; every random step uses `C.SEED`

## Privacy

Names and candidate numbers are used only for linking. Output files contain a keyed hash (`person_id`). Before
sharing `outputs/clean/`, set a private secret: `export EXIT_EXAM_SALT="..."` and rerun. Do not publish `data/raw/`.

## Adding a new round

Add the file to `data/raw/`, then add its label to `EXAMS` and its column layout to `SCHEMAS` in
`src/exitexam/config.py` (and to `PART_A_EXAMS` if it contains institution information).

## Publishing on GitHub

* The raw Excel files contain candidate names. They are excluded by `.gitignore` (see `data/README.md`); never commit them.
* `.gitignore` also excludes `outputs/clean/`, `outputs/tables/` and the three figures that show
  institute names (A03, A04, A05), because they can identify institutes or individuals.
* Some modules (`cleaning.py`, `labels.py`, `public_*.py`) contain Pashto strings on purpose: they are the text patterns of the
  source files and the Pashto report/chart text, not documentation.
* The notebooks are written in English. Before sharing derived data, set `EXIT_EXAM_SALT` to a private secret.
* Check the legal permission to use and publish the data before making the repository public.
