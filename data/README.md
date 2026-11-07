# data/raw

The five raw Excel files are **not** included in this repository because they contain candidate names.
To reproduce the analysis, place them here with these names:

| Round | File name |
|---|---|
| 2023 | `ama-afghanistan-exit-2023-1.xlsx` |
| 2024 | `ama-afghanistan-exit-2024-1.xlsx` |
| 2025 | `ama-afghanistan-exit-2025-1.xlsx` |
| 2026 round 1 | `ama-afghanistan-exit-2026-1.xlsx` |
| 2026 round 2 | `ama-afghanistan-exit-2026-2.xlsx` |

Column layouts are described in `src/exitexam/config.py`. Never commit these files.
Set a private secret before sharing any derived file: `export EXIT_EXAM_SALT="..."`.
