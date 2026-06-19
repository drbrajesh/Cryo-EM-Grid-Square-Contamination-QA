# Cryo-EM Grid Square Contamination QA

Complete Eris-style challenge package for a scientific-imaging visual question answering task.

- `raw/generate_raw.py` creates deterministic cryo-EM grid-square inspection images and `raw/data.csv`.
- `dataset_description_eris_upload.md` is the dataset description for Eris.
- `prepare.py` creates public/private scene-level splits and copies images to opaque public filenames.
- `problem.md` is the solver-facing challenge statement.
- `grade.py` validates submissions and computes robust grouped accuracy.
- `rubrics.yaml` contains task-specific rubric criteria.
- `solution.ipynb` and `reference_solution.py` provide a lightweight solvability baseline.

## Submission Mapping

Dataset upload:
- Title: `Cryo-EM Grid Square Contamination QA Dataset`
- Description: paste `dataset_description_eris_upload.md`
- Data files: upload a zip containing top-level `data.csv`, `images/`, and `generate_raw.py`
- License: `CC0 1.0 Public Domain`

Challenge:
- Domain: `Computer Vision`
- Difficulty: `Medium`
- Title: `Cryo-EM Grid Square Contamination QA`
- Grade direction: `Maximize`
- Min score: `0`
- Max score: `1`
- Tags: `image`, `multimodal`, `feature-engineering`, `small-data`
- Problem description: paste `problem.md`
- Grading script: paste `grade.py`
- Prepare script: paste `prepare.py`
- Rubrics: use `rubrics.yaml`
- Reference solution: upload `solution.ipynb`

Reviewer-facing notes:
- This is scientific image VQA, not simple image classification.
- Train/test splitting is scene-level to avoid repeated-image leakage.
- Private scoring includes worst-group terms for question type, difficulty, visibility, layout, and OOD axis.
- The raw upload includes `generate_raw.py`, making the synthetic source auditable.
