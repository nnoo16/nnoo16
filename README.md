# Data Science Projects

Three small, reproducible projects exploring data analysis, classification, and regression. Each folder includes an executed Jupyter notebook, Python code, documented public data, charts, and measured results.

| Project | Focus | Result |
|---|---|---|
| [Palmer Penguins](data-science-projects/penguin-data-exploration) | Missing data, grouped summaries, bootstrap intervals | Pooled flipper–mass correlation: 0.871; subgroup comparisons included |
| [Iris Species Classifier](data-science-projects/iris-species-classifier) | Pipelines, cross-validation, held-out classification | 28/30 correct test labels (93.3%); 33.3% baseline accuracy |
| [Auto MPG Prediction](data-science-projects/auto-mpg-prediction) | Temporal validation, regression, error analysis | 3.93 MPG test MAE; 11.98 MPG baseline MAE |

## Start here

Open any project above and follow its README. To run locally, clone this repository, enter the project's directory, install its `requirements.txt` in a Python 3.12 environment, and open `analysis.ipynb` in JupyterLab. `python run_notebook.py` executes all cells from a clean kernel and fails on errors.

The projects use public datasets with source attribution and dataset-specific licenses. Random operations use seed 42. Models are selected on training data before held-out evaluation. All notebook cells were executed and checked locally; see each project's `VERIFICATION.md` and `results/`.

These educational projects were developed with ChatGPT/Codex assistance. They do not claim novel research or Kaggle competition results.
