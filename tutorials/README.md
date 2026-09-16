# MLFactory tutorials

Six executed Jupyter notebooks that take you from a first tuning run to a
trustworthy evaluation. They build on each other but each one runs on its own.
The machine-learning background comes from *Hands-On Machine Learning*
(Géron), chapters 1 to 3, referenced inline where a design choice follows the
book.

| Notebook | What you learn | Runtime |
|---|---|---|
| [01_quickstart](01_quickstart.ipynb) | The holdout recipe, `FineTuneConfig`, `FineTuner.run`, reading a `FineTuneResult`, using the fitted model | seconds |
| [02_search_spaces](02_search_spaces.ipynb) | int / float / categorical specs, `log=True`, inspecting the Optuna study, the random-startup caveat, failing trials, eager validation | under a minute |
| [03_classification](03_classification.ipynb) | Why accuracy misleads, tuning on F1 / ROC AUC / custom scorers, rebuilding the test split for a confusion matrix, multiclass metrics, stratification, string labels | under a minute |
| [04_regression](04_regression.ipynb) | RMSE via negated scorers, linear baselines vs boosting, predicted-vs-actual plots, confidence intervals, pandas inputs, multi-output targets | under a minute |
| [05_models_and_xgboost](05_models_and_xgboost.ipynb) | Listing valid model names, shortlisting model families on shared test rows, XGBoost, model/problem-type checks, meta-estimators, sparse inputs, the no-pipeline limitation | about a minute |
| [06_reproducibility_and_pitfalls](06_reproducibility_and_pitfalls.ipynb) | Seeds from experiment names, validation optimism, split noise, giving TPE room, saving models and configs | about a minute |

## Running them

From the repository root:

```bash
pip install -e .[test] jupyter matplotlib joblib
jupyter lab tutorials/
```

XGBoost cells detect when `xgboost` cannot be imported (on macOS install the
OpenMP runtime with `brew install libomp`) and print a note instead of failing.

The notebooks are committed with their outputs so they read well on GitHub
without being executed. To refresh the outputs after a library change:

```bash
jupyter nbconvert --to notebook --execute --inplace tutorials/0*.ipynb
```
