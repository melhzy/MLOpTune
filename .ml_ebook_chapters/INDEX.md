# ml_book reference index

Routing table for the ML references behind the frailty classification work.
Open the one chapter a question needs instead of the ~120k-token whole book.

For a pinpoint fact, grep across `ml_ebook_chapters/*.md` and pull a line
window; that costs a few hundred tokens and beats opening any file. Use a whole
chapter file when a question needs a topic rather than one fact.

Do not read `../.archive/`. It holds the superseded monolith `ml_ebook.md` and
the `_build/` pipeline that produced these files, kept for provenance and
rebuild only. Its text is already fully covered by the chapters below.

## Hands-On Machine Learning (Geron, 2nd ed. early release)

| File | Lines | ~Tokens | Sections |
|---|---|---|---|
| `ch00_front_matter.md` | 207 | ~1,708 | Copyright, Table of Contents |
| `ch01_machine_learning_landscape.md` | 639 | ~13,069 | What Is Machine Learning?, Why Use Machine Learning?, Types of Machine Learning Systems, Main Challenges of Machine Learning, Testing and Validating, Exercises |
| `ch02_end_to_end_machine_learning_project.md` | 1409 | ~23,593 | Working with Real Data, Look at the Big Picture, Get the Data, Discover and Visualize the Data to Gain Insights, Prepare the Data for Machine Learning Algorithms, Select and Train a Model, Fine-Tune Your Model, Launch, Monitor, and Maintain Your System, Try It Out!, Exercises |
| `ch03_classification.md` | 750 | ~11,499 | MNIST, Training a Binary Classifier, Performance Measures, Multiclass Classification, Error Analysis, Multilabel Classification, Multioutput Classification, Exercises |
| `ch04_training_models.md` | 1043 | ~17,101 | Linear Regression, Gradient Descent, Polynomial Regression, Learning Curves, Regularized Linear Models, Logistic Regression, Exercises |
| `ch05_support_vector_machines.md` | 514 | ~8,924 | Linear SVM Classification, Nonlinear SVM Classification, SVM Regression, Under the Hood, Exercises |
| `ch06_decision_trees.md` | 276 | ~5,633 | Training and Visualizing a Decision Tree, Making Predictions, Estimating Class Probabilities, The CART Training Algorithm, Computational Complexity, Gini Impurity or Entropy?, Regularization Hyperparameters, Regression, Instability, Exercises |
| `ch07_ensemble_learning_and_random_forests.md` | 561 | ~9,772 | Voting Classifiers, Bagging and Pasting, Random Patches and Random Subspaces, Random Forests, Boosting, Stacking, Exercises |
| `ch08_dimensionality_reduction.md` | 454 | ~9,046 | The Curse of Dimensionality, Main Approaches for Dimensionality Reduction, PCA, Kernel PCA, LLE, Exercises |
| `ch09_unsupervised_learning_techniques.md` | 932 | ~19,925 | Clustering, Gaussian Mixtures |

## Frailty cytokine preliminary results

Not split, on purpose. One Read covers it.

| File | Lines | ~Tokens | Sections |
|---|---|---|---|
| `../preliminary_results.md` | 202 | ~9,031 | eScholarship@UMassChan, T lymphocyte regulatory cytokines, TITLE, AUTHORS, ABSTRACT, INTRODUCTION, METHODS, RESULTS, DISCUSSION, FUNDING, ACKNOWLEDGEMENTS, REFERENCES, FIGURE LEGENDS, Table 2. Odds ratios of frailty for unadjusted and logistic regression adjusted, Figure 1, Figure 2, Figure 3, Supplemental Figure 1 |
