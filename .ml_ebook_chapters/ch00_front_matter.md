<!-- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (Geron, 2nd ed. early release) | Front matter | source lines 1-207 of ml_ebook.md -->

# Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow

**Concepts, Tools, and Techniques to Build Intelligent Systems**

Second Edition. Early Release (2019-01-24, second release).

Aurélien Géron. Published by O'Reilly Media, Inc.

## Copyright

Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow by Aurélien Géron

Copyright © 2019 O'Reilly Media. All rights reserved.

Printed in the United States of America.

Published by O'Reilly Media, Inc., 1005 Gravenstein Highway North, Sebastopol, CA 95472.

O'Reilly books may be purchased for educational, business, or sales promotional use. Online editions are also available for most titles (http://oreilly.com). For more information, contact our corporate/institutional sales department: 800-998-9938 or corporate@oreilly.com.

- Editor: Nicole Tache
- Cover Designer: Karen Montgomery
- Interior Designer: David Futato
- Illustrator: Rebecca Demarest

June 2019: Second Edition

- Revision History for the Early Release
- 2018-11-05: First Release
- 2019-01-24: Second Release

See http://oreilly.com/catalog/errata.csp?isbn=9781492032649 for release details.

The O'Reilly logo is a registered trademark of O'Reilly Media, Inc. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, the cover image, and related trade dress are trademarks of O'Reilly Media, Inc.

While the publisher and the author have used good faith efforts to ensure that the information and instructions contained in this work are accurate, the publisher and the author disclaim all responsibility for errors or omissions, including without limitation responsibility for damages resulting from the use of or reliance on this work. Use of the information and instructions contained in this work is at your own risk. If any code samples or other technology this work contains or describes is subject to open source licenses or the intellectual property rights of others, it is your responsibility to ensure that your use thereof complies with such licenses and/or rights.

978-1-492-03264-9

[LSI]

## Table of Contents

- Chapter 1. The Machine Learning Landscape
  - What Is Machine Learning?
  - Why Use Machine Learning?
  - Types of Machine Learning Systems
    - Supervised/Unsupervised Learning
    - Batch and Online Learning
    - Instance-Based Versus Model-Based Learning
  - Main Challenges of Machine Learning
    - Insufficient Quantity of Training Data
    - Nonrepresentative Training Data
    - Poor-Quality Data
    - Irrelevant Features
    - Overfitting the Training Data
    - Underfitting the Training Data
    - Stepping Back
  - Testing and Validating
  - Exercises
- Chapter 2. End-to-End Machine Learning Project
  - Working with Real Data
  - Look at the Big Picture
    - Frame the Problem
    - Select a Performance Measure
    - Check the Assumptions
  - Get the Data
    - Create the Workspace
    - Download the Data
    - Take a Quick Look at the Data Structure
    - Create a Test Set
  - Discover and Visualize the Data to Gain Insights
    - Visualizing Geographical Data
    - Looking for Correlations
    - Experimenting with Attribute Combinations
  - Prepare the Data for Machine Learning Algorithms
    - Data Cleaning
    - Handling Text and Categorical Attributes
    - Custom Transformers
    - Feature Scaling
    - Transformation Pipelines
  - Select and Train a Model
    - Training and Evaluating on the Training Set
    - Better Evaluation Using Cross-Validation
  - Fine-Tune Your Model
    - Grid Search
    - Randomized Search
    - Ensemble Methods
    - Analyze the Best Models and Their Errors
    - Evaluate Your System on the Test Set
  - Launch, Monitor, and Maintain Your System
  - Try It Out!
  - Exercises
- Chapter 3. Classification
  - MNIST
  - Training a Binary Classifier
  - Performance Measures
    - Measuring Accuracy Using Cross-Validation
    - Confusion Matrix
    - Precision and Recall
    - Precision/Recall Tradeoff
    - The ROC Curve
  - Multiclass Classification
  - Error Analysis
  - Multilabel Classification
  - Multioutput Classification
  - Exercises
- Chapter 4. Training Models
  - Linear Regression
    - The Normal Equation
    - Computational Complexity
  - Gradient Descent
    - Batch Gradient Descent
    - Stochastic Gradient Descent
    - Mini-batch Gradient Descent
  - Polynomial Regression
  - Learning Curves
  - Regularized Linear Models
    - Ridge Regression
    - Lasso Regression
    - Elastic Net
    - Early Stopping
  - Logistic Regression
    - Estimating Probabilities
    - Training and Cost Function
    - Decision Boundaries
    - Softmax Regression
  - Exercises
- Chapter 5. Support Vector Machines
  - Linear SVM Classification
    - Soft Margin Classification
  - Nonlinear SVM Classification
    - Polynomial Kernel
    - Adding Similarity Features
    - Gaussian RBF Kernel
    - Computational Complexity
  - SVM Regression
  - Under the Hood
    - Decision Function and Predictions
    - Training Objective
    - Quadratic Programming
    - The Dual Problem
    - Kernelized SVM
    - Online SVMs
  - Exercises
- Chapter 6. Decision Trees
  - Training and Visualizing a Decision Tree
  - Making Predictions
  - Estimating Class Probabilities
  - The CART Training Algorithm
  - Computational Complexity
  - Gini Impurity or Entropy?
  - Regularization Hyperparameters
  - Regression
  - Instability
  - Exercises
- Chapter 7. Ensemble Learning and Random Forests
  - Voting Classifiers
  - Bagging and Pasting
    - Bagging and Pasting in Scikit-Learn
    - Out-of-Bag Evaluation
  - Random Patches and Random Subspaces
  - Random Forests
    - Extra-Trees
    - Feature Importance
  - Boosting
    - AdaBoost
    - Gradient Boosting
  - Stacking
  - Exercises
- Chapter 8. Dimensionality Reduction
  - The Curse of Dimensionality
  - Main Approaches for Dimensionality Reduction
    - Projection
    - Manifold Learning
  - PCA
    - Preserving the Variance
    - Principal Components
    - Projecting Down to d Dimensions
    - Using Scikit-Learn
    - Explained Variance Ratio
    - Choosing the Right Number of Dimensions
    - PCA for Compression
    - Randomized PCA
    - Incremental PCA
  - Kernel PCA
    - Selecting a Kernel and Tuning Hyperparameters
  - LLE
    - Other Dimensionality Reduction Techniques
  - Exercises
- Chapter 9. Unsupervised Learning Techniques
  - Clustering
    - K-Means
    - Limits of K-Means
    - Using clustering for image segmentation
    - Using Clustering for Preprocessing
    - Using Clustering for Semi-Supervised Learning
    - DBSCAN
    - Other Clustering Algorithms
  - Gaussian Mixtures
    - Anomaly Detection using Gaussian Mixtures
    - Selecting the Number of Clusters
    - Bayesian Gaussian Mixture Models
    - Other Anomaly Detection and Novelty Detection Algorithms

<!-- page 9 -->

