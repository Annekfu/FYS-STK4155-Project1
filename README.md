# FYS-STK4155 Project 1

Regression analysis, resampling methods and gradient descent for Runge's
function

    f(x) = 1 / (1 + 25 x^2),   x in [-1, 1]

Course: FYS-STK4155, Applied Data Analysis and Machine Learning,
University of Oslo, fall 2026. Author: Anne Kristin Furu.

## Structure

The repository has three folders, following the course guide: code, results
and report.

    code/                 all the source code
      common.py             colours, seed and a savefig helper
      Get_data.py           Runge data, polynomial design matrix, scaling, split
      Errors.py             MSE, R2 and the bias-variance decomposition
      LinearRegression.py   shared base class for the regression models
      OLS.py                ordinary least squares via SVD and pinv (part a)
      Ridge.py              Ridge closed form (part b)
      LASSO.py              Lasso via proximal and subgradient descent (part g)
      optimisers.py         plain, momentum, AdaGrad, RMSProp, Adam, SGD (parts e, f, h)
      helper.py             gradients, Hessian eigenvalues, learning-rate bounds
      Resample.py           bootstrap and k-fold cross-validation (parts c, d)
      notebooks/
        1_ols_ridge.ipynb        parts a, b
        2_resampling.ipynb       parts c, d
        3_gradient_descent.ipynb parts e, f
        4_lasso_sgd.ipynb        parts g, h
        5_model_selection.ipynb  part i

    results/              saved figures and selected outputs
 

## Reproducibility

All randomness uses the seed 2026, set in code/common.py. Create the
environment with

    pip install -r requirements.txt

and run the notebooks in code/notebooks in order, selecting the fys-stk4155
kernel. Figures are written to the results folder.

## Deliverables

The report is handed in on Canvas as a PDF. This repository is the code
link, and results holds the selected figures. The LLM usage declaration is
included after the conclusion of the report, per the course guidelines.
