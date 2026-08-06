"""Classes and functions for force constant regression."""
# -*- coding: utf-8 -*-
# Copyright (C) 2021-2023 Changpeng Lin
# All rights reserved.

__all__ = ["Optimizer"]

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LinearRegression,
    LassoCV,
    RidgeCV,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


class Optimizer(object):
    """Intertatomic force constant optimizer.
    
    This class defines the regression methods to extract interatomic
    force constants of a lattice potential. It contains all details
    including parameters of a regressor to fit force constants. The
    implementation is based on scikit-learn package.
    
    """

    def __init__(
        self,
        method="ols",
        nalpha=100,
        alpha_min=1e-6,
        alpha_max=1e-2,
        alpha=None,
        cv=5,
        tol=1e-4,
        max_iter=20000,
        rand_seed=None,
        standardize=False,
        fit_intercept=False,
    ):
        """Initialization function.

        Parameters
        ----------
        method : str
           Fit method to get interatomic force constants. It can be
           'ols' for ordinary least-square, 'lasso' for least absolute
           shrinkage and selection operator in case of compressive sensing
           problem, and 'ridge' for ridge regression.
        nalpha : int
            The number of alpha parameters to be generated. A list of alpha
            parameters spaced evenly on a log scale will be generated.
        alpha_min : float
            The minimum value of alpha. A list of alpha parameters spaced
            evenly on a log scale will be generated.
        alpha_max : float
            The maximum value of alpha. A list of alpha parameters spaced
            evenly on a log scale will be generated.
        alpha : numpy.ndarray
            A list of parameters to control the sparseness of fitting results.
            Cross validation will be used to determine the optimal value. It 
            cannot be set together with nalpha, alpha_min and alpha_max.
        cv : int
            The fold of cross-validation splitting strategy.
        tol : float
            The tolerance for the optimization.
        max_iter: int
            The maximum number of iterations to find the solution.
        rand_seed : int
            The seed for random number generator.
        standardize : bool
            If True, the fit matrix and target values are standardized before fitting.
        fit_intercept : bool
            If True, calculate the intercept for the model.

        """
        self._method = method
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._nalpha = nalpha
        self._cv = cv
        self._tol = tol
        self._max_iter = max_iter
        self._rand_seed = rand_seed
        self._standardize = standardize
        self._fit_intercept = fit_intercept

        if alpha is not None:
            self._alpha = alpha
        else:
            self._alpha = np.logspace(alpha_min, alpha_max, nalpha)

        self._results = {}
        self._metrics = {}

    def fit(self, A, F, weights=None):
        """Fit regression model with the specified method and parameters.

        Parameters
        ----------
        A : numpy.ndarray
            2D sensing matrix.
        F : numpy.ndarray
            1D interatomic force array.
        weights : numpy.ndarray
            Weights for each sample in sensing matrix A.

        """
        """Initialization"""
        if self._method.upper() == "OLS":
            self._model = LinearRegression(fit_intercept=self._fit_intercept)
        elif self._method.upper() == "LASSO":
            self._model = LassoCV(
                alphas=self._alpha,
                max_iter=self._max_iter,
                tol=self._tol,
                cv=self._cv,
                fit_intercept=self._fit_intercept,
                random_state=self._rand_seed,
                selection="random",
            )
        elif self._method.upper() == "RIDGE":
            self._model = RidgeCV(
                alphas=self._alpha,
                fit_intercept=self._fit_intercept,
                store_cv_values=True,
            )

        """Fit"""
        if self._standardize:
            scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
            scaler.fit_transform(A)
            F_scale = 1.0 / np.std(F)
            F_scaled = F * F_scale
            self._model.fit(A, F_scaled, sample_weight=weights)
            scaler.inverse_transform(A)
            coef = self._model.coef_ / F_scale
            scaler.transform(coef.reshape(1, -1)).reshape(-1,)
        else:
            self._model.fit(A, F, sample_weight=weights)
            coef = self._model.coef_
        self._results["coef"] = np.where(abs(coef) < 1e-6, 0, coef)
        self._model.coef_ = self._results["coef"]

        if self._method.upper() == "LASSO":
            self._results["alpha"] = self._model.alpha_
            self._results["n_iter"] = self._model.n_iter_
            alpha_idx = np.argmin(np.abs(self._model.alphas_ - self._results["alpha"]))
            self._metrics["mse_path"] = self._model.mse_path_[alpha_idx]
            self._metrics["mse_path_mean"] = np.mean(self._model.mse_path_[alpha_idx])
            self._metrics["rmse_path"] = np.sqrt(self._metrics["mse_path"])
            self._metrics["rmse_path_mean"] = np.mean(self._metrics["rmse_path"])
            self._metrics["n_featrues"] = self._model.n_features_in_
        elif self._method.upper() == "RIDGE":
            self._results["alpha"] = self._model.alpha_

        """Metrics"""
        eps = np.finfo(np.float64).eps
        F_pred = self.predict(A)
        F_err = np.abs(F_pred - F)
        F_re = F_err / np.maximum(np.abs(F), eps)

        # Relative error
        self._metrics["re"] = np.sqrt(np.dot(F_err, F_err) / np.dot(F, F))
        # R^2 score
        self._metrics["r2_score"] = r2_score(F, F_pred, sample_weight=weights)
        # Mean absolute error
        self._metrics["mae"] = mean_absolute_error(F, F_pred, sample_weight=weights)
        # Mean absolute percentage error
        self._metrics["mape"] = mean_absolute_percentage_error(
            F, F_pred, sample_weight=weights
        )
        # Mean squared error
        self._metrics["mse"] = mean_squared_error(F, F_pred, sample_weight=weights)
        # Root mean squared error
        self._metrics["rmse"] = np.sqrt(self._metrics["mse"])
        # Mean square percentage error
        self._metrics["mspe"] = np.average(np.square(F_re), weights=weights, axis=0)
        # Root mean square percentage error
        self._metrics["rmspe"] = np.sqrt(self._metrics["mspe"])

    def predict(self, A):
        """Predit using the trained model.

        Parameters
        ----------
        A : numpy.ndarray
            2D sensing matrix.
            
        Returns:
        -------
        numpy.ndarray
        Predicted interatomic forces.

        """
        return self._model.predict(A)

    @property
    def results(self):
        """Return fitting results."""
        return self._results

    @property
    def metrics(self):
        """Return multiple metrics of the model."""
        return self._metrics

    @property
    def model(self):
        """Return sklearn estimator instance."""
        return self._model

    def get_paras(self):
        """Return parameters used in fittings."""
        return self._model.coef_

    def __repr__(self):
        pass
