#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:30:01 2020

@author: robson
"""
import pandas as pd
import numpy as np
import os
import re
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import statsmodels.api as sm
import catboost as cb
from sklearn.preprocessing import MinMaxScaler


from measuringResults import apply_grid_search, apply_cross_validate, \
    apply_adjusted_cross_validate
from sklearn.base import BaseEstimator, RegressorMixin


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """
    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


def apply_feature_scaling(X_train):

    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                           columns=X_train.columns)

    return X_train, scaler


def backward_elimination(X_train, y_train, sl=0.05):
    elimitedColumns = list()

    # Creating a first column with ones
    X = np.append(arr=np.ones((len(X_train), 1)),
                  values=X_train, axis=1)
    columns = list(range(X.shape[1]))
    X_opt = X[:, columns]

    for i in range(len(X_opt[0])):
        regressor_OLS = sm.OLS(y_train, X_opt).fit()
        maxVar = float(max(regressor_OLS.pvalues))
        if maxVar > sl:
            for j in range(len(columns)):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    del columns[j]
                    X_opt = X[:, columns]
    # print(regressor_OLS.summary())

    return regressor_OLS, columns


def run_linear_regression(X_train, y_train, grid_search=False):
    '''
    best precision: 0.26
    best_parameters: {'copy_X': True, 'fit_intercept': True}
    '''

    # X_train, scaler = apply_feature_scaling(X_train)

    if grid_search:
        parameters = {'fit_intercept': [True, False],
                      'copy_X': [True, False]}

        parameters = apply_grid_search(X_train, y_train, parameters,
                                       LinearRegression())
    else:
        parameters = {'fit_intercept': True,
                      'copy_X': True}

    regressor = LinearRegression(copy_X = parameters['copy_X'], 
                                 fit_intercept = parameters['fit_intercept'])
    
    regressor.fit(X_train, y_train)

    r2 = apply_cross_validate(X_train, y_train, regressor, "Linear Regression")
    
    scaler = False

    return regressor, scaler, r2


def run_multiple_linear_regression(X_train, y_train):
    '''
    backwardElimination rsquare -3.782738300893323e+16
    backwardEliminationRSquare rsquare -3.782738300893323e+16
    :return:
    '''
    
    X_train, scaler = apply_feature_scaling(X_train)

    # Building the model using Backward Elimination
    regressor, columns = backward_elimination(X_train, y_train)
    # regressor = self.backwardEliminationRSquare()

    # X_opt_test = np.append(arr=np.ones((len(X_test), 1)),
    #               values=X_test, axis=1)
    #
    # X_opt_test = X_opt_test[:, columns]
    #
    # # Predicting the Test set results
    # y_pred = regressor.predict(X_opt_test)
    #
    # print('Multiple Linear Regression r2: {}'.format(r2_score(y_test, y_pred)))

    X_opt_train = np.append(arr=np.ones((len(X_train), 1)),
                           values=X_train, axis=1)

    X_opt_train = X_opt_train[:, columns]

    regressor = SMWrapper(sm.OLS)

    apply_cross_validate(X_opt_train, y_train,
                        regressor, "Multiple Regression")
    
    return regressor, scaler


def run_randomforest(X_train, y_train, grid_search=False):
    '''
    best_precision: 0.53
    best_parameters = {'bootstrap': True, 'criterion': 'mae',
                          'max_features': 0.7, 'max_samples': 0.9,
                          'n_estimators': 200, 'warm_start': False}

    'max_depth': 200, 'min_samples_leaf': 100,
                      'min_samples_split': 100,

    '''

    if grid_search:
        parameters_grid = {'n_estimators':[50, 100, 200],
                       'criterion': ["mse", "mae"],
                       'warm_start': [True, False],
                       'max_samples': [0.7, 0.9],
                       'max_features': [0.7, 0.9],
                       'bootstrap': [True]}

        best_parameters = apply_grid_search(X_train, y_train, parameters_grid,
                                       RandomForestRegressor())

    else:
        # best_precision: 0.53
        best_parameters = {'bootstrap': True, 'criterion': 'mae',
                          'max_features': 0.7, 'max_samples': 0.9,
                          'n_estimators': 200, 'warm_start': False}

    regressor = RandomForestRegressor(n_estimators=best_parameters['n_estimators'],
                                      criterion=best_parameters['criterion'],
                                      warm_start=best_parameters['warm_start'],
                                      max_samples=best_parameters['max_samples'],
                                      max_features=best_parameters['max_features'],
                                      bootstrap=best_parameters['bootstrap'],
                                      max_depth=200,
                                      min_samples_leaf=100,
                                      min_samples_split=100,
                                      random_state=0)

    # regressor = RandomForestRegressor()
    
    regressor.fit(X_train, np.ravel(y_train))

    # r2 = apply_cross_validate(X_train, y_train, regressor, "randomforest")
    r2 = 0
    
    return regressor, r2


def run_extratrees(X_train, y_train, grid_search=False):

    '''
    best precision: 0.52
    best_parameters: {'bootstrap': True, 'criterion': 'mae', 'max_features': 0.7,
                        'max_samples': 0.9,  'n_estimators': 100, 'warm_start': False}
    '''

    if grid_search:
        parameters_grid = {'n_estimators':[50, 100, 200],
                       'criterion': ["mse", "mae"],
                       'warm_start': [True, False],
                       'max_samples': [0.7, 0.9],
                       'max_features': [0.7, 0.9],
                       'bootstrap': [True]}

        best_parameters = apply_grid_search(X_train, y_train, parameters_grid,
                                            ExtraTreesRegressor())
    else:
        best_parameters = {'bootstrap': True, 'criterion': 'mae',
                          'max_features': 0.7,  'max_samples': 0.9,
                          'n_estimators': 100, 'warm_start': False}

    regressor = ExtraTreesRegressor(n_estimators=best_parameters['n_estimators'],
                                      criterion=best_parameters['criterion'],
                                      warm_start=best_parameters['warm_start'],
                                      max_samples=best_parameters['max_samples'],
                                      max_features=best_parameters['max_features'],
                                      bootstrap=best_parameters['bootstrap'],
                                      max_depth=200,
                                      min_samples_leaf=100,
                                      min_samples_split=100,
                                      random_state=0)

    regressor.fit(X_train, np.ravel(y_train))

    r2 = apply_cross_validate(X_train, y_train, regressor, "extratrees")

    return regressor, r2


def run_polynomial_regression(X_train, y_train, grid_search=False):
    '''
    rsquare -1.4091198027793736e+16
    :return:
    '''

    # X_train, scaler = apply_feature_scaling(X_train)

    if grid_search:
        parameters = {'fit_intercept': True,
                      'copy_X': True}

        for n in [X_train.shape[1], round(X_train.shape[1]*0.7, 0), round(X_train.shape[1]*0.5, 0)]:

            poly_reg = PolynomialFeatures(degree=n)
            X_poly = poly_reg.fit_transform(X_train)

            best_parameters = apply_grid_search(X_poly, y_train, parameters,
                                           LinearRegression())
    else:
        best_parameters = {'fit_intercept': True,
                      'copy_X': True}

    # Fitting Polynomial Regression to the dataset
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)

    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)

    r2 = apply_adjusted_cross_validate(X_train, y_train, regressor, "Regressão Polinomial")

    scaler = False

    return regressor, scaler, r2


def run_svr(X_train, y_train, grid_search=False):
    '''
    best precision: 0.5
    best_parameters: {'C': 10, 'degree': 3, 'epsilon': 0.5, 'gamma': 'scale', 'kernel': 'poly'}
    '''
    # Feature Scaling
    # X_train, scaler = apply_feature_scaling(X_train)

    if grid_search:
        parameters = {'kernel': ['rbf', 'poly', 'sigmoid'],
                      'degree': [3, 5, 10],
                      'gamma': ['scale', 'auto'],
                      'C': [1, 5, 10],
                      'epsilon': [0.1, 0.5, 0.7]
                      }

        best_parameters = apply_grid_search(X_train, y_train, parameters,
                                       SVR())
    else:
        best_parameters = {'kernel': 'poly',
                      'degree': 3,
                      'gamma': 'scale',
                      'C': 10,
                      'epsilon': 0.5}

    # Fitting the Regression Model to the dataset
    regressor = SVR(kernel=best_parameters['kernel'], degree=best_parameters['degree'],
                    gamma=best_parameters['gamma'], C=best_parameters['C'],
                    epsilon=best_parameters['epsilon'])

    regressor.fit(X_train, y_train)

    # r2 = apply_adjusted_cross_validate(X_train, y_train, regressor, "SVM")
    r2 = 0

    scaler = False
    
    return regressor, scaler, r2


def run_xgboost(X_train, y_train, grid_search=False):
    '''

    best precision: 0.5
    best_parameters: {'colsample_bytree': 0.7, 'gamma': 1, 'min_child_weight': 1,
                        'n_estimators': 100, 'subsample': 0.9, 'tree_method': 'exact'}

    '''
    if grid_search:
        parameters = {
            'min_child_weight': [1, 5, 10],
            'gamma': [1, 2, 5],
            'subsample': [0.6, 0.7, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.9],
            'n_estimators': [100, 300, 500],
            'tree_method': ['hist', 'exact', 'approx'],
        }

        best_parameters = apply_grid_search(X_train, y_train, parameters,
                                            XGBRegressor())
    else:
        best_parameters = {'colsample_bytree': 0.7, 'gamma': 1, 'min_child_weight': 1,
                    'n_estimators': 100, 'subsample': 0.9, 'tree_method': 'exact'}

    regressor = XGBRegressor(n_estimators=best_parameters['n_estimators'],
                              min_child_weight=best_parameters['min_child_weight'],
                              gamma=best_parameters['gamma'],
                              colsample_bytree=best_parameters['colsample_bytree'],
                              tree_method=best_parameters['tree_method'])

    regressor.fit(X_train, np.ravel(y_train))

    r2 = apply_cross_validate(X_train, y_train, regressor, "XGBoost")

    return regressor, r2


def run_lightGBM(x_train, y_train, grid_search=False, standart=False):

    '''
    best precision: 0.53
    best_parameters: {'bagging_fraction': 0.7, 'bagging_freq': 1,
                        'boosting': 'dart', 'feature_fraction': 0.7,
                        'n_estimators': 100}

    :param X_train:
    :param y_train:
    :param grid_search:
    :return:
    '''
    if standart:
        regressor = LGBMRegressor()

    else:
        if grid_search:
            parameters = {
                'n_estimators': [100, 200, 500],
                'boosting': ['gbdt', 'rf', 'dart'],
                'bagging_freq': [1, 5, 10],
                'bagging_fraction': [0.5, 0.7, 0.9],
                'feature_fraction': [0.5, 0.7, 1]
            }

            best_parameters = apply_grid_search(x_train, y_train, parameters,
                                                LGBMRegressor())

        else:
            # best_parameters = {
            #     'n_estimators': 100,
            #     'boosting': 'dart',
            #     'bagging_freq': 1,
            #     'bagging_fraction': 0.7,
            #     'feature_fraction': 0.7
            # }

            best_parameters = {'bagging_fraction': 0.5,
                              'bagging_freq': 10,
                              'boosting': 'rf',
                              'feature_fraction': 0.5,
                              'n_estimators': 100}

        regressor = LGBMRegressor(n_estimators=best_parameters['n_estimators'],
                                          boosting=best_parameters['boosting'],
                                          bagging_freq=best_parameters['bagging_freq'],
                                          bagging_fraction=best_parameters['bagging_fraction'],
                                          feature_fraction=best_parameters['feature_fraction'])

    regressor.fit(x_train, np.ravel(y_train))

    # r2 = apply_cross_validate(x_train, y_train, regressor, "lightGBM")

    return regressor, 0


def run_catboost(x_train, y_train, grid_search=False, standart=False):

    if standart:
        regressor = cb.CatBoostRegressor()

    else:
        regressor = cb.CatBoostRegressor(learning_rate=0.1,
                                         bootstrap_type='Bernoulli',
                                         grow_policy='Lossguide',
                                         boosting_type='Plain')

    regressor.fit(x_train, y_train)

    return regressor, 0

