#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:25:43 2020

@author: robson
"""
import pandas as pd
import numpy as np
import os
import re
# from treeinterpreter import treeinterpreter as ti, utils
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate
import seaborn as sns
from matplotlib import pyplot as plt
import shap
from datetime import date
from IPython import embed
import pickle
base_path = os.getcwd()


##########################
# Métodos utilizados     #
##########################
def save_to_pickle(jobject, name_arq):

    arq = open(base_path+\
            '/data_sus/finais/shaps/{}'.format(name_arq), 'wb')
    pickle.dump(object, arq, pickle.HIGHEST_PROTOCOL)
    arq.close()


def get_indices_columns_time(columns):
    columns_time = list()
    columns_others = list()

    for c in columns:
        if re.match("t-.", c):
            columns_time.append(c)
        else:
            columns_others.append(c)

    if len(columns_time) > 0:

        index_time_cols = (len(columns_others),
                           len(columns_others) + len(columns_time) - 1)

        columns_others = columns_others + columns_time

        return columns_others, index_time_cols

    else:
        return columns_others, list()


def apply_shap_values(X_train, regressor, m, base_path):

    if m in ['randomforest', "extratrees", "xgboost", "lightGBM"]:
        explainer = shap.TreeExplainer(regressor)
        shap_values = explainer.shap_values(X_train, check_additivity=False)
    elif m in ['svr']:
        explainer = shap.KernelExplainer(regressor.predict_proba, X_train, link="logit")
        shap_values = explainer.shap_values(X_train, nsamples=5)
    elif m in ["linear_regression", "polynomial_regression"]:
        explainer = shap.LinearExplainer(regressor, X_train, feature_dependence="independent")
        shap_values = explainer.shap_values(X_train)

    df_shap_values_1 = pd.DataFrame(shap_values, columns=list(X_train.columns))

    list_importance = list()

    for c in list(df_shap_values_1.columns):
        df_shap_values_1[c] = df_shap_values_1[c].apply(lambda x: abs(x))
        list_importance.append(np.mean(df_shap_values_1[c]))

    df_feature_importance = pd.DataFrame(columns=["feature", "importance"])
    df_feature_importance["feature"] = list(X_train.columns)
    df_feature_importance["importance"] = list_importance
    df_feature_importance["importance_perc"] = df_feature_importance["importance"].apply(
        lambda x: round(x/np.sum(df_feature_importance["importance"]), 2))
    df_feature_importance.sort_values(by=["importance"], inplace=True, ascending=False)

    # df_feature_importance.to_csv(base_path+\
    #         '/data_sus/finais/feature_importances/feature_importance_{}.csv'.format(m))

    f, ax = plt.subplots(figsize=(15, 15))
    shap.summary_plot(shap_values, X_train)

    return df_feature_importance, shap_values


def apply_cross_validate(X_train, y_train, regressor, name):

    scoring = {'r2': 'r2',
               'neg_mean_squared_error': 'neg_mean_squared_error',
               'neg_mean_absolute_error': 'neg_mean_absolute_error'}

    accuracies = cross_validate(estimator=regressor,
                                X=X_train,
                                y=y_train,
                                scoring=scoring,
                                cv=10)

    print("####################### {} ###################".format(name))
    print('test mean r2: {}'.format(round(accuracies['test_r2'].mean(), 2)))
    print('test std r2: {}'.format(round(accuracies['test_r2'].std(), 2)))
    print(
        'MSE: {}'.format(round(accuracies['test_neg_mean_squared_error'].mean(), 2)))
    print('MSE std: {}'.format(round(accuracies['test_neg_mean_squared_error'].std(), 2)))
    print('MAE: {}'.format(round(accuracies['test_neg_mean_absolute_error'].mean(), 2)))
    print('MAE std: {}'.format(round(accuracies['test_neg_mean_absolute_error'].std(), 2)))

    return round(accuracies['test_r2'].mean(), 2)


def apply_adjusted_cross_validate(X_train, y_train, regressor, name):

    # Listas onde serão armazenados os resultados
    result_r_squared = list()
    mae = list()
    rmse = list()

    # Define quantidade que vai para teste, nesse caso 10%
    bucket = int(0.1 * X_train.shape[0])

    # Identificaçao das linhas para poder separar em treino e teste
    X_train['id'] = list(range(X_train.shape[0]))
    df_y_train = pd.DataFrame(list(y_train), columns=['values'])
    df_y_train['id'] = list(range(X_train.shape[0]))

    for i in list(range(10)):

        # Seleciona os 10% da base para teste
        X_test = X_train.iloc[bucket * i:bucket * (i + 1), :]
        df_y_test = df_y_train.iloc[bucket * i:bucket * (i + 1), :]
        y_test = list(df_y_test['values'])

        # Seleciona os 90% de treinamento
        X_train_sample = X_train[~X_train['id'].isin(list(X_test['id']))]
        y_train_sample = list(df_y_train[~df_y_train['id'].isin(list(df_y_test['id']))]['values'])

        # Elimina a coluna de identificação
        X_train_sample.drop(columns=['id'], inplace=True)
        X_test.drop(columns=['id'], inplace=True)

        # Ajusta o modelo com a base de treinamento
        regressor.fit(X_train_sample, y_train_sample)

        # Previsão da base de teste
        y_pred = list(regressor.predict(X_test))

        variance_model = list()
        variance_avg = list()

        for i in list(range(len(y_pred))):
            variance_model.append(pow(y_test[i] - y_pred[i], 2))

        y_train = list(y_train)
        for i in list(range(len(y_train))):
            variance_avg.append(pow(y_train[i] - np.mean(y_train), 2))

        r_square_adjusted = 1 - (np.sum(variance_model) / np.sum(variance_avg))

        result_r_squared.append(round(r_square_adjusted, 2))
        mae.append(round(mean_absolute_error(y_test, y_pred), 2))
        rmse.append(round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))

    print("######################## {} ########################".format(name))
    print('adjusted_r_square: ({}, {})'.format(round(np.mean(result_r_squared), 2), round(np.std(result_r_squared), 2)))
    print('mae: ({}, {})'.format(round(np.mean(mae), 2), round(np.std(mae), 2)))
    print('rmse: ({}, {})'.format(round(np.mean(rmse), 2), round(np.std(rmse), 2)))


def apply_grid_search(X_train, y_train, parameters_grid, regressor):

    grid_search = GridSearchCV(estimator=regressor,
                               param_grid=parameters_grid,
                               scoring='r2',
                               cv=10,
                               n_jobs=-1
                               )

    grid_search = grid_search.fit(X_train, np.ravel(y_train))
    print('best precision: {}'.format(round(grid_search.best_score_, 2)))
    print('best_parameters: {}'.format(grid_search.best_params_))

    return grid_search.best_params_


