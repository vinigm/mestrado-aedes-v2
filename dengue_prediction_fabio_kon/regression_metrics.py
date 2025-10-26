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
from mlSupportMethods import *
from mlMethods import run_lightGBM, run_randomforest

# from treeinterpreter import treeinterpreter as ti, utils
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_validate
import math
from IPython import embed
from sklearn.metrics import mean_absolute_percentage_error


base_path = os.getcwd()


######################
# Métodos utilizados #
######################
def epsilon_mape(y_true, y_pred, epsilon=1):
    '''
    y_true=np.array([3, -0.5, 2, 7])
    y_pred=np.array([2.5, 0.0, 2, 8])
    epsilon=1
    '''

    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)

    return np.average(output_errors)


def analyse_r2_by_time_v3(m, df_treino, df_teste,
                       columns_filtered, categorical_columns,
                       columns_filtered_categorical, neighbor_columns, target):
    '''
    :param m: nome do modelo que será utilizado
    :param df_input_X: base de dados de treinamento x
    :param df_input_y: base de dados de treinamento y
    :param df_previsao_real: base de dados de teste
    :param columns_filtered: colunas que serão selecionadas na base de teste
    :param categorical_columns: quais colunas selecionadas são categoricas
    :param columns_filtered_categorical: das variáveis actegóricas, quais se mantém
    :param neighbor_columns: quais colunas de soma de vizinhos são mantidas
    :return: df com a análise de R2 por trimestre dentro do ano de teste
    '''

    X_train, y_train = prepare_base_treino(df_treino, columns_filtered, columns_filtered_categorical,
                            categorical_columns, target_variable=target)

    df_results = pd.DataFrame()

    for i in [1, 3, 6, 9, 12]:
        for n in list(range(10)):
            try:
                X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X_train,
                                                                    y_train,
                                                                    test_size=0.1,
                                                                    random_state=n)

                regressor, r2 = globals()["run_{}".format(m)](X_train_sample, y_train_sample)

                df_previsao = df_teste[df_teste['mes'] <= i]

                df_prev = apply_future_prediction_update_neighbors(regressor, df_previsao,
                                                  columns_filtered, columns_filtered_categorical,
                                                  list(X_train_sample.columns), categorical_columns,
                                                  neighbor_columns, target_variable=target)

                df_results = df_results.append(pd.Series(["t+" + str(i), "M_" + str(n + 1),
                                                          apply_adjusted_r2(df_treino, df_teste,
                                                          df_prev, target_variable=target)]), ignore_index=True)

            except Exception as ex:
                print("analyse_r2_by_time_v3: {}".format(ex))
                embed()

    df_results.columns = ["Time Prediction (Months)", "Model", "R2"]

    sns.boxplot(x='Time Prediction (Months)', y='R2',
                data=df_results)
    # sns.despine(offset=10, trim=True)
    plt.ylim(0, 1)
    plt.title("Model:{} (Ano: {})".format(m, df_teste['ano'].unique()[0]))
    plt.show()

    return df_results


def apply_adjusted_r2_for_each_region_test_data_variance_all_region_historical_data(df_treino, df_teste,
                                                                             target_variable='dengue_diagnosis'):
    '''
    Nesse caso o cálculo do denominador da formula considera a diferença com o valor médio para bairro, ou seja,
    no denominador é levado em consideração os dados de treino e teste
    '''

    df_treino[target_variable+'_previsto'] = [np.nan] * df_treino.shape[0]
    df_merge = df_treino.append(df_teste)

    df_adjusted_r2 = pd.DataFrame(columns=["city", "r2"])

    for b in list(df_merge['nome_bairro'].unique()):

        distance_avg_values = list()
        df_region_all_data = df_merge[df_merge["nome_bairro"] == b]
        df_region_all_data.index = df_region_all_data["chave"]

        # Caculando a variância geral (considerando dados de treino e teste) de cada região com a média
        for c in list(df_region_all_data["chave"]):
            distance_avg_values.append(
                pow(df_region_all_data.loc[c, target_variable]-np.mean(df_region_all_data[target_variable]), 2))

        distance_predicted_values = list()
        df_region = df_teste[df_teste["nome_bairro"] == b]
        df_region.index = df_region["chave"]

        # Calculo da variância para cada região com o resultado do modelo
        for c in list(df_region["chave"]):
            distance_predicted_values.append(
                pow(df_region.loc[c,  target_variable] - df_region.loc[c,  target_variable+'_previsto'], 2))

        # Calculo do R2 para cada região
        try:
            if np.sum(distance_avg_values) == 0 and np.sum(distance_predicted_values) == 0:
                adjusted_r2 = 1
            else:
                adjusted_r2 = 1 - ((np.sum(distance_predicted_values)/df_region.shape[0])/\
                                   (np.sum(distance_avg_values)/df_region_all_data.shape[0]))
                if str(adjusted_r2) == "-inf":
                    adjusted_r2 = 1 - (df_region[target_variable+'_previsto'].sum()/df_region_all_data.shape[0])
        except Exception as ex:
            embed()

        temp = pd.DataFrame([[b, round(adjusted_r2, 2)]], columns=["city", "r2"])
        df_adjusted_r2 = df_adjusted_r2.append(temp)

    df_adjusted_r2.sort_values(by=['r2'], inplace=True, ascending=False)

    return df_adjusted_r2


def apply_adjusted_r2_for_each_region_test_data_variance_all_historical_data(df_treino, df_result,
                                                                             target_variable='dengue_diagnosis'):
    '''
    Nesse caso o cálculo do denominador da formula considera toda a base de dados, a fim de calcular no denominadar
    a variância da cidade toda em comparação com cada bairro
    '''

    df_treino[target_variable+'_previsto'] = [np.nan] * df_treino.shape[0]
    df_merge = df_treino.append(df_result)

    df_adjusted_r2 = pd.DataFrame(columns=["nome_bairro", "R2", "MAE", "RMSE", 'MAPE'])

    for b in list(df_merge['nome_bairro'].unique()):

        df_region_all_data = df_merge.copy()
        df_region_all_data.index = df_region_all_data["chave"]

        # Caculando a variância geral (considerando dados de treino e teste) do município com a média
        avg_variancia = np.mean(df_region_all_data[target_variable])

        df_region_all_data['variancia_media'] = df_region_all_data[target_variable].apply(
            lambda x: pow(x-avg_variancia, 2))

        distance_avg_values = list(df_region_all_data['variancia_media'])

        df_region = df_result[df_result["nome_bairro"] == b]
        df_region.index = df_region["chave"]

        df_region['variancia_previsao'] = df_region[[target_variable, target_variable+'_previsto']].apply(
            lambda x: pow(x[0]-x[1], 2), axis=1)
        distance_predicted_values = list(df_region['variancia_previsao'])

        # Calculo do R2 para cada região
        try:
            if np.sum(distance_avg_values) == 0 and np.sum(distance_predicted_values) == 0:
                adjusted_r2 = 1
            else:
                adjusted_r2 = 1 - ((np.sum(distance_predicted_values)/df_region.shape[0])/\
                                   (np.sum(distance_avg_values)/df_region_all_data.shape[0]))
                if str(adjusted_r2) == "-inf":
                    adjusted_r2 = 1 - (df_region[target_variable+'_previsto'].sum()/df_region_all_data.shape[0])
        except Exception as ex:
            embed()

        temp = pd.DataFrame([[b, round(adjusted_r2, 2)]], columns=["nome_bairro", "R2"])

        temp['MAE'] = round(mean_absolute_error(list(df_region[target_variable]),
                                  list(df_region[target_variable + '_previsto'])))
        temp['RMSE'] = round(pow(mean_squared_error(list(df_region[target_variable]),
                                     list(df_region[target_variable + '_previsto'])), 1 / 2))

        temp['MAPE'] = epsilon_mape(df_region[target_variable],
                                            df_region[target_variable + '_previsto'])

        df_adjusted_r2 = df_adjusted_r2.append(temp)

    # df_adjusted_r2.sort_values(by=['r2'], inplace=True, ascending=False)

    return df_adjusted_r2


def apply_adjusted_r2(df_treino, df_teste, df_prev, target_variable='dengue_diagnosis'):
    '''
    Nesse caso o cálculo do denominador da formula considera a diferença com o valor médio para bairro, ou seja,
    no denominador é levado em consideração os dados de treino e teste

    Esse método calcula da mesma forma que o "apply_adjusted_r2_for_each_region_test_data_variance_all_historical_data",
    aqui ele só foi adaptado para o método que calcula r2 no tempo "analyse_r2_by_time_v2"
    '''

    # Junta as bases de treino e de teste
    df_teste = df_teste[df_teste["mes"].isin(list(df_prev['mes'].unique()))]
    df_merge = df_treino.append(df_teste)

    # Cria a lista que ira armazenar os valores para aplicaçao da somatoria
    distance_avg_values = list()
    df_merge.index = df_merge["chave"]

    # Caculando a variância geral (considerando dados de treino e teste) de cada região com a média
    for c in list(df_merge["chave"]):
        distance_avg_values.append(
            pow(df_merge.loc[c, target_variable]-np.mean(df_merge[target_variable]), 2))

    distance_predicted_values = list()

    df_prev.index = df_prev["chave"]
    # Calculo da variância para cada região com o resultado do modelo
    for c in list(df_prev["chave"]):
        distance_predicted_values.append(
                pow(df_merge.loc[c,  target_variable] - df_prev.loc[c,  target_variable+'_previsto'], 2))

    # Calculo do R2 para cada região
    if np.sum(distance_avg_values) == 0 and np.sum(distance_predicted_values) == 0:
        adjusted_r2 = 1
    else:
        adjusted_r2 = 1 - ((np.sum(distance_predicted_values)/df_prev.shape[0])/\
                           (np.sum(distance_avg_values)/df_merge.shape[0]))
        if str(adjusted_r2) == "-inf":
            adjusted_r2 = 1 - (df_prev[target_variable+'_previsto'].sum()/df_merge.shape[0])

    return round(adjusted_r2, 2)


def apply_adjusted_r2_all_base(df_treino, df_teste, target_variable='dengue_diagnosis'):
    '''
    Nesse caso o cálculo do denominador da formula considera a diferença com o valor médio para bairro, ou seja,
    no denominador é levado em consideração os dados de treino e teste

    Esse método calcula da mesma forma que o "apply_adjusted_r2",
    aqui ele só foi adaptado para o cálculo da base como um \rtodo. Não desagregando por bairro
    '''

    # Cria a lista que ira armazenar os valores para aplicaçao da somatoria
    distance_avg_values = list()
    df_treino.index = df_treino["chave"]
    df_treino = df_treino[df_treino["ano"] < df_teste["ano"].unique()[0]]

    # Caculando a variância geral (considerando dados de treino e teste) de cada região com a média
    for c in list(df_treino["chave"]):
        aux = df_treino[df_treino['nome_bairro'] == df_treino.loc[c, ['nome_bairro']]['nome_bairro']]
        distance_avg_values.append(
            pow(df_treino.loc[c, target_variable]-\
                np.mean(aux[target_variable]), 2))

    distance_predicted_values = list()

    df_teste.index = df_teste["chave"]
    # Calculo da variância para cada região com o resultado do modelo
    for c in list(df_teste["chave"]):
        distance_predicted_values.append(
                pow(df_teste.loc[c,  target_variable] - df_teste.loc[c,  target_variable+'_previsto'], 2))

    # Calculo do R2 para cada região
    if np.sum(distance_avg_values) == 0 and np.sum(distance_predicted_values) == 0:
        adjusted_r2 = 1
    else:
        adjusted_r2 = 1 - ((np.sum(distance_predicted_values)/df_teste.shape[0])/\
                           (np.sum(distance_avg_values)/df_treino.shape[0]))
        if str(adjusted_r2) == "-inf":
            adjusted_r2 = 1 - (df_teste[target_variable+'_previsto'].sum()/df_treino.shape[0])

    df = pd.DataFrame([[round(adjusted_r2, 2),
                        round(mean_absolute_error(list(df_teste[target_variable]),
                                                  list(df_teste[target_variable+'_previsto'])), 2),
                        round(pow(mean_squared_error(list(df_teste[target_variable]),
                                                  list(df_teste[target_variable+'_previsto'])), 1/2), 2)]],
                      columns=["R2", "MAE", "RMSE"])

    return df


def apply_r2_standart_for_each_city_test_data(df_valores_previstos, df_previsao_real,
                                              target_variable='dengue_diagnosis'):
    '''
    Calculate r2 of each city
    '''

    df_r2 = pd.DataFrame(columns=['city', 'r2'])

    for i, c in enumerate(list(df_previsao_real['nome_bairro'].unique())):
        df_aux = pd.DataFrame({'city': [c], 'r2': [r2_score(
            y_true=df_previsao_real[df_previsao_real['nome_bairro'] == c][target_variable],
            y_pred=df_valores_previstos[df_valores_previstos['nome_bairro'] == c][target_variable+'_previsto'])]})
        df_r2 = df_r2.append(df_aux)

    df_r2.sort_values(by=['r2'], inplace=True, ascending=False)

    return df_r2


