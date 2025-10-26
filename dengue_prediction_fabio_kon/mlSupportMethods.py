#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:36:31 2020

@author: robson
"""
import datetime

import pandas as pd
import numpy as np
import os
import re
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import pickle
from IPython import embed
from datetime import date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

from neighbors_methods import sum_neighboors_to_prediction
from mlMethods import run_lightGBM, run_catboost
from measuringResults import get_indices_columns_time
base_path = os.getcwd()


def epsilon_mape(y_true, y_pred, epsilon=1):
    '''
    y_true=np.array([3, -0.5, 2, 7])
    y_pred=np.array([2.5, 0.0, 2, 8])
    epsilon=1
    '''

    mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
    output_errors = np.average(mape, axis=0)

    return np.average(output_errors)


def identify_critical_cases_in_data_train_and_test(df_teste, df_treino, target):

    df_pivot = df_treino.pivot_table(index=['nome_bairro'],
                                          values=[target], aggfunc='sum').reset_index()

    scaler = MinMaxScaler(feature_range = (0,1))

    df_pivot['critical_neighbor'] = scaler.fit_transform(np.array(df_pivot[target]).reshape(-1, 1))
    df_pivot['critical_neighbor'] = df_pivot['critical_neighbor'].apply(lambda x: round(x, 2))

    df_teste = pd.merge(df_teste, df_pivot[['nome_bairro', 'critical_neighbor']],
                        on='nome_bairro', how='left')

    df_treino = pd.merge(df_treino, df_pivot[['nome_bairro', 'critical_neighbor']],
                        on='nome_bairro', how='left')

    return df_teste, df_treino


def run_main_first_step_1(target, ano_teste, dengue_columns, columns_filtered, categorical_columns,
                             columns_filtered_categorical, mes_inicio_teste=1, sample=False, n=0,
                             meses_previsao=6, log=False, quartil=False, classification=False):
    '''
    Step 1 of main process: prepare train and test bases
    '''
    if target == "taxa_dengue":
        df_dengue_origem = pd.read_csv(base_path + "/data/dengue_input_from_source_taxa_v5.csv")
    else:
        df_dengue_origem = pd.read_csv(base_path + "/data/dengue_input_from_source_v10.csv")

    if not classification == False:
        # Apply shift to predict months ahead
        df_shift = pd.DataFrame(columns=list(df_dengue_origem.columns))
        for n in list(df_dengue_origem['nome_bairro'].unique()):
            df = df_dengue_origem[df_dengue_origem['nome_bairro'] == n]
            df['dengue_diagnosis'] = df['dengue_diagnosis'].shift(classification+1)
            df_shift = df_shift.append(df)

        df_dengue_origem = df_shift.dropna()

    df_dengue_origem.drop(columns=['Unnamed: 0'], inplace=True)
    df_dengue_origem = df_dengue_origem[df_dengue_origem['ano'] > 2014]

    if log:
        for c in list(df_dengue_origem.columns):
            if c in dengue_columns:
                df_dengue_origem[c] = df_dengue_origem[c].replace(0, 1)
                df_dengue_origem[c] = np.log10(df_dengue_origem[c])

    df_treino, df_teste = base_prepare(df_dengue_origem, ano_teste=ano_teste,
                                              mes_inicio_teste=mes_inicio_teste, meses_previsao=meses_previsao)

    if quartil:
        df_teste, df_treino = identify_critical_cases_in_data_train_and_test(df_teste, df_treino, target)

    x_train, y_train = prepare_base_train(df_treino, columns_filtered, columns_filtered_categorical,
                            categorical_columns, target_variable=target)

    if sample:
        x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(x_train,
                                                                                        y_train,
                                                                                        test_size=0.1,
                                                                                        random_state=n)
    else:
        x_train_sample = x_train.copy()
        y_train_sample = y_train.copy()

    return df_treino, df_teste, x_train_sample, y_train_sample


def run_main_first_step_2(df_test, x_train, y_train, selected_columns, categorical_columns,
           columns_filtered_categorical, neighbor_columns, target, model, standart=False):
    '''
    Step 2 of main process: training model and predicting values
    '''

    if model in ["randomforest", "extratrees", "xgboost", "lightGBM"]:
        # regressor, r2 = globals()["run_{}".format(model)](X_train, y_train, grid_search=False)
        regressor, r2 = run_lightGBM(x_train, y_train, grid_search=False, standart=standart)
        scaler = False
    elif model in ['catboost']:
        regressor, r2 = run_catboost(x_train, y_train, grid_search=False, standart=standart)
        scaler = False
    elif model in ["polynomial_regression"]:
        regressor, scaler, r2 = globals()["run_{}".format(model)](x_train, y_train, grid_search=False)
    else:
        regressor, scaler, r2 = globals()["run_{}".format(model)](x_train, y_train, grid_search=False)
        scaler = False

    train_columns = list(x_train.columns).copy()
    df_output_model = apply_future_prediction_update_neighbors_v2(regressor, df_test,
                                                               selected_columns, columns_filtered_categorical,
                                                               train_columns, categorical_columns,
                                                               neighbor_columns, target=target,
                                                               scaler=scaler)

    df_test = pd.merge(df_test, df_output_model[["chave", target+'_previsto']], on="chave", how="left")

    return df_test, df_output_model, regressor


def prepare_base_test(df_previsao, columns_filtered, columns_filtered_categorical,
                            categorical_columns, train_columns, scaler):

    # O método recebe a base de teste com todas as colunas da base original
    # df_previsao_X é o df que será inserido no modelo para previsão
    df_previsao_X = df_previsao[columns_filtered]

    # Se existem variáveis categoricas então transforma em binário
    if not categorical_columns == []:
        df_previsao_X = pd.get_dummies(df_previsao_X, columns=categorical_columns)

    # para as variáveis categóricas filtra as variáveis e valores previamente secionados
    mes_list = ["mes_{}".format(i) for i in list(df_previsao["mes"].unique())]

    if not columns_filtered_categorical == []:

        # para o caso de análise de r2 no tempo
        if 'mes' in categorical_columns:
            mes_list = [m for m in mes_list if m in columns_filtered_categorical]

            categorical_list = [a for a in categorical_columns if not re.match(".*mes.*", a)]

            # Se houver mais variáveis categóricas além de mes
            if len(categorical_list) > 0:
                columns_filtered_categorical_new = categorical_list + mes_list
            else:
                columns_filtered_categorical_new = mes_list

        new_columns_features = [a for a in columns_filtered if a not in categorical_columns]
        df_previsao_X = df_previsao_X[new_columns_features + columns_filtered_categorical_new]

    # Caso o modelo foi treinado com meses que estão além dos meses na base de teste
    meses_ausentes = [m for m in train_columns if re.match(".*mes.*", m) and not m in mes_list]
    if len(meses_ausentes) > 0:
        for m in meses_ausentes:
            df_previsao_X[m] = [0] * df_previsao_X.shape[0]

    # Se o regressor exige valores normalizados
    if scaler:
        df_previsao_X = pd.DataFrame(scaler.transform(df_previsao_X),
                                     columns=df_previsao_X.columns)

    columns, index_time_cols = get_indices_columns_time(df_previsao_X.columns)
    df_previsao_X = df_previsao_X[columns]

    # Precisa dessas variáveis para percorrer na previsão
    df_previsao_X['ano'] = [0] * df_previsao_X.shape[0]
    df_previsao_X['mes'] = [0] * df_previsao_X.shape[0]
    df_previsao_X['nome_bairro'] = [0] * df_previsao_X.shape[0]
    df_previsao_X['chave'] = [0] * df_previsao_X.shape[0]
    df_previsao_X['cod_bairro'] = [0] * df_previsao_X.shape[0]

    # Precisa dessas variáveis para percorrer na previsão
    df_previsao_X.loc[:,['ano']] = list(df_previsao['ano'])
    df_previsao_X.loc[:,['mes']] = list(df_previsao['mes'])
    df_previsao_X.loc[:,['nome_bairro']] = list(df_previsao['nome_bairro'])
    df_previsao_X.loc[:,['chave']] = list(df_previsao['chave'])
    df_previsao_X.loc[:,['cod_bairro']] = list(df_previsao['cod_bairro'])

    return df_previsao_X, columns


def prepare_base_train(df_treino, columns_filtered, columns_filtered_categorical,
                            categorical_columns, target_variable='dengue_diagnosis'):

    X_train = df_treino[columns_filtered]
    y_train = df_treino[target_variable]

    X_train = pd.get_dummies(X_train, columns=categorical_columns)

    # Para o caso de haver variáveis categoricas que serão selecionadas
    if not columns_filtered_categorical == []:

        # Tratativa especial para o caso de mês
        if 'mes' in categorical_columns:

            # Irá buscar somente os meses que estão na base de treino, pode ir de 1 a 12
            mes_list = ["mes_{}".format(i) for i in list(df_treino["mes"].unique())]
            mes_list = [m for m in mes_list if m in columns_filtered_categorical]
            categorical_list = [a for a in categorical_columns if not re.match(".*mes.*", a)]

            # Se na lista de variáveis categoricas não tivessem somente o mês
            if len(categorical_list) > 0:
                columns_filtered_categorical_new = categorical_list + mes_list
            else:
                columns_filtered_categorical_new = mes_list

        new_columns_features = [a for a in columns_filtered if a not in categorical_columns]
        X_train = X_train[new_columns_features + columns_filtered_categorical_new]

        meses_ausentes = [m for m in columns_filtered_categorical if re.match(".*mes.*", m) and not m in mes_list]
        if len(meses_ausentes) > 0:
            for m in meses_ausentes:
                X_train[m] = [0] * X_train.shape[0]


    columns, index_time_cols = get_indices_columns_time(X_train.columns)
    X_train = X_train[columns]

    return X_train, y_train



def apply_future_prediction_update_neighbors(regressor, df_test, selected_columns,
                                             columns_filtered_categorical, train_columns,
                                             categorical_columns=[], neighbor_columns=[],
                                             target='dengue_diagnosis', scaler=False):

    df_previsao_X, columns = prepare_base_test(df_test, selected_columns, columns_filtered_categorical,
                            categorical_columns, train_columns, scaler)
    df_previsao_X['date'] = df_previsao_X[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)

    list_results = list()
    df_result_consolidate = pd.DataFrame(
        columns=list(df_previsao_X.columns)+[target+"_previsto"])
    df_result_consolidate.drop(columns=['date'], inplace=True)

    for d in list(df_previsao_X['date'].unique()):
        try:
            # Filtra o ano e mes de forma sequencial
            df = df_previsao_X[df_previsao_X['date'] == d]
            df.drop(columns=['date'], inplace=True)

            # Se não for o primeiro, então precisa mover os casos para os
            # meses anteriores e recalcular os vizinhos
            if len(list_results) > 0:

                # Atribui os valores que já foram previstos para o novo df
                for n in list(range(1, len(list_results)+1)):
                    df['t-{}'.format(n)] = list_results[-n]

                # Apaga as colunas dos vizinhos para calcular novamente
                if not neighbor_columns == []:
                    for n in neighbor_columns:
                        df.drop(columns=[n], inplace=True)

                    # Ajustar soma casos dos vizinhos
                    df_neighbors = sum_neighboors_to_prediction(df.copy(), neighbor_columns)
                    for n in neighbor_columns:
                        df[n] = list(df_neighbors[n])

            df_prev = df.drop(columns=['nome_bairro', 'chave', 'ano', 'mes', 'cod_bairro'])
            df_prev = df_prev[columns]

            # Faz a previsão e armazena na lista de suporte e no df de resultado final
            list_results.append(regressor.predict(df_prev))
            if target == 'dengue_diagnosis':
                list_results[-1] = [0 if a < 0 else round(a) for a in list_results[-1]]
            df[target+"_previsto"] = list_results[-1]
            df_result_consolidate = df_result_consolidate.append(df)

        except Exception as ex:
            print("########### apply_future_prediction_update_neighbors ############")
            embed()

    # Inverte novamente as colunas categóricas
    for cat in categorical_columns:
        for c in df_test[cat].unique():
            if cat + '_' + str(c) in list(df_result_consolidate.columns):
                df_result_consolidate.drop(columns=[cat + '_' + str(c)], inplace=True)

    df_result_consolidate.sort_values(by=["cod_bairro", "ano", "mes"], inplace=True)

    return df_result_consolidate


def apply_future_prediction_update_neighbors_v2(regressor, df_test, selected_columns,
                                             columns_filtered_categorical, train_columns,
                                             categorical_columns=[], neighbor_columns=[],
                                             target='dengue_diagnosis', scaler=False):
    '''
    d = list(df_previsao_X['date'].unique())[0]
    '''

    df_previsao_X, columns = prepare_base_test(df_test, selected_columns, columns_filtered_categorical,
                            categorical_columns, train_columns, scaler)
    df_previsao_X['date'] = df_previsao_X[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)

    list_results = list()
    df_result_consolidate = pd.DataFrame(
        columns=list(df_previsao_X.columns)+[target+"_previsto"])
    df_result_consolidate.drop(columns=['date'], inplace=True)

    for d in list(df_previsao_X['date'].unique()):
        # for num_pred in list(range(len(list(df_previsao_X['date'].unique())))):
            try:
                # Filtra o ano e mes que será previsto
                df_base = df_previsao_X[df_previsao_X['date'] == d]

                # Filtra mês em que a previsão vai começar
                df = df_previsao_X[df_previsao_X['date'] == list(df_previsao_X['date'].unique())[0]]
                df.drop(columns=['date'], inplace=True)

                # Se não for o primeiro, então precisa mover os casos para os
                # meses anteriores e recalcular os vizinhos
                if len(list_results) > 0:

                    # Atribui os valores que já foram previstos para o novo df
                    for n in list(range(1, len(list_results)+1)):
                        df['t-{}'.format(n)] = list_results[-n]

                    # Apaga as colunas dos vizinhos para calcular novamente
                    if not neighbor_columns == []:
                        for n in neighbor_columns:
                            df.drop(columns=[n], inplace=True)

                        # Ajustar soma casos dos vizinhos
                        df_neighbors = sum_neighboors_to_prediction(df.copy(), neighbor_columns)
                        for n in neighbor_columns:
                            df[n] = list(df_neighbors[n])

                df_prev = df.drop(columns=['nome_bairro', 'chave', 'ano', 'mes', 'cod_bairro'])
                df_prev = df_prev[columns]

                # Faz a previsão e armazena na lista de suporte e no df de resultado final
                list_results.append(regressor.predict(df_prev))
                if target == 'dengue_diagnosis':
                    list_results[-1] = [0 if a < 0 else round(a) for a in list_results[-1]]
                df_base[target+"_previsto"] = list_results[-1]
                df_result_consolidate = df_result_consolidate.append(df_base)

            except Exception as ex:
                print("########### apply_future_prediction_update_neighbors ############")
                embed()

    # Inverte novamente as colunas categóricas
    for cat in categorical_columns:
        for c in df_test[cat].unique():
            if cat + '_' + str(c) in list(df_result_consolidate.columns):
                df_result_consolidate.drop(columns=[cat + '_' + str(c)], inplace=True)

    df_result_consolidate.sort_values(by=["cod_bairro", "ano", "mes"], inplace=True)

    return df_result_consolidate



def base_prepare(df_dengue_origem, ano_teste=2019,
                        mes_inicio_teste=1,
                        meses_previsao=6):

    df_dengue_origem['data'] = df_dengue_origem[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)

    list_data_teste = list()
    for m in list(range(mes_inicio_teste, mes_inicio_teste+meses_previsao)):
        if m < 13:
            list_data_teste.append(date(ano_teste, m, 1))
        else:
            list_data_teste.append(date(ano_teste+1, m-12, 1))

    # Filter dates which is in test data list
    df_teste = df_dengue_origem[df_dengue_origem['data'].isin(list_data_teste)]

    # Filter dates which is not in test data list
    df_treino = df_dengue_origem[~df_dengue_origem['data'].isin(list_data_teste)]

    df_treino.drop(columns=['data'], inplace=True)
    df_teste.drop(columns=['data'], inplace=True)

    return df_treino, df_teste


