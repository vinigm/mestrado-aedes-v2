import pandas as pd
import numpy as np
import os
import re
from IPython import embed

base_path = os.getcwd()


def get_indices_columns_time(columns):

    columns_time_um = list()
    columns_time_tres = list()
    columns_time_seis = list()
    columns_others = list()

    for c in columns:
        if re.match(".*\(t-1\).*", c):
            columns_time_um.append(c)
        elif re.match(".*\(t-3\).*", c):
            columns_time_tres.append(c)
        elif re.match(".*\(t-6\).*", c):
            columns_time_seis.append(c)
        else:
            columns_others.append(c)

    columns_sequence = columns_others + columns_time_um + columns_time_tres + columns_time_seis

    index_time_cols_um = (len(columns_others), len(columns_others) + len(columns_time_um) - 1)
    index_time_cols_tres = (index_time_cols_um[1]+1,  index_time_cols_um[1] + len(columns_time_tres))
    index_time_cols_seis = (index_time_cols_tres[1] + 1, index_time_cols_tres[1] + len(columns_time_seis))

    return columns_sequence, index_time_cols_um, index_time_cols_tres, index_time_cols_seis


def network_neigboor(df, neighbor_columns):
    '''
    O método consolida as informações de vizinhança e casos
    :param path_input_read:
    :return: dataframe com colunas dos vizinhos de todos os bairros, cada posição de vizinhos tem
    a quantidade de casos de t-1, t-3 e t-6
    '''

    df_net = pd.read_excel(os.path.join(base_path, "data", "Pasta de trabalho.xlsx"),
                           sheet_name="Higienizado")
        # base_path + \
        # '/data_sus/input/Network_Bairros/Pasta de trabalho.xlsx',
        # sheet_name="Higienizado")

    columns_net = df['nome_bairro'].unique()

    df_net_geral = df.copy()

    # Cria a coluna de todas as regiões vizinhas
    for c in columns_net:
        df_net_geral[c + ' (t-1)'] = [np.nan] * df_net_geral.shape[0]
        if 'sum_vizinhos_t-3' in neighbor_columns:
            df_net_geral[c + ' (t-3)'] = [np.nan] * df_net_geral.shape[0]
        if 'sum_vizinhos_t-6' in neighbor_columns:
            df_net_geral[c + ' (t-6)'] = [np.nan] * df_net_geral.shape[0]

    df.index = df['nome_bairro']
    df_net_geral.index = df_net_geral['chave']

    # Cada linha do df_net possui a região na coluna Regiao e os respectivos vizinhos nas colunas seguintes
    for i in list(range(0, df_net.shape[0])):

        # connection é a linha que possui a região e os respectivos vizinhos
        connection = df_net.iloc[i, :]
        # c será a região de conexão, o range vai até 18 porque é o número máximo de vizinhos
        for c in list(range(1, 18)):

            # Se igual a zero significa que acabaram as regiões vizinhas e pode analisar ir para a próxima região
            if connection[c] == 0:
                break
            else:

                for a in list(df_net_geral['ano'].unique()):
                    for m in list(df_net_geral['mes'].unique()):
                        try:
                            aux = df[(df["ano"] == a) & (df["mes"] == m)]

                            # Para os casos em que nem todos os anos forem até 12 meses
                            if aux.shape[0] == 0:
                                continue

                            # Cria a chave para especificar a região, ano e mês que terá o valor inserido
                            # chave_regiao_origem = str(df_input.loc[connection['Regiao'], "cod_bairro"])\
                            #                       +str(a)+str(m)

                            chave_regiao_origem = str(aux.loc[connection['Regiao'], "cod_bairro"]) \
                                                  + str(a) + str(m)

                            # Busca a chave para a região de análise
                            # chave_regiao_conectada = str(df_input.loc[connection[c], "cod_bairro"])\
                            #                          +str(a)+str(m)
                            chave_regiao_conectada = str(aux.loc[connection[c], "cod_bairro"]) \
                                                     + str(a) + str(m)

                            # Na linha cidade conectada coloca os valores de t-1, soma até t-3 e soma até t-6
                            df_net_geral.loc[int(chave_regiao_origem), connection[c] + " (t-1)"] = np.sum(
                                df_net_geral.loc[
                                    int(chave_regiao_conectada), "t-1"])
                            if 'sum_vizinhos_t-3' in neighbor_columns:
                                df_net_geral.loc[int(chave_regiao_origem), connection[c] + " (t-3)"] = np.sum(
                                    df_net_geral.loc[
                                        int(chave_regiao_conectada), ["t-1", "t-2", "t-3"]])
                            if 'sum_vizinhos_t-6' in neighbor_columns:
                                df_net_geral.loc[int(chave_regiao_origem), connection[c] + " (t-6)"] = np.sum(
                                    df_net_geral.loc[
                                        int(chave_regiao_conectada), ["t-1", "t-2", "t-3", "t-4", "t-5", "t-6"]])
                        except Exception as ex:
                            embed()

    return df_net_geral


def sum_neighboors_to_prediction(df, neighbor_columns, taxa=False):
    '''
    Consolida as colunas de vizinhos nas colunas sum_vizinhos
    :param df_input:
    :return:
    '''
    if not taxa:

        df_input_net = network_neigboor(df, neighbor_columns)

        for c in list(df_input_net.columns):
            if re.match(".*Unname.*", c):
                df_input_net.drop(columns=[c], inplace=True)
            # elif c in ['min_vizinhos', 'max_vizinhos']:
            #     df_input.drop(columns=[c], inplace=True)

        columns_sequence, index_time_cols_um, index_time_cols_tres, index_time_cols_seis = \
            get_indices_columns_time(list(df_input_net.columns))

        df_input_net = df_input_net[columns_sequence]

        sum_cases_um = list()
        sum_cases_tres = list()
        sum_cases_seis = list()

        for i in list(range(df_input_net.shape[0])):
            sum_cases_um.append(np.sum(df_input_net.iloc[i, index_time_cols_um[0]:index_time_cols_um[1]]))
            if 'sum_vizinhos_t-3' in neighbor_columns:
                sum_cases_tres.append(np.sum(df_input_net.iloc[i, index_time_cols_tres[0]:index_time_cols_tres[1]]))
            if 'sum_vizinhos_t-6' in neighbor_columns:
                sum_cases_seis.append(np.sum(df_input_net.iloc[i, index_time_cols_seis[0]:index_time_cols_seis[1]]))

        df['sum_vizinhos_t-1'] = sum_cases_um
        if 'sum_vizinhos_t-3' in neighbor_columns:
            df['sum_vizinhos_t-3'] = sum_cases_tres
        if 'sum_vizinhos_t-6' in neighbor_columns:
            df['sum_vizinhos_t-6'] = sum_cases_seis

        df['sum_vizinhos_t-1'] = df['sum_vizinhos_t-1'].apply(
            lambda x: round(x) if re.match(".*[0-9].*", str(x)) else 0)

        if 'sum_vizinhos_t-3' in neighbor_columns:
            df['sum_vizinhos_t-3'] = df['sum_vizinhos_t-3'].apply(
                lambda x: round(x) if re.match(".*[0-9].*", str(x)) else 0)

        if 'sum_vizinhos_t-6' in neighbor_columns:
            df['sum_vizinhos_t-6'] = df['sum_vizinhos_t-6'].apply(
                lambda x: round(x) if re.match(".*[0-9].*", str(x)) else 0)

    # Não da certo utilizar isso aqui ainda, pois precisa modificar o método network_neigboor para calcular a
    # média dos t-n, porque nesse caso os t-n já são taxas por 100 habitantes
    # else:
    #
    #     df_input_net = network_neigboor_mean(df)
    #
    #     for c in list(df_input_net.columns):
    #         if re.match(".*Unname.*", c):
    #             df_input_net.drop(columns=[c], inplace=True)
    #         # elif c in ['min_vizinhos', 'max_vizinhos']:
    #         #     df_input.drop(columns=[c], inplace=True)
    #
    #     columns_sequence, index_time_cols_um, index_time_cols_tres, index_time_cols_seis = \
    #         get_indices_columns_time(list(df_input_net.columns))
    #
    #     df_input_net = df_input_net[columns_sequence]
    #
    #     avg_cases_um = list()
    #     avg_cases_tres = list()
    #     avg_cases_seis = list()
    #
    #     for i in list(range(df_input_net.shape[0])):
    #         avg_cases_um.append(np.mean(df_input_net.iloc[i, index_time_cols_um[0]:index_time_cols_um[1]]))
    #         avg_cases_tres.append(np.mean(df_input_net.iloc[i, index_time_cols_tres[0]:index_time_cols_tres[1]]))
    #         avg_cases_seis.append(np.mean(df_input_net.iloc[i, index_time_cols_seis[0]:index_time_cols_seis[1]]))
    #
    #     df['sum_vizinhos_t-1'] = avg_cases_um
    #     df['sum_vizinhos_t-3'] = avg_cases_tres
    #     df['sum_vizinhos_t-6'] = avg_cases_seis
    #
    #     df['sum_vizinhos_t-1'] = df['sum_vizinhos_t-1'].apply(
    #         lambda x: round(x, 2) if re.match(".*[0-9].*", str(x)) else 0)
    #     df['sum_vizinhos_t-3'] = df['sum_vizinhos_t-3'].apply(
    #         lambda x: round(x, 2) if re.match(".*[0-9].*", str(x)) else 0)
    #     df['sum_vizinhos_t-6'] = df['sum_vizinhos_t-6'].apply(
    #         lambda x: round(x, 2) if re.match(".*[0-9].*", str(x)) else 0)

    # df_input.to_csv(base_path + \
    #         '/data_sus/finais/input_ml_ocorrencia_doencas_v6.csv')

    return df[neighbor_columns]
