import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import shap
import numpy as np
import seaborn as sns
import os
from datetime import date
from sklearn import metrics
import catboost as cb
from IPython import embed
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, f1_score, recall_score, precision_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
import dateutil
import geopandas as gpd
<<<<<<< HEAD
# import geoplot
=======
import geoplot
>>>>>>> 6007da3035542093d3e826527fd6de8a52a11b69
from matplotlib.colors import LogNorm

from measuringResults import *
from mlSupportMethods import *
from mlMethods import *


selected_columns = ['qtd_cnes', 'precipitacao (mm)-1', 'temperatura (°C)-1',
       'umidade ar (%)-1', 'sum_vizinhos_t-1',
       't-1', 't-2', 't-3', 'densidade_demografica',
       'zika-1', 'chikungunya-1', 'critical_neighbor', 'liraa']

base_path = os.getcwd()
dengue_columns = ['t-1', 't-2', 't-3', 'sum_vizinhos_t-1', 'dengue_diagnosis']
categorical_columns = []
columns_filtered_categorical = []
neighbor_columns=['sum_vizinhos_t-1']
log_flag = False
target = 'dengue_diagnosis'
year_begin = 2016
year_end = 2020
list_year = list(range(year_begin, year_end+1))
model = "catboost"
standart = False
months = list(range(1, 13))
months_ahead_predict = 3
file_name = f"{year_begin}_{year_end}_{months_ahead_predict}_months"
fontsize_title = 15
fontsize_label = 14
dpi = 180


def build_model_prediction(months_ahead):
    '''
    Build dataframe, calculating predictions through diferents initial months
    :return:
    a = 2016
    initial_test_month = 1
    months_ahead = 3
    '''
    variables = ['nome_bairro', 'ano', 'mes_inicial', 'mes', target, target + '_previsto']
    variables.extend(selected_columns)
    df_consolidate = pd.DataFrame(columns=variables)

    for a in list_year:
        for initial_test_month in months:
            try:
                if (a == 2020 and initial_test_month > 12 - months_ahead) or \
                        (a == 2020 and initial_test_month in [10, 11, 12]):
                    break

                df_train, df_test, x_train, y_train = run_main_first_step_1(target,
                                                                               ano_teste=a,
                                                                               log=log_flag,
                                                                               mes_inicio_teste=initial_test_month,
                                                                               meses_previsao=months_ahead,
                                                                               dengue_columns=dengue_columns,
                                                                               columns_filtered=selected_columns,
                                                                               categorical_columns=categorical_columns,
                                                                               columns_filtered_categorical=columns_filtered_categorical,
                                                                               quartil=True)

                df_result, df_output_model, regressor = run_main_first_step_2(df_test, x_train, y_train,
                                                                              selected_columns,
                                                                              categorical_columns,
                                                                              columns_filtered_categorical,
                                                                              neighbor_columns, target, model,
                                                                              standart=standart)

                df_result['mes_inicial'] = [initial_test_month] * df_result.shape[0]

                df_consolidate = df_consolidate.append(
                    df_result[variables])

            except Exception as ex:
                embed()

    return df_consolidate


def build_baseline_prediction(months_ahead):

    df_input = pd.read_csv(base_path + "/data/dengue_input_from_source_v10.csv")

    order = (2, 0, 0)
    seasonal_order = (1, 0, 0, 12)
    trend = 'n'

    hist_window = dateutil.relativedelta.relativedelta(months=48)
    time_prediction = dateutil.relativedelta.relativedelta(months=months_ahead)

    df_input['data'] = df_input['data'].apply(lambda x: date(int(str(x).split('-')[0]),
                                                             int(str(x).split('-')[1]),
                                                             int(str(x).split('-')[2])))

    columns = list(df_input.columns)
    columns.extend(['dengue_diagnosis_previsto'])
    columns.extend(['mes_inicial'])
    df_res_consolidate = pd.DataFrame(
        columns=columns)

    for b in list(df_input['nome_bairro'].unique()):
        for year in list_year:
            for initial_month in list(range(1, 13)):
                if year == 2020 and initial_month > 8:
                    continue
                try:
                    df = df_input[df_input['nome_bairro'] == b]

                    data_hist_begin = date(year, initial_month, 1) - hist_window

                    df_train = df[(df['data'] >= data_hist_begin) & (
                            df['data'] < date(year, initial_month, 1))]

                    df_test = df[(df['data'] >= date(year, initial_month, 1)) & \
                                 (df['data'] < date(year, initial_month, 1) + time_prediction)]
                    y_train = list(df_train[target])
                    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                                    trend=trend, simple_differencing=True,
                                    enforce_invertibility=False, enforce_stationarity=False)
                    model_fit = model.fit(disp=False)
                    df_test['dengue_diagnosis_previsto'] = [
                        int(n) if n > 0 else 0 for n in model_fit.forecast(months_ahead)]
                    df_test['mes_inicial'] = [initial_month] * df_test.shape[0]
                    df_res_consolidate = df_res_consolidate.append(df_test)

                except Exception as ex:
                    embed()

    return df_res_consolidate


def apply_df_percentile_95_99(df_in):


    # Consolidating different predictions of differents initial months
    df_in = df_in.pivot_table(index=['nome_bairro', 'ano', 'mes'],
                                                values=[target, target + '_previsto'],
                                                aggfunc='mean').reset_index()

    df_result = pd.DataFrame(columns=['nome_bairro', 'ano', 'mes',
                                      'dengue_diagnosis', 'dengue_diagnosis_previsto',
                                      'grupo_real', 'grupo_previsto'])

    for a in list_year:
        # Select years used in train dataset
        filter_years = list_year.copy()
        filter_years.remove(a)

        # Put percentiles in the dataframe
        df = df_in[~df_in['ano'].isin(filter_years)]
        df_train = df_in[df_in['ano'].isin(filter_years)]

        group_1 = np.percentile(df_train['dengue_diagnosis'], 95)
        group_2 = np.percentile(df_train['dengue_diagnosis'], 99)

        # print(f"Ano: {a}. Percentil 95: {round(group_1)}. Percentil 99:{round(group_2)}")

        df['grupo_real'] = df['dengue_diagnosis'].apply(lambda x: 1 if x <= group_1 else \
            2 if x <= group_2 else 3)
        df['grupo_previsto'] = df['dengue_diagnosis_previsto'].apply(lambda x: 1 if x <= group_1 else \
            2 if x <= group_2 else 3)

        # Consolidate the result
        df_result = df_result.append(df)

    return df_result


def calculate_regression_metrics(df):

    df['date'] = df[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)

    df_model = df[['nome_bairro', 'date', target, target+'_previsto']]

    # Need to do that because the results is in differents initial months
    df_pivot = df_model.pivot_table(index=['nome_bairro', 'date'],
                                    values=[target, target+'_previsto'],
                                    aggfunc='mean').reset_index()

    df_pivot.columns = ['nome_bairro', 'date', 'Real', 'Predicted']

    df_metrics = pd.DataFrame(columns=['neighbor', 'MAE', 'RMSE', 'MAPE', 'R2'])

    for b in list(df_pivot['nome_bairro'].unique()):
        df_b = df_pivot[df_pivot['nome_bairro'] == b]
        mae = round(mean_absolute_error(list(df_b['Real']),
                                        list(df_b['Predicted'])))
        rmse = round(pow(mean_squared_error(list(df_b['Real']),
                                            list(df_b['Predicted'])), 1 / 2))
        mape = round(epsilon_mape(df_b['Real'], df_b['Predicted']), 2)
        r2 = round(r2_score(df_b['Real'], df_b['Predicted']), 2)

        df_metrics = df_metrics.append(pd.DataFrame([[b, mae, rmse, mape, r2]],
                                                    columns=['neighbor', 'MAE', 'RMSE', 'MAPE', 'R2']))

    return df_pivot, df_metrics


def generate_results_model_and_baseline():

    version = '_v4'

    months_ahead = 1
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_1 = build_model_prediction(months_ahead)
    df_model_1.to_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")

    df_baseline_1 = build_baseline_prediction(months_ahead)
    df_baseline_1.to_csv(base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")

    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_3 = build_model_prediction(months_ahead)
    df_model_3.to_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")

    df_baseline_3 = build_baseline_prediction(months_ahead)
    df_baseline_3.to_csv(base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")


def histogram_analysis():

    df_input = pd.read_csv(base_path + "/data/dengue_input_from_source_v10.csv")
    df_input = df_input[df_input['ano'] > 2015]
    # df_input = df_input[df_input[target] < 800]

    f, ax = plt.subplots(figsize=(8, 6), dpi=dpi)

    df_input[target].hist(ax=ax, color='pink')

    ax.set_title("Dengue Cases Distribution (2016 - 2020)", fontsize=fontsize_title)
    ax.set_yscale('log')
    ax.set_xticklabels([0, 100, 200, 300, 400, 500, 600, 700, 800], fontsize=fontsize_label)
    plt.vlines(x=np.percentile(df_input[target], 95),
               ymin=0, ymax=np.log(df_input.shape[0])*1500,
               lw=3, ls='--', colors='green', label=f"Percentile 95: {round(np.percentile(df_input[target], 95))}")
    plt.vlines(x=np.percentile(df_input[target], 99),
               ymin=0, ymax=np.log(df_input.shape[0])*1500,
               lw=3, ls='--', colors='blue', label=f"Percentile 99: {round(np.percentile(df_input[target], 99))}")
    plt.legend(loc='upper right', fontsize=fontsize_label)
    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_0_histogram.eps',
                dpi=dpi)
    plt.show()


def confusion_matrix():

    def pivot_to_plot(df):

        pivot = df.pivot_table(index=['grupo_real'], columns=['grupo_previsto'],
                                      values=['nome_bairro'], aggfunc='count')

        pivot.columns = [str(j) for j in list(range(1, pivot.shape[1] + 1))]

        pivot.fillna(0, inplace=True)

        return pivot

    def plot_heatmap(pivot, ax, i, model, mh, fontsize, fontsize_title, abs=True):

        if abs:
            sns.heatmap(pivot, annot=True, fmt='d', linewidths=.5, annot_kws={'fontsize':fontsize},
                        norm=LogNorm(), ax=ax[i])
        else:
            pivot_perc = pivot.copy()

            for p in list(pivot.columns):
                for n in list(range(pivot.shape[0])):
                    pivot_perc.iloc[n, int(p) - 1] = round(pivot.iloc[n, int(p) - 1] / np.sum(pivot.iloc[n, :]), 2)

            sns.heatmap(pivot_perc, annot=True, linewidths=.5, ax=ax[i], annot_kws={'fontsize': fontsize})

        if mh == 3:
            ax[i].set_title(f"{mh} months ({model})", fontsize=fontsize_title)
        else:
            ax[i].set_title(f"{mh} month ({model})", fontsize=fontsize_title)
        if i == 0 or i == 2:
            ax[i].set_ylabel("Real Group", fontsize=fontsize)
            ax[i].set_yticklabels(["No", "Mild", "Severe"], fontsize=fontsize)
        else:
            ax[i].set_yticklabels(["", "", ""], fontsize=fontsize)
            ax[i].set_ylabel("")

        if i == 0 or i == 1:
            ax[i].set_xlabel("")
            ax[i].set_xticklabels(["", "", ""], fontsize=fontsize)
        else:
            ax[i].set_xlabel("Predicted Group", fontsize=fontsize)
            ax[i].set_xticklabels(["No", "Mild", "Severe"], fontsize=fontsize)

    def plot_all(df_percentil_model_1, df_percentil_model_3, df_percentil_baseline_1, df_percentil_baseline_3):

        dpi = 180

        pivot_model_1 = pivot_to_plot(df_percentil_model_1)
        pivot_model_3 = pivot_to_plot(df_percentil_model_3)
        pivot_baseline_1 = pivot_to_plot(df_percentil_baseline_1)
        pivot_baseline_3 = pivot_to_plot(df_percentil_baseline_3)

        f, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=dpi)
        ax = ax.flatten()
        plot_heatmap(pivot_model_1, ax, 0, 'Model', 1, fontsize_label, fontsize_title)
        plot_heatmap(pivot_baseline_1, ax, 1, 'Baseline', 1, fontsize_label, fontsize_title)
        plot_heatmap(pivot_model_3, ax, 2, 'Model', 3, fontsize_label, fontsize_title)
        plot_heatmap(pivot_baseline_3, ax, 3, 'Baseline', 3, fontsize_label, fontsize_title)
        plt.tight_layout()
        plt.savefig(base_path + '/data/figure_1_matrix_abs.eps',
                    dpi=dpi)
        plt.show()

        f, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=dpi)
        ax = ax.flatten()
        plot_heatmap(pivot_model_1, ax, 0, 'Model', 1, fontsize_label, fontsize_title, False)
        plot_heatmap(pivot_baseline_1, ax, 1, 'Baseline', 1, fontsize_label, fontsize_title, False)
        plot_heatmap(pivot_model_3, ax, 2, 'Model', 3, fontsize_label, fontsize_title, False)
        plot_heatmap(pivot_baseline_3, ax, 3, 'Baseline', 3, fontsize_label, fontsize_title, False)
        plt.tight_layout()
        plt.savefig(base_path + '/data/figure_1_matrix_perc.eps',
                    dpi=dpi)
        plt.show()

    version = '_v3'

    months_ahead = 1
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_1 = pd.read_csv(base_path + \
                             f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_1 = pd.read_csv(base_path + \
                                f"/data/df_baseline_prediction_{file_name}{version}.csv")

    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_3 = pd.read_csv(base_path + \
                             f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_3 = pd.read_csv(base_path + \
                                f"/data/df_baseline_prediction_{file_name}{version}.csv")


    df_percentil_model_1 = apply_df_percentile_95_99(df_model_1)
    df_percentil_baseline_1 = apply_df_percentile_95_99(df_baseline_1)

    df_percentil_model_3 = apply_df_percentile_95_99(df_model_3)
    df_percentil_baseline_3 = apply_df_percentile_95_99(df_baseline_3)

    plot_all(df_percentil_model_1, df_percentil_model_3, df_percentil_baseline_1, df_percentil_baseline_3)


def assessment_regression_metrics():

    '''
    Atenção aqui no cálculo do MAE, RMSE, MAPE e R2
    Colocar todas as métricas no mesmo gráfico
    :return:
    '''

    def visualizating_regression_results_boxplot_v2(df):

        dpi = 180

        f, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=dpi)
        ax = ax.flatten()
        sns.boxplot(x='Model', y='MAE', data=df, hue='Category', ax=ax[0])
        ax[0].set_title('MAE', fontsize=fontsize_title)
        ax[0].set_xlabel('')
        ax[0].set_ylabel('MAE', fontsize=fontsize_label)
        ax[0].set_yscale('log')
        ax[0].legend(loc='upper left', fontsize=fontsize_label)
        ax[0].set_xticklabels(['1 month', '3 months'], fontsize=fontsize_label)

        sns.boxplot(x='Model', y='RMSE', data=df, hue='Category', ax=ax[1])
        ax[1].set_title('RMSE', fontsize=fontsize_title)
        ax[1].set_xlabel('')
        ax[1].set_ylabel('RMSE', fontsize=fontsize_label)
        ax[1].set_yscale('log')
        ax[1].legend(loc='upper left', fontsize=fontsize_label)
        ax[1].set_xticklabels(['1 month', '3 months'], fontsize=fontsize_label)

        sns.boxplot(x='Model', y='MAPE', data=df, hue='Category', ax=ax[2])
        ax[2].set_title('MAPE', fontsize=fontsize_title)
        ax[2].set_xlabel('')
        ax[2].set_ylabel('MAPE', fontsize=fontsize_label)
        ax[2].set_yscale('log')
        ax[2].legend(loc='upper left', fontsize=fontsize_label)
        ax[2].set_xticklabels(['1 month', '3 months'], fontsize=fontsize_label)

        #df_r2_without_outlier = df[df['R2'] > -10]
        #sns.boxplot(x='Model', y='R2', data=df_r2_without_outlier, hue='Category', ax=ax[3])
        sns.boxplot(x='Model', y='R2', data=df, hue='Category', ax=ax[3])
        ax[3].set_title('R2', fontsize=fontsize_title)
        ax[3].set_xlabel('')
        ax[3].set_ylabel('R2', fontsize=fontsize_label)
        ax[3].legend(loc='lower left', fontsize=fontsize_label)
        ax[3].set_xticklabels(['1 month', '3 months'], fontsize=fontsize_label)
        ax[3].set_ylim(-5, 1)
        plt.tight_layout()
        plt.savefig(base_path + '/data/figure_2_boxplot_model_vs_baseline.eps',
                    dpi=dpi)

        plt.show()

    def regression_metrics_calculation(df, model, category):

        df_metrics = pd.DataFrame(columns=['mes_inicial', 'MAE', 'RMSE', 'MAPE', 'R2'])

        for m in months:

            df_calc = df[df['mes_inicial'] == m]
            mae = round(mean_absolute_error(list(df_calc[target]), list(df_calc[target + '_previsto'])))
            rmse = round(pow(mean_squared_error(list(df_calc[target]), list(df_calc[target + '_previsto'])), 1 / 2))
            mape = epsilon_mape(df_calc[target], df_calc[target + '_previsto'])
            r2 = r2_score(df_calc[target], df_calc[target + '_previsto'])

            df_metrics = df_metrics.append(pd.DataFrame([[m, mae, rmse, mape, r2]],
                                                        columns=['mes_inicial', 'MAE', 'RMSE', 'MAPE', 'R2']))

        df_metrics['Model'] = [model] * df_metrics.shape[0]
        df_metrics['Category'] = [category] * df_metrics.shape[0]

        return df_metrics

    def plot_table(df, ax, title):

        ax.axis('off')
        ax.axis('tight')
        ax.set_title(title, fontsize=fontsize_title)
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)

    version = '_v3'
    months_ahead = 1
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_1 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_1 = pd.read_csv(base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")

    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_3 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_3 = pd.read_csv(base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")

    df_model_1 = regression_metrics_calculation(df_model_1, "1 Month", "Model")
    df_baseline_1 = regression_metrics_calculation(df_baseline_1, "1 Month", "Baseline")

    df_model_3 = regression_metrics_calculation(df_model_3, "3 Months", "Model")
    df_baseline_3 = regression_metrics_calculation(df_baseline_3, "3 Months", "Baseline")

    df_append = df_model_1.append(df_baseline_1)
    df_append = df_append.append(df_model_3)
    df_append = df_append.append(df_baseline_3)

    visualizating_regression_results_boxplot_v2(df_append)

    df_model_perc_1 = pd.DataFrame()
    df_baseline_perc_1 = pd.DataFrame()
    df_model_perc_3 = pd.DataFrame()
    df_baseline_perc_3 = pd.DataFrame()

    for i in [0, 25, 50, 75, 100]:

        df_model_perc_1 = df_model_perc_1.append(pd.DataFrame([[i,
                                                    round(np.percentile(df_model_1['MAE'], i)),
                                                   round(np.percentile(df_model_1['RMSE'], i)),
                                                   round(np.percentile(df_model_1['MAPE'], i),2),
                                                   round(np.percentile(df_model_1['R2'], i),2)]],
                                                 columns=['Percentil', 'MAE', 'RMSE', 'MAPE', 'R2']))
        df_baseline_perc_1 = df_baseline_perc_1.append(pd.DataFrame([[i,
                                                    round(np.percentile(df_baseline_1['MAE'], i)),
                                                   round(np.percentile(df_baseline_1['RMSE'], i)),
                                                   round(np.percentile(df_baseline_1['MAPE'], i),2),
                                                   round(np.percentile(df_baseline_1['R2'], i),2)]],
                                                 columns=['Percentil', 'MAE', 'RMSE', 'MAPE', 'R2']))
        df_model_perc_3 = df_model_perc_3.append(pd.DataFrame([[i,
                                                    round(np.percentile(df_model_3['MAE'], i)),
                                                   round(np.percentile(df_model_3['RMSE'], i)),
                                                   round(np.percentile(df_model_3['MAPE'], i),2),
                                                   round(np.percentile(df_model_3['R2'], i),2)]],
                                                 columns=['Percentil', 'MAE', 'RMSE', 'MAPE', 'R2']))
        df_baseline_perc_3 = df_baseline_perc_3.append(pd.DataFrame([[i,
                                                    round(np.percentile(df_baseline_3['MAE'], i)),
                                                   round(np.percentile(df_baseline_3['RMSE'], i)),
                                                   round(np.percentile(df_baseline_3['MAPE'], i),2),
                                                   round(np.percentile(df_baseline_3['R2'], i),2)]],
                                                 columns=['Percentil', 'MAE', 'RMSE', 'MAPE', 'R2']))

    dpi=360

    fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=dpi)
    fig.patch.set_visible(False)
    ax = ax.flatten()
    plot_table(df_model_perc_1, ax[0], "Model (1 month)")
    plot_table(df_model_perc_3, ax[1], "Model (3 months)")
    fig.tight_layout()
    plt.savefig(base_path + f'/data/figure_2_table_part_1.eps',
                dpi=dpi)
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(4, 3), dpi=dpi)
    fig.patch.set_visible(False)
    ax = ax.flatten()
    plot_table(df_baseline_perc_1, ax[0], "Baseline (1 month)")
    plot_table(df_baseline_perc_3, ax[1], "Baseline (3 months)")
    fig.tight_layout()
    plt.savefig(base_path + f'/data/figure_2_table_part_2.eps',
                dpi=dpi)
    plt.show()


def time_series_analysis_model_vs_baseline():

    def prepare_to_plot(df_model, df_baseline):

        df_model['date'] = df_model[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)
        df_model.sort_values(by=['nome_bairro', 'date'], inplace=True)
        df_baseline['date'] = df_baseline[['ano', 'mes']].apply(lambda x: date(x[0], x[1], 1), axis=1)

        df_m_cons = pd.DataFrame(columns=['date', 'Real', 'Model', 'Baseline'])

        for m in months:
            df_m = df_model[df_model['mes_inicial'] == m]
            df_m = df_m.pivot_table(index=['date'], values=[target, target + '_previsto'], aggfunc='sum').reset_index()
            df_m.sort_values(by=['date'], inplace=True)
            df_b = df_baseline[df_baseline['mes_inicial'] == m]
            df_b = df_b.pivot_table(index=['date'], values=[target + '_previsto'], aggfunc='sum').reset_index()
            df_b.sort_values(by=['date'], inplace=True)
            df_m.columns = ['date', 'Real', 'Model']
            df_m['Baseline'] = df_b[target + '_previsto']
            df_m_cons = df_m_cons.append(df_m)

        df_m_cons.sort_values(by=['date'], inplace=True)

        df = pd.melt(df_m_cons, id_vars=['date'], value_vars=['Real', 'Model', 'Baseline'])
        df.columns = ['date', 'Category', 'Dengue Cases']
        df['Dengue Cases'] = df['Dengue Cases'].astype(float)

        df = df.pivot_table(index=['date', 'Category'], values=['Dengue Cases'], aggfunc='mean').reset_index()

        return df

    version = '_v3'
    dpi = 180
    months_ahead = 1
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_1 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_1 = pd.read_csv(base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")

    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_model_3 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_baseline_3 = pd.read_csv(
        base_path + f"/data/df_baseline_prediction_{file_name}{version}.csv")

    df_1 = prepare_to_plot(df_model_1, df_baseline_1)
    df_3 = prepare_to_plot(df_model_3, df_baseline_3)

    df_1.sort_values(by=['Category', 'date'], inplace=True)
    df_3.sort_values(by=['Category', 'date'], inplace=True)

    f, ax = plt.subplots(2, 1, figsize=(8, 6), dpi=dpi)
    ax = ax.flatten()
    sns.lineplot(x="date", y='Dengue Cases', hue='Category', linewidth=1.5, data=df_1, ax=ax[0])
    ax[0].set_title(f"Prediction Analysis - Model vs Real (1 month)", fontsize=fontsize_title)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Dengue Cases', fontsize=fontsize_label)
    ax[0].legend(loc='upper right', fontsize=fontsize_label)
    ax[0].set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020], fontsize=fontsize_label)

    sns.lineplot(x="date", y='Dengue Cases', hue='Category', linewidth=1.5, data=df_3, ax=ax[1])
    ax[1].set_title(f"Prediction Analysis - Model vs Real (3 months)", fontsize=fontsize_title)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('Dengue Cases', fontsize=fontsize_label)
    ax[1].legend(loc='upper right', fontsize=fontsize_label)
    ax[1].set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020], fontsize=fontsize_label)

    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_3_time_series_general_comparison.eps',
                dpi=dpi)
    plt.show()

    df_b_1 = df_baseline_1.pivot_table(index=['ano', 'mes'], values=['dengue_diagnosis_previsto'], aggfunc='mean').reset_index()
    df_b_3 = df_baseline_3.pivot_table(index=['ano', 'mes'], values=['dengue_diagnosis_previsto'], aggfunc='mean').reset_index()


def map_f_score_and_map_analysis():

    def clean_neighbor_name(x):
        x = re.sub(' Da ', ' da ', x)
        x = re.sub(' De ', ' de ', x)
        x = re.sub(' Do ', ' do ', x)
        x = re.sub(' Dos ', ' dos ', x)
        x = re.sub(' Os ', ' os ', x)

        return x.strip()

    def plot_regression_metric(df_metrics, gdf, metric, ax):

        df_metrics['neighbor'] = df_metrics['neighbor'].apply(clean_neighbor_name)

        gdf = pd.merge(df_metrics[['neighbor', metric]],
                       gdf, left_on='neighbor', right_on='NM_BAIRRO')

        gdf = gpd.GeoDataFrame(gdf)
        gdf[metric] = gdf[metric].astype(float)

        gdf.plot(ax=ax, column=metric, legend=True, edgecolor="black", cmap='viridis')
        ax.set_title("Regression Results by MAE (3 months)", fontsize=fontsize_title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    def plot_classification_metric(df_metrics, gdf, metric, ax):

        df_metrics['neighbor'] = df_metrics['neighbor'].apply(clean_neighbor_name)

        gdf = pd.merge(df_metrics[['neighbor', metric]],
                       gdf, left_on='neighbor', right_on='NM_BAIRRO')

        gdf = gpd.GeoDataFrame(gdf)

        gdf.plot(ax=ax, column=metric, legend=True, edgecolor="black", cmap='viridis_r')
        ax.set_title("Classification Results by F-Score (3 months)", fontsize=fontsize_title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_yticklabels([])
        ax.set_xticklabels([])

    version = '_v3'

    regression_metric = 'MAE'
    classification_metric = 'F-Score'

    df_model_3 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")

    df_pivot, df_regression_metrics = calculate_regression_metrics(df_model_3)

    df_model = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")

    df_percentil_model = apply_df_percentile_95_99(df_model)

    df_percentil_model['grupo_real'] = df_percentil_model['grupo_real'].astype(int)
    df_percentil_model['grupo_previsto'] = df_percentil_model['grupo_previsto'].astype(int)

    df_classification_metric = pd.DataFrame(columns=['neighbor', 'F-Score', 'Recall', 'Precision'])
    for b in list(df_percentil_model['nome_bairro'].unique()):
        df_b = df_percentil_model[df_percentil_model['nome_bairro'] == b]
        f_score = f1_score(df_b['grupo_real'], df_b['grupo_previsto'], average='macro')
        precision = precision_score(df_b['grupo_real'], df_b['grupo_previsto'], average='macro', zero_division=1)
        recall = recall_score(df_b['grupo_real'], df_b['grupo_previsto'], average='macro')
        df_classification_metric = df_classification_metric.append(
            pd.DataFrame([[b, f_score, precision, recall]], columns=['neighbor', 'F-Score', 'Recall', 'Precision']))

    gdf = gpd.read_file(os.getcwd() + '/data/Bairros.json').to_crs(epsg=3857)
    # basemap, basemap_extent = ctx.bounds2img(*gdf.total_bounds, zoom=10)

    f, ax = plt.subplots(2, 1, figsize=(8, 8), dpi=dpi)
    ax = ax.flatten()
    plot_regression_metric(df_regression_metrics, gdf, regression_metric, ax[0])
    plot_classification_metric(df_classification_metric, gdf, classification_metric, ax[1])
    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_4_map_mae_fscore.eps',
                dpi=dpi)
    plt.show()


def performance_by_month_analysis():

    def generate_matrix_multi_step(file_name, version):

        df_model = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")

        df_model = df_model[['mes', 'mes_inicial', target, target + '_previsto']]
        df_model.columns = ['mes', 'mes_inicial', 'Real', 'Predicted']

        df_metrics = pd.DataFrame(columns=['Initial Month', 'Month', 'MAE', 'RMSE', 'MAPE', 'R2'])

        for mi in list(df_model['mes_inicial'].unique()):
            df_mi = df_model[df_model['mes_inicial'] == mi]
            for m in list(df_mi['mes'].unique()):
                df_b = df_mi[df_mi['mes'] == m]
                mae = round(mean_absolute_error(list(df_b['Real']),
                                                list(df_b['Predicted'])))
                rmse = round(pow(mean_squared_error(list(df_b['Real']),
                                                    list(df_b['Predicted'])), 1 / 2))
                mape = round(epsilon_mape(df_b['Real'], df_b['Predicted']),1)
                r2 = round(r2_score(df_b['Real'], df_b['Predicted']), 1)

                df_metrics = df_metrics.append(pd.DataFrame([[mi, m, mae, rmse, mape, r2]],
                                                            columns=['Initial Month', 'Month', 'MAE', 'RMSE', 'MAPE', 'R2']))

        df_metrics.index = df_metrics['Initial Month']

        return df_metrics

    version = '_v3'
    dpi = 180
    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"

    df_multi_step = generate_matrix_multi_step(file_name, version)

    f, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=dpi)
    ax = ax.flatten()

    df = df_multi_step.pivot(columns='Month', values='MAE')
<<<<<<< HEAD
=======
    # sns.heatmap(df, annot=True, linewidths=.5, mask=df.isna())
>>>>>>> 6007da3035542093d3e826527fd6de8a52a11b69
    df = df.astype(float)
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax[0])
    ax[0].set_title("MAE", fontsize=fontsize_title)
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Initial Month', fontsize=fontsize_label)

    df = df_multi_step.pivot(columns='Month', values='RMSE')
    df = df.astype(float)
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax[1])
    ax[1].set_title("RMSE", fontsize=fontsize_title)
    ax[1].set_xlabel('')
    ax[1].set_ylabel('')

    df = df_multi_step.pivot(columns='Month', values='MAPE')
    df = df.astype(float)
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax[2])
    ax[2].set_title("MAPE", fontsize=fontsize_title)
    ax[2].set_xlabel('Predicted Month', fontsize=fontsize_label)
    ax[2].set_ylabel('Initial Month', fontsize=fontsize_label)

    df = df_multi_step.pivot(columns='Month', values='R2')
    df = df.astype(float)
    sns.heatmap(df, annot=True, linewidths=.5, ax=ax[3])
    ax[3].set_title("R2", fontsize=fontsize_title)
    ax[3].set_xlabel('Predicted Month', fontsize=fontsize_label)
    ax[3].set_ylabel('', fontsize=fontsize_label)
    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_7_multi_step_month_assessment.eps',
                dpi=dpi)

    # plt.show()


def realengo_and_bangu_district_analysis():

    def plot_line_neighbor(df, neighbor, ax, months_ahead):

        df = df[df['nome_bairro'] == neighbor]

        df_pivot = pd.melt(df, id_vars=['date'], value_vars=['Real', 'Predicted'])

        sns.lineplot(x='date', y='value', hue='variable', data=df_pivot, linewidth=1.5, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel('Dengue Cases', fontsize=fontsize_label)

        if months_ahead == 3:
            ax.set_title(f"{neighbor} (3 months)", fontsize=fontsize_title)
        else:
            ax.set_title(f"{neighbor} (1 month)", fontsize=fontsize_title)

        ax.legend(loc='upper right', fontsize=fontsize_label)
        ax.set_xticklabels([2015, 2016, 2017, 2018, 2019, 2020], fontsize=fontsize_label)

    neighbor_1 = 'Bangu'
    neighbor_2 = 'Realengo'

    # neighbor_1 = 'Pedra De Guaratiba'
    # neighbor_2 = 'Sampaio'

    version = '_v3'

    months_ahead = 1
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"
    df_model_1 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_model_1, df_model_1_metrics = calculate_regression_metrics(df_model_1)

    months_ahead = 3
    file_name = f"{year_begin}_{year_end}_{months_ahead}_months"
    df_model_3 = pd.read_csv(base_path + f"/data/df_model_prediction_{file_name}{version}.csv")
    df_model_3, df_model_3_metrics = calculate_regression_metrics(df_model_3)

    fontsize_label = 13

    f, ax = plt.subplots(2, 2, figsize=(8, 6), dpi=180)
    ax = ax.flatten()
    plot_line_neighbor(df_model_1, neighbor_1, ax[0], 1)
    plot_line_neighbor(df_model_1, neighbor_2, ax[1], 1)
    plot_line_neighbor(df_model_3, neighbor_1, ax[2], 3)
    plot_line_neighbor(df_model_3, neighbor_2, ax[3], 3)
    plt.tight_layout()
    plt.savefig(base_path + f'/data/figure_5_time_series_neighbors_analysis.eps',
                dpi=180)
    plt.show()

    df = df_model_3_metrics[df_model_3_metrics['neighbor'].isin([neighbor_1, neighbor_2])]
    df.columns = ['District', 'MAE', 'RMSE', 'MAPE', 'R2']

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()
    plt.savefig(base_path + f'/data/figure_5_table.eps',
                dpi=300)

    plt.show()


def shap_summary_and_dependence(read=False):

    version = '_v3'

    if not read:

        # Consolidate the results for differents years as a train dataset
        shap_values_consolidated = pd.DataFrame(columns=selected_columns)
        x_train_consolidated = pd.DataFrame(columns=selected_columns)

        df_train_cons = pd.DataFrame(columns=['nome_bairro', 'ano', 'mes'])

        for a in list_year:
            for initial_test_month in months:

                try:
                    if (a == 2020 and initial_test_month > 12 - months_ahead_predict) or \
                            (a == 2020 and initial_test_month in [10, 11, 12]):
                        break

                    years_for_train = list_year.copy()
                    years_for_train.remove(a)
                    df_train, df_test, x_train, y_train = run_main_first_step_1(target,
                                                                                   ano_teste=a,
                                                                                   log=False,
                                                                                   mes_inicio_teste=initial_test_month,
                                                                                   dengue_columns=dengue_columns,
                                                                                   columns_filtered=selected_columns,
                                                                                   categorical_columns=categorical_columns,
                                                                                   columns_filtered_categorical=columns_filtered_categorical,
                                                                                   quartil=True
                                                                                   )

                    regressor, r2 = run_catboost(x_train, y_train, grid_search=False, standart=standart)

                    explainer = shap.TreeExplainer(regressor)
                    shap_values = explainer.shap_values(x_train, check_additivity=False)
                    shap_values = pd.DataFrame(shap_values, columns=x_train.columns)
                    shap_values_consolidated = shap_values_consolidated.append(shap_values)
                    x_train_consolidated = x_train_consolidated.append(x_train)

                    df_train_cons = df_train_cons.append(df_train[['nome_bairro', 'ano', 'mes']])

                except Exception as ex:
                    embed()

        shap_values_consolidated.to_csv(base_path+f"/data/shap_values_{file_name}{version}.csv")
        x_train_consolidated.to_csv(base_path + f"/data/shap_values_x_train_{file_name}{version}.csv")


    else:

        shap_values_consolidated = pd.read_csv(os.path.join(base_path,
                                               "data",f"shap_values_{file_name}.csv"))
        x_train_consolidated = pd.read_csv(os.path.join(base_path,
                                           "data",f"shap_values_x_train_{file_name}.csv"))

        shap_values_consolidated.drop(columns=['Unnamed: 0'], inplace=True)
        x_train_consolidated.drop(columns=['Unnamed: 0'], inplace=True)

    x_train_consolidated.columns = ['num_health_unit', 'precipitation (mm)', 'temperature (°C)',
       'air_humidity (%)', 'neighbor_cases', 'cases_m-1', 'cases_m-2', 'cases_m-3', 'zika',
       'chikungunya', 'dengue_prevalence', 'liraa', 'demographic density']

    shap_values_consolidated_values = shap_values_consolidated.values
    shap.summary_plot(shap_values_consolidated_values,
                      x_train_consolidated, plot_size=(8, 6), show=False)
    # plt.yticks(labels=['cases_m-1', 'dengue_prevalence', 'precipitation (mm)', 'neighbor_cases',
    #             'temperature (°C)', 'cases_m-2', 'liraa', 'chikungunya', 'zika',
    #             'air_humidity (%)', 'demographic density', 'cases_m-3', 'num_health_unit'], fontsize=17)
    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_6_shap_summary_v2.eps',
                dpi=300)
    plt.show()

    f, ax = plt.subplots(2, 2, figsize=(16, 8), dpi=80)
    ax = ax.flatten()
    shap.dependence_plot('temperature (°C)', shap_values_consolidated_values,
                         x_train_consolidated, interaction_index="cases_m-1", show=False, ax=ax[0])
    ax[0].set_ylabel(f"Shap Value for\ntemperature (°C)", fontsize=17)
    ax[0].set_xlabel('temperature (°C)', fontsize=17)

    shap.dependence_plot('precipitation (mm)', shap_values_consolidated_values,
                         x_train_consolidated, interaction_index="cases_m-1", show=False, ax=ax[1])
    ax[1].set_ylabel(f"Shap Value for\nprecipitation (mm)", fontsize=17)
    ax[1].set_xlabel('precipitation (mm)', fontsize=17)

    shap.dependence_plot('dengue_prevalence', shap_values_consolidated_values,
                         x_train_consolidated, interaction_index="cases_m-1", show=False, ax=ax[2])
    ax[2].set_ylabel(f"Shap Value for\ndengue_prevalence", fontsize=17)
    ax[2].set_xlabel('dengue_prevalence', fontsize=17)

    shap.dependence_plot('cases_m-2', shap_values_consolidated_values,
                         x_train_consolidated, interaction_index="cases_m-1", show=False, ax=ax[3])
    ax[3].set_ylabel(f"Shap Value for\ncases_m-2", fontsize=17)
    ax[3].set_xlabel('cases_m-2', fontsize=17)

    plt.savefig(base_path + '/data/figure_6_shap_dependence_plot.eps', dpi=180)
    plt.tight_layout()
    plt.show()


def shap_decision_analysis():

    def understand_prediction(df_shap, df_train):

        df_shap['dengue_diagnosis_explainer'] = df_shap[x_train.columns].apply(
            lambda x: 0 if round(np.sum(x) + explainer.expected_value) < 0 \
                else round(np.sum(x) + explainer.expected_value), axis=1)

        df_concat = pd.concat([df_shap, df_train[target]], axis=1)

        df_concat['outbreak_explainer'] = df_concat['dengue_diagnosis_explainer'].apply(
            lambda x: 1 if x > 77 else 0)
        df_concat['outbreak_real'] = df_concat['dengue_diagnosis'].apply(
            lambda x: 1 if x > 77 else 0)
        df_concat['confusion_matrix'] = df_concat[['outbreak_real', 'outbreak_explainer']].apply(
            lambda x: 'true positive' if x[0] == 1 and x[1] == 1 else \
                'true negative' if x[0] == 0 and x[1] == 0 else \
                    'false positive' if x[0] == 0 and x[1] == 1 else \
                        'false negative', axis=1)

        df_concat['confusion_matrix'].value_counts()

        return df_concat

    df_train, df_test, x_train, y_train = run_main_first_step_1(target,
                                                                   ano_teste=2016,
                                                                   log=False,
                                                                   mes_inicio_teste=1,
                                                                   dengue_columns=dengue_columns,
                                                                   columns_filtered=selected_columns,
                                                                   categorical_columns=categorical_columns,
                                                                   columns_filtered_categorical=columns_filtered_categorical,
                                                                   quartil=True
                                                                   )

    regressor, r2 = run_catboost(x_train, y_train, grid_search=False, standart=standart)

    explainer = shap.TreeExplainer(regressor)
    df_shap = pd.DataFrame(explainer.shap_values(x_train, check_additivity=False), columns=x_train.columns)
    df_shap['densidade_demografica'] = df_shap['densidade_demografica'].apply(lambda x: round(x))
    shap_values = df_shap.values

    true_positive = 388
    false_negative = 3308
    false_positive = 3310
    true_negative = 6828

    # df_concat = understand_prediction(df_shap, df_train)
    # df = df_concat.iloc[[true_positive, false_negative, false_positive, true_negative],:]

    x_train.columns = ['num_health_unit', 'precipitation (mm)', 'temperature (°C)',
           'air_humidity (%)', 'neighbor_cases', 'demographic density', 'zika', 'chikungunya',
            'dengue_prevalence', 'liraa', 'cases_m-1', 'cases_m-2', 'cases_m-3']
    x_train['demographic density'] = x_train['demographic density'].apply(lambda x: round(x))
    x_train['neighbor_cases'] = x_train['neighbor_cases'].apply(lambda x: round(x))

    dpi=240
    plt.figure(figsize=(8, 6), dpi=dpi)

    # true positive
    plt.subplot(1,2,1)
    shap.decision_plot(explainer.expected_value, shap_values[true_positive, :],
                    x_train.iloc[true_positive, :], show=False, title="True Positive\nReal: 969\nPredicted:1060")
    # false negative
    plt.subplot(1,2,2)
    shap.decision_plot(explainer.expected_value, shap_values[false_negative, :],
                    x_train.iloc[false_negative, :], show=False, title=f"False Negative\nReal: 79\nPredicted:72")

    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_6_shap_decision_plot_part_1.eps', dpi=dpi)
    plt.show()

    plt.figure(figsize=(8, 6), dpi=dpi)

    # false positive
    plt.subplot(1,2,1)
    shap.decision_plot(explainer.expected_value, shap_values[false_positive, :],
                    x_train.iloc[false_positive, :], show=False, title=f"False Positive\nReal: 50\nPredicted:116")
    # true negative
    plt.subplot(1,2,2)
    shap.decision_plot(explainer.expected_value, shap_values[true_negative, :],
                    x_train.iloc[true_negative, :], show=False, title=f"True Negative\nReal: 2\nPredicted: 0")
    plt.tight_layout()
    plt.savefig(base_path + '/data/figure_6_shap_decision_plot_part_2.eps', dpi=dpi)
    plt.show()


if __name__ == '__main__':

    histogram_analysis()

    confusion_matrix()

    assessment_regression_metrics()

    time_series_analysis_model_vs_baseline()

    map_f_score_and_map_analysis()

    performance_by_month_analysis()

    realengo_and_bangu_district_analysis()

    shap_summary_and_dependence()

    shap_decision_analysis()

