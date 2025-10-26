'''
Code got from:

https://machinelearningmastery.com/how-to-grid-search-sarima-model-hyperparameters-for-time-series-forecasting-in-python/
'''

from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import pandas as pd
import os

base_path = os.getcwd()

# one-step sarima forecast
def sarima_forecast(history, config):
    order, sorder, trend = config
    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(len(history), len(history))
    return yhat[0]


# root mean squared error or rmse
def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # fit model and make forecast for history
        yhat = sarima_forecast(history, cfg)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
    # estimate prediction error
    error = measure_rmse(test, predictions)
    return error


# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = walk_forward_validation(data, n_test, cfg)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = walk_forward_validation(data, n_test, cfg)
        except:
            error = None
    # check for an interesting result
    if result is not None:
        print(' > Model[%s] %.3f' % (key, result))
    return (key, result)


# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):

    scores = None
    if parallel:
        # execute configs in parallel
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
    # remove empty results
    scores = [r for r in scores if r[1] != None]
    # sort configs by error, asc
    scores.sort(key=lambda tup: tup[1])
    return scores


# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
    models = list()
    # define config lists
    p_params = [0, 1, 2]
    d_params = [0, 1]
    q_params = [0, 1, 2]
    t_params = ['n', 'c', 't', 'ct']
    P_params = [0, 1, 2]
    D_params = [0, 1]
    Q_params = [0, 1, 2]
    m_params = seasonal
    # create config instances
    for p in p_params:
        for d in d_params:
            for q in q_params:
                for t in t_params:
                    for P in P_params:
                        for D in D_params:
                            for Q in Q_params:
                                for m in m_params:
                                    cfg = [(p, d, q), (P, D, Q, m), t]
                                    models.append(cfg)
    return models


if __name__ == '__main__':

    # define dataset
    df = pd.read_csv(base_path + "/data_sus/finais/Inputs/dengue_input_from_source_v10.csv")
    df = df[df['ano'].isin(list(range(2012, 2015)))]
    df = df.pivot_table(index=['ano', 'mes'], values=['dengue_diagnosis'], aggfunc='sum')

    data = list(df['dengue_diagnosis'])

    # data split
    n_test = 4
    # model configs
    cfg_list = sarima_configs(seasonal=[12])
    # grid search
    scores = grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:30]:
        print(cfg, error)

    '''
        [(2, 1, 0), (0, 0, 0, 12), 'n'] 21.22817471710413
        [(2, 0, 0), (1, 0, 0, 12), 'n'] 23.210183178577143
        [(0, 1, 2), (2, 0, 2, 12), 'c'] 24.306377375825388
        [(0, 1, 2), (2, 0, 0, 12), 'n'] 26.40234125572175
        [(1, 0, 1), (0, 0, 2, 12), 'c'] 31.638644369232185
        [(0, 1, 1), (0, 0, 2, 12), 'n'] 32.63315105795377
        [(1, 1, 0), (0, 0, 0, 12), 'n'] 33.70135346159033
        [(2, 0, 0), (0, 0, 0, 12), 'n'] 33.80877664292555
        [(1, 0, 1), (2, 0, 0, 12), 'c'] 35.1882714907409
        [(0, 1, 2), (0, 0, 2, 12), 'n'] 36.62757786603621
        [(1, 0, 0), (1, 0, 2, 12), 'n'] 36.71767767022515
        [(1, 0, 0), (0, 0, 0, 12), 'n'] 37.94695620166669
        [(1, 1, 0), (2, 0, 0, 12), 'n'] 39.635788863063595
        [(0, 1, 1), (2, 0, 1, 12), 'n'] 40.40637845483406
        [(0, 1, 0), (0, 0, 2, 12), 't'] 40.746077897888036
        [(0, 1, 0), (0, 0, 0, 12), 'n'] 40.97255178775176
        [(0, 1, 0), (0, 0, 2, 12), 'n'] 41.2535328071982
        [(0, 1, 0), (1, 0, 2, 12), 'n'] 41.283865065523436
        [(1, 1, 0), (1, 0, 0, 12), 'n'] 41.33559152586288
        [(0, 1, 0), (2, 0, 1, 12), 'n'] 41.952879433196465
        [(0, 1, 0), (2, 0, 0, 12), 'n'] 41.9533567709807
        [(0, 1, 0), (2, 0, 2, 12), 'n'] 42.10897171756217
        [(0, 1, 2), (1, 0, 2, 12), 'n'] 42.65667133683036
        [(1, 0, 0), (0, 0, 2, 12), 'n'] 43.625824917947845
        [(0, 0, 2), (2, 0, 1, 12), 'ct'] 44.9426673629182
        [(2, 1, 1), (0, 0, 0, 12), 'n'] 45.229269799703495
        [(1, 0, 0), (2, 0, 0, 12), 'n'] 46.25744548503802
        [(1, 0, 0), (1, 0, 0, 12), 'n'] 46.84109364813003
        [(1, 0, 0), (2, 0, 1, 12), 'n'] 47.12238700021865
        [(0, 1, 0), (2, 0, 0, 12), 'c'] 47.98545669608636
    
    '''