import pandas as pd
import numpy as np
import scipy
import copy
import sklearn
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import matlab.engine
import matlab
eng = matlab.engine.start_matlab()
import datetime


def knn_interpolation(samples, time_point_to_complete, K):
    """
    gets a list of all samples (as a dataframe. the columns names shoul be the timepoints), a time point to interpolate and K

    returns an interpolation of the data point using KNN kernel with Epanechnikov function
    """
    all_time_points = list(samples.columns)
    samples_dist = [time_point - time_point_to_complete for time_point in all_time_points]
    samples_dist = np.abs(samples_dist)
    samples_dist.sort()
    b = samples_dist[K]

    time_point_interpolated = None

    for i in range(samples.shape[1]):
        sample = samples.columns[i]
        if sample - b <= time_point_to_complete and time_point_to_complete <= sample + b:
            ker_norm = (time_point_to_complete - sample) / b
            kernel = 0.75 * (1 - (ker_norm ** 2))
            if type(time_point_interpolated) == np.ndarray:
                time_point_interpolated += kernel * np.array(samples.iloc[:, i])
            else:
                time_point_interpolated = kernel * np.array(samples.iloc[:, i])

    sum_time_point = sum(time_point_interpolated)
    for i in range(time_point_interpolated.shape[0]):
        time_point_interpolated[i] = time_point_interpolated[i] / sum_time_point
    return time_point_interpolated

def spline_interpolation(train, test_day):
    """
    gets train data and a day to interpolate.
    interpolates all species in this day
    """
    spline_per_spec = []
    for cur_spec in range(train.shape[0]):
        tmp_spline = scipy.interpolate.interp1d(list(train.columns), list(train.iloc[cur_spec, :]), kind='cubic')
        spline_per_spec.append(float(max(0, tmp_spline(test_day))))
    return spline_per_spec


def weighted_avg_interpolation(all_data, index_to_interpolate):

    """
    :param all_data:
    :param index_to_interpolate:
    :return: the index of the timepoint to interpolate
    """

    weighted_avg_per_spec = []

    for cur_spec in range(all_data.shape[0]):
        prev_data_point = all_data.iloc[cur_spec, index_to_interpolate - 1]
        next_data_point = all_data.iloc[cur_spec, index_to_interpolate + 1]

        prev_time_point = int(all_data.columns[index_to_interpolate - 1])
        cur_time_point = int(all_data.columns[index_to_interpolate])
        next_time_point = int(all_data.columns[index_to_interpolate + 1])

        weighted_avg = (next_data_point * (cur_time_point - prev_time_point) +
                        prev_data_point * (next_time_point - cur_time_point)) / (next_time_point - prev_time_point)

        weighted_avg_per_spec.append(weighted_avg)
    return weighted_avg_per_spec


def dbn_learning(data_to_learn_dbn, working_path, index_to_learn, pheno, cols2learn, maxParents,
                      localNetStruc, intraEdges, mle, nu, alpha, sigma2):
    """
     pheno         % string matching one of the colunms. Defaults to none. (optional)
     cols2learn    % variables to be learned. Input as indices. Defaults to all. needs to be in brackets []
     maxParents;   % maximum number of parents for any node. Defaults to 3 max parents per node. (optional)
     localNetStruc % if true, compare only local network structures using log Bayes factor instead of BIC/AIC score. Defaults to false. (optional)
     intraEdges;   % if true, intra edges will be learned in the network structure. Defaults to false. (optional)
     mle;          % if true, maximum likelihood estimation will be used.
                   % Otherwise a Bayesian estimation (MAP) will be employed. Defaults to true. (optional)
     nu            % prior sample size for prior variance estimate. Defaults to 1. (optional)
     alpha         % prior sample size for discrete nodes. Defaults to 1. (optional)
     sigma2        % prior variance estimate. Defaults to 1. (optional)
    """
    path_data_to_learn = working_path + "\\learn_dbn"
    data_to_learn_dbn.to_csv(path_data_to_learn + ".tsv", sep="\t", index=False)
    dbn = eng.learnDynamicBayesNetwork(path_data_to_learn, pheno, cols2learn, maxParents,
                                       localNetStruc, intraEdges, mle, nu, alpha, sigma2)
    return dbn


def dbn_predict(inference_dbn, data_to_use_in_prediction, working_dir,
                  first_index_to_return, last_index_to_return):

    data_inference_loc = working_dir + "\\tmp_prediction_data"
    prediction_loc = working_dir + "\\predictions"

    data_to_use_in_prediction.to_csv(data_inference_loc + ".tsv", sep="\t", index=False)

    eng.inferenceDynamicBayesNetwork(inference_dbn, data_inference_loc,
                                     matlab.double([i for i in range(first_index_to_return, last_index_to_return)]),
                                     prediction_loc)
    tmp_pred = pd.read_csv(prediction_loc + "_predictions_mb.csv", header=None)

    tmp_pred = tmp_pred.iloc[:, 1:]
    tmp_pred[tmp_pred < 0] = 0
    tmp_sum = np.sum(tmp_pred.iloc[0, :])
    infer_to_ret = []
    for j in range(tmp_pred.shape[1]):
        infer_to_ret.append(tmp_pred.iloc[0, j] / tmp_sum)
    return infer_to_ret

