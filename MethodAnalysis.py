import pandas as pd
import numpy as np
import scipy
import copy
import sklearn
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
#import matlab.engine
#import matlab
#eng = matlab.engine.start_matlab()
import datetime
import interpolation_methods
import ToolsForTemporalAnalysis
import gLV_interpolation
import warnings
warnings.filterwarnings("ignore")


def find_best_K(max_K, data):
    """
    :param max_K: int
    :param data: numpy array
    plots a graph of average bray-curtis score for each K using KNN with Epanechnikov kernel
    """

    first = 1
    last = data.shape[1] - 1
    all_K_abs = {}

    for cur_K in range(2, max_K):
        cur_knn_kernel_data = [[] for i in range(data.shape[0])]
        cur_knn_kernel_data_dif = []
        for cur_index in range(first, last):
            train = pd.concat((data.iloc[:, :cur_index], data.iloc[:, cur_index + 1:]), axis=1)

            cur_knn_interpolation = interpolation_methods.knn_interpolation(train, data.columns[cur_index], cur_K)
            for cur_spec in range(data.shape[0]):
                cur_knn_kernel_data[cur_spec].append(cur_knn_interpolation[cur_spec])

        cur_knn_kernel_data = np.array(cur_knn_kernel_data)
        for i in range(cur_knn_kernel_data.shape[1]):
            cur_knn_kernel_data_dif.append(
                ToolsForTemporalAnalysis.bray_curtis(cur_knn_kernel_data[:, i], data.iloc[:, i + 1]))
        all_K_abs[cur_K] = np.mean(cur_knn_kernel_data_dif)

    for i in all_K_abs.keys():
        plt.plot(i, all_K_abs[i], marker='.', color="blue")
    sns.set()
    plt.show()


def find_best_limits_values(LIMITS_repeats, data, jumps, upper_to_check):

    error_LIMITS_by_threshold = {}
    range_to_check = int(upper_to_check / jumps + 1)

    first = 1
    last = data.shape[1]
    for i in range(1, range_to_check):
        LIMITS_error_threshold = jumps * i
        print(LIMITS_error_threshold)
        cur_LIMITS_data_dif = []
        for cur_index in range(first, last):
            train = pd.concat((data.iloc[:, :cur_index], data.iloc[:, cur_index + 1:]), axis=1)
            test = data.iloc[:, cur_index: cur_index + 1]
            F = gLV_interpolation.create_F_matrix(np.array(train), list(train.columns))

            # lotka volterra using LIMITS with median
            tmp_LIMITS_mue, tmp_LIMITS_M = gLV_interpolation.LIMITS(F, np.array(train.iloc[:, 1:]),
                                                                    LIMITS_error_threshold, LIMITS_repeats, med_or_avg=0)
            tmp_prediction_LIMITS_cur = gLV_interpolation.predict_abundances_day_by_day(
                np.array(data.iloc[:, cur_index - 1: cur_index]),
                data.columns[cur_index] - data.columns[cur_index - 1],
                tmp_LIMITS_mue, tmp_LIMITS_M)
            cur_LIMITS_data_dif.append(ToolsForTemporalAnalysis.bray_curtis(tmp_prediction_LIMITS_cur, np.array(test)))

        error_LIMITS_by_threshold[LIMITS_error_threshold] = np.mean(cur_LIMITS_data_dif)

    for i in error_LIMITS_by_threshold.keys():
        plt.plot(i, error_LIMITS_by_threshold[i], marker='.', color="blue")
    sns.set()
    plt.show()
