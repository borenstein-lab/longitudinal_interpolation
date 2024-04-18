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
import interpolation_methods
import gLV_interpolation



def bray_curtis(sample_1, sample_2):
    bray_curtis_score = 0
    total_sum = np.sum((np.sum(sample_1), np.sum(sample_2)))
    for i in range(len(sample_1)):
        cur_min = np.min((sample_1[i], sample_2[i]))
        bray_curtis_score += (2 * cur_min) / total_sum
    return bray_curtis_score

def test_train_partition(X, percentage):
    """
    gets a pandas DF (with some unique indicator as column name) and a number between 0 and 1
    returns a random partition of the data into both train dataset (DF) which includes the percentage as input and
    test data (as a DF) that includes the rest.
    """
    # select the indeces of each group
    col_nums = [i for i in range(X.shape[1])]
    scipy.random.shuffle(col_nums)
    train_indeces = col_nums[:int(scipy.rint(percentage * len(col_nums)))]
    train_indeces.sort()
    train_df = X.iloc[:, train_indeces[0]]
    for i in range(1, len(train_indeces)):
        train_df = pd.concat((train_df, X.iloc[:, train_indeces[i]]), axis=1)
    test_indeces = col_nums[int(scipy.rint(percentage * len(col_nums))) + 1:]
    test_indeces.sort()
    test_df = X.iloc[:, test_indeces[0]]
    for i in range(1, len(test_indeces)):
        test_df = pd.concat((test_df, X.iloc[:, test_indeces[i]]), axis=1)
    return train_df, test_df

def partition_data(X, k):
    """
    gets a pandas DF (with some unique indicator as column name) and k.
    returns a list of the DF partitioned into k equal sized (up to 1) groups
    """
    # select the indeces of each group
    col_nums = [i for i in range(X.shape[1])]
    scipy.random.shuffle(col_nums)
    tmp_col_partition_nums = np.array_split(col_nums, k)
    col_partition_nums = []
    for i in range(k):
        col_partition_nums.append(list(tmp_col_partition_nums[i]))
        col_partition_nums[-1].sort()
    # makes the actual partition
    col_partition = []
    for i in col_partition_nums:
        col_partition.append(X.iloc[:, i[0]])
        for j in range(1, len(i)):
            col_partition[-1] = pd.concat((col_partition[-1], X.iloc[:, i[j]]), axis=1)
    return col_partition


def interpolate_all_methods_removing_one_day(data, lambda_M, lambda_mue, K, LIMITS_error_threshold, LIMITS_repeats,
                                             working_path_for_dbn):
    """

    :param data: as a panda dataframe where the columns represents the timepoint
    :param lambda_M: for MLRR
    :param lambda_mue: for MLRR
    :param K: for KNN
    :param LIMITS_error_threshold:
    :param LIMITS_repeats:
    :param working_path_for_dbn:
    :return: interpolation of the data for each method removing and interpolating one day at a time.
    returns a dictionary where the keys are the interpolation methods' names and the values are lists where the i'th index
    represents the i'th specie and the j'th index in that list represents the j'th time point interpolated
    """
    first_index_to_return = 2
    last_index_to_return = data.shape[0] + 2

    first = 1
    last = data.shape[1] - 1

    real_data = [[] for i in range(data.shape[0])]
    spline_vals = [[] for i in range(data.shape[0])]
    last_time_point_vals = [[] for i in range(data.shape[0])]
    weighted_avg_vals = [[] for i in range(data.shape[0])]
    average_data = [[] for i in range(data.shape[0])]
    median_data = [[] for i in range(data.shape[0])]
    glv_data = [[] for i in range(data.shape[0])]
    glv_MSE = [[] for i in range(data.shape[0])]
    glv_LIMITS = [[] for i in range(data.shape[0])]
    knn_kernel_data = [[] for i in range(data.shape[0])]
    equal = [[] for i in range(data.shape[0])]
    dbn_sparse_normal = [[] for i in range(data.shape[0])]
    dbn_dense_normal = [[] for i in range(data.shape[0])]

    # organizing the data for the dbn
    data_transpose = data.T
    data_transpose["time_points"] = data.columns
    time_dif = [data_transpose.index[1] - data_transpose.index[0]]
    for i in range(1, data_transpose.shape[0] - 1):
        time_dif.append(data_transpose.index[i + 1] - data_transpose.index[i])
    time_dif.append(0)
    data_transpose["time_dif"] = time_dif
    data_transpose["SubjectID"] = 1
    tmp_data = data_transpose.iloc[:, -1:]
    tmp_data = pd.concat((tmp_data, data_transpose.iloc[:, -2:-1]), axis=1)
    data_transpose = pd.concat((tmp_data, data_transpose.iloc[:, : -3]), axis=1)

    for cur_index in range(first, last):
        return_all_methods_in_timepoint(data, data_transpose, cur_index, K, lambda_M, lambda_mue,
                                    LIMITS_error_threshold, LIMITS_repeats,
                                    working_path_for_dbn, first_index_to_return, last_index_to_return,
                                    real_data, last_time_point_vals, average_data, median_data, knn_kernel_data,
                                    glv_data, glv_MSE, glv_LIMITS, equal,
                                    spline_vals, weighted_avg_vals, dbn_sparse_normal, dbn_dense_normal)

    all_methods_interpolation_data = {}
    all_methods_interpolation_data["real data"] = real_data
    all_methods_interpolation_data["last time point"] = last_time_point_vals
    all_methods_interpolation_data["average"] = average_data
    all_methods_interpolation_data["median"]= median_data
    all_methods_interpolation_data["KNN"] = knn_kernel_data
    all_methods_interpolation_data["MLRR"] = glv_data
    all_methods_interpolation_data["gLV MSE"] = glv_MSE
    all_methods_interpolation_data["LIMITS"] = glv_LIMITS
    all_methods_interpolation_data["equal"] = equal
    all_methods_interpolation_data["spline"] = spline_vals
    all_methods_interpolation_data["weighted average"] = weighted_avg_vals
    all_methods_interpolation_data["sparse DBN"] = dbn_sparse_normal
    all_methods_interpolation_data["dense DBN"] = dbn_dense_normal

    return all_methods_interpolation_data


def bray_curtis_for_all_methods(real_data, all_methods_result):
    """

    :param real_data: array
    :param all_methods_result: dictionary of arrays corresponding to the real data
    :return: a dictionary of lists where every list is the bray curtis result between real data and the method
    """
    all_bray_curtis_per_method = {}

    for method in all_methods_result.keys():
        cur_method_bray_curtis = []
        for i in range(real_data.shape[1]):
            cur_method_bray_curtis.append(bray_curtis(list(real_data[:, i]), list(all_methods_result[method][:,i])))
        all_bray_curtis_per_method[method] = cur_method_bray_curtis
    return all_bray_curtis_per_method


def product_per_method(real_data, all_methods_result, method_to_compare = "relative_error"):
    """
    :param real_data: array
    :param all_methods_result: dictionary of arrays corresponding to the real data
    :param method_to_compare: whether to use relative error or other
    :return: a dictionary of arrays where every array is the product of min(real_data, method)/max(real_data, method) or relative error
    """
    all_precentage_per_method = {}
    for method in all_methods_result.keys():
        cur_method = copy.deepcopy(all_methods_result[method])
        current_method_min_val = np.min(np.array(cur_method)[np.nonzero(np.array(cur_method))])
        cur_method[cur_method == 0] = current_method_min_val
        if method_to_compare == "min_max":
            cur_min = np.minimum(real_data, cur_method)
            cur_max = np.maximum(real_data, cur_method)
            cur_division_product = np.divide(cur_min, cur_max)
        elif method_to_compare == "relative_error":
            tmp_res = np.abs(cur_method - real_data)
            cur_division_product = np.divide(tmp_res, real_data)
        all_precentage_per_method[method] = cur_division_product
    return all_precentage_per_method

def product_per_method2(real_data, all_methods_result):
    """
    :param real_data: array
    :param all_methods_result: dictionary of arrays corresponding to the real data
    :return: a dictionary of arrays where every array is the product of min(real_data, method)/max(real_data, method)
    """
    all_precentage_per_method = {}
    for method in all_methods_result.keys():
        cur_method = copy.deepcopy(all_methods_result[method])
        current_method_min_val = np.min(np.array(cur_method)[np.nonzero(np.array(cur_method))])
        cur_method[cur_method == 0] = current_method_min_val
        cur_min = np.minimum(real_data, cur_method)
        cur_max = np.maximum(real_data, cur_method)
        cur_division_product = np.divide(cur_min, cur_max)
        all_precentage_per_method[method] = cur_division_product
    return all_precentage_per_method

def return_all_methods_in_timepoint(data, data_transpose, cur_index, K, lambda_M, lambda_mue,
                                    LIMITS_error_threshold, LIMITS_repeats,
                                    working_path_for_dbn, first_index_to_return, last_index_to_return,
                                    real_data, last_time_point_vals, average_data, median_data, knn_kernel_data,
                                    glv_data, glv_MSE, glv_LIMITS, equal,
                                    spline_vals, weighted_avg_vals, dbn_sparse_normal, dbn_dense_normal,
                                    should_use_limits = True):

    train = pd.concat((data.iloc[:, :cur_index], data.iloc[:, cur_index + 1:]), axis=1)
    test = data.iloc[:, cur_index: cur_index + 1]

    cur_knn_interpolation = interpolation_methods.knn_interpolation(train, data.columns[cur_index], K)

    ##lotka-VOLTERRA using MLRR
    F = gLV_interpolation.create_F_matrix(np.array(train), list(train.columns))
    Y = gLV_interpolation.create_y_matrix(np.array(train))
    coef_matrix = gLV_interpolation.coef_matrix_by_paper_trick_ridge(Y, F, lambda_M, lambda_mue, 0)
    tmp_M = coef_matrix[:, :-1]
    tmp_mue = coef_matrix[:, -1].reshape(coef_matrix.shape[0], 1)

    # lotka volterra using MSE
    if should_use_limits:
        M_MSE, mue_MSE, pert_coef_MSE = gLV_interpolation.MSE_gLV(F, Y)

    tmp_prediction = gLV_interpolation.predict_abundances_day_by_day(
        np.array(data.iloc[:, cur_index - 1: cur_index]),
        data.columns[cur_index] - data.columns[cur_index - 1],
        tmp_mue, tmp_M)

    if should_use_limits:
        tmp_prediction_MSE = gLV_interpolation.predict_abundances_day_by_day(
            np.array(data.iloc[:, cur_index - 1: cur_index]),
            data.columns[cur_index] - data.columns[cur_index - 1],
            mue_MSE, M_MSE)

    # lotka volterra using LIMITS with median
    if should_use_limits:
        LIMITS_mue, LIMITS_M = gLV_interpolation.LIMITS(F, np.array(train.iloc[:, 1:]),
                                                        LIMITS_error_threshold, LIMITS_repeats, med_or_avg=0)
        tmp_prediction_LIMITS = gLV_interpolation.predict_abundances_day_by_day(
            np.array(data.iloc[:, cur_index - 1: cur_index]),
            data.columns[cur_index] - data.columns[cur_index - 1],
            LIMITS_mue, LIMITS_M)

    # interpolate using spline

    cur_spline_interpolation = interpolation_methods.spline_interpolation(train, test.columns[0])

    # interpolate using weighted average
    cur_weighted_avg_interpolation = interpolation_methods.weighted_avg_interpolation(data, cur_index)

    # dbn basics
    cur_data_to_learn_dbn = data_transpose.iloc[:cur_index, :]
    cur_data_to_learn_dbn = pd.concat((cur_data_to_learn_dbn, data_transpose.iloc[cur_index + 1:, :]))
    data_to_use_in_prediction = pd.concat((data_transpose.iloc[cur_index - 1:cur_index, :],
                                           data_transpose.iloc[cur_index - 1:cur_index, :]))

    # sparse dbn
    sparse_dbn_matlab_eng = interpolation_methods.dbn_learning(cur_data_to_learn_dbn, working_path_for_dbn, 0,
                                                               '', [], 3, False, False, True, 10, 10, 1)

    tmp_dbn_sparse_normal = interpolation_methods.dbn_predict(sparse_dbn_matlab_eng, data_to_use_in_prediction,
                                                              working_path_for_dbn,
                                                              first_index_to_return, last_index_to_return)

    # dense dbn

    dense_dbn_matlab_eng = interpolation_methods.dbn_learning(cur_data_to_learn_dbn, working_path_for_dbn, 0,
                                                              '', [], 5, False, False, True, 50, 50, 1)

    tmp_dbn_dense_normal = interpolation_methods.dbn_predict(dense_dbn_matlab_eng, data_to_use_in_prediction,
                                                             working_path_for_dbn,
                                                             first_index_to_return, last_index_to_return)
    for cur_spec in range(data.shape[0]):
        real_data[cur_spec].append(np.max((0, float(test.iloc[cur_spec]))))

        ###last time point interpolation
        last_time_point_vals[cur_spec].append((data.iloc[cur_spec, cur_index - 1]))

        ###interpolate using average and median
        average_data[cur_spec].append(np.average(train.iloc[cur_spec, :]))
        median_data[cur_spec].append(np.median(train.iloc[cur_spec, :]))

        # adding all data

        knn_kernel_data[cur_spec].append(cur_knn_interpolation[cur_spec])

        glv_data[cur_spec].append(float(tmp_prediction[cur_spec]))
        if should_use_limits:
            glv_MSE[cur_spec].append(float(tmp_prediction_MSE[cur_spec]))

            glv_LIMITS[cur_spec].append(float(tmp_prediction_LIMITS[cur_spec]))

        equal[cur_spec].append(1 / data.shape[0])

        spline_vals[cur_spec].append(np.max(cur_spline_interpolation[cur_spec], 0))

        weighted_avg_vals[cur_spec].append(cur_weighted_avg_interpolation[cur_spec])

        dbn_sparse_normal[cur_spec].append(tmp_dbn_sparse_normal[cur_spec])

        dbn_dense_normal[cur_spec].append(tmp_dbn_dense_normal[cur_spec])



def monte_carlo(all_data, number_of_iterations, sample_size,
                K, lambda_M, lambda_mue,
                LIMITS_error_threshold, LIMITS_repeats,
                working_path_for_dbn, first_index_to_return, last_index_to_return, succesive = False):

    monte_carlo_by_time_dif = {}

    for iteration in range(number_of_iterations):
        column_list = copy.deepcopy(list(all_data.columns))
        if succesive:
            random_start_point = np.random.randint(0, len(column_list) - sample_size)
            column_list = column_list[random_start_point: random_start_point + sample_size]
        else:
            np.random.shuffle(column_list)
            column_list = column_list[:sample_size]
            column_list.sort()

        cur_data = all_data.loc[:, column_list]
        cur_data.columns = cur_data.columns - cur_data.columns[0]
        cur_index = np.random.randint(1, cur_data.shape[1] - 1)

        data_transpose = cur_data.T
        data_transpose["time_points"] = cur_data.columns
        time_dif = [data_transpose.index[1] - data_transpose.index[0]]
        for i in range(1, data_transpose.shape[0] - 1):
            time_dif.append(data_transpose.index[i + 1] - data_transpose.index[i])
        time_dif.append(0)
        data_transpose["time_dif"] = time_dif
        data_transpose["SubjectID"] = 1
        tmp_data = data_transpose.iloc[:, -1:]
        tmp_data = pd.concat((tmp_data, data_transpose.iloc[:, -2:-1]), axis=1)
        data_transpose = pd.concat((tmp_data, data_transpose.iloc[:, : -3]), axis=1)

        cur_time_dif = cur_data.columns[cur_index] - cur_data.columns[cur_index - 1]
        if cur_time_dif not in monte_carlo_by_time_dif:
            monte_carlo_by_time_dif[cur_time_dif] = {}
            monte_carlo_by_time_dif[cur_time_dif]["real_data"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["spline_vals"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["last_time_point_vals"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["weighted_avg_vals"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["average_data"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["median_data"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["glv_data"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["glv_MSE"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["glv_LIMITS"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["knn_kernel_data"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["equal"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["dbn_sparse_normal"] = [[] for i in range(cur_data.shape[0])]
            monte_carlo_by_time_dif[cur_time_dif]["dbn_dense_normal"] = [[] for i in range(cur_data.shape[0])]


        return_all_methods_in_timepoint(cur_data, data_transpose, cur_index, K, lambda_M, lambda_mue,
                                        LIMITS_error_threshold, LIMITS_repeats,
                                        working_path_for_dbn, first_index_to_return, last_index_to_return,
                                        monte_carlo_by_time_dif[cur_time_dif]["real_data"],
                                        monte_carlo_by_time_dif[cur_time_dif]["last_time_point_vals"],
                                        monte_carlo_by_time_dif[cur_time_dif]["average_data"],
                                        monte_carlo_by_time_dif[cur_time_dif]["median_data"],
                                        monte_carlo_by_time_dif[cur_time_dif]["knn_kernel_data"],
                                        monte_carlo_by_time_dif[cur_time_dif]["glv_data"],
                                        monte_carlo_by_time_dif[cur_time_dif]["glv_MSE"],
                                        monte_carlo_by_time_dif[cur_time_dif]["glv_LIMITS"],
                                        monte_carlo_by_time_dif[cur_time_dif]["equal"],
                                        monte_carlo_by_time_dif[cur_time_dif]["spline_vals"],
                                        monte_carlo_by_time_dif[cur_time_dif]["weighted_avg_vals"],
                                        monte_carlo_by_time_dif[cur_time_dif]["dbn_sparse_normal"],
                                        monte_carlo_by_time_dif[cur_time_dif]["dbn_dense_normal"],
                                        should_use_limits=False)

    return monte_carlo_by_time_dif

