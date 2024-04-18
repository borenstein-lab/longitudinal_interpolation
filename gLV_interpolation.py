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


def predict_abundances_day_by_day(x, delta_t, self_growth, interaction_matrix, perturbation_matrix = None, perturbation_incidence = None):
    """gets the abundances at time t (x), the passed time (delta_t), the self_growth factor, interaction matrix
    and possible perturbations (all as numpy arrays shape of vectors should be (i,1)).
    returns the expected *relative* abundances based on the Lotka-volterra model.
    if delta_t>1 and there are perturbations in few time points, than the first columnin perturbation_incidence is
    the perturbation before the first time point, the second column is before the second one etc..
    based on https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388#s4 descritization"""
    x_t = copy.deepcopy(x)
    for k in range(delta_t): #calculate for each time point seperatly
        cur_x_t_plus_one = []
        for i in range(x_t.shape[0]): #calc single specie abundance
            delta_x = 0
            delta_x += self_growth[i][0] #add self growth factor
            for j in range(x_t.shape[0]): #add growth by interaction between species
                delta_x += interaction_matrix[i][j] * x_t[j]
            if perturbation_incidence != None: # add growth by perturbations (if exist)
                for j in perturbation_incidence.shape[0]:
                    delta_x += perturbation_incidence[j][0] * perturbation_matrix[i][j]
                perturbation_incidence = np.delete(perturbation_incidence, 0, 1) #delete current perturbation
                if perturbation_incidence.shape[1] == 0: #if there aren't any more perturbations, from now on ignore
                    perturbation_incidence = None
            delta_x = np.exp(delta_x) * x_t[i] #calculate predicted x_(t+1)
            if delta_x < 0:
                delta_x = 0
            cur_x_t_plus_one.append(delta_x)
        sum_cur_x_t_plus_one = sum(cur_x_t_plus_one)
        cur_x_t_plus_one = [y / sum_cur_x_t_plus_one for y in cur_x_t_plus_one] #all relative abundances need to sum up to 1
        x_t = np.array(cur_x_t_plus_one).reshape(len(cur_x_t_plus_one),1)
    return x_t

def create_y_matrix(X, pert_incidence=None):
    """
    gets the abundances matrix (X) and perturbations incidence matrix (if exists)
    returns the y matrix as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388#s4
    """
    ones = np.ones((1, X.shape[1] - 1))
    y_mat = np.concatenate((X[:, 1:], ones), axis=0)
    if pert_incidence != None:
        y_mat = np.concatenate((y_mat, pert_incidence), axis=0)
    return y_mat

def create_F_matrix(X, time_points):
    """
    gets the abundances matrix (X) and time points (as a list)
    returns the F matrix as mantioned in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388#s4
    """
    F_mat = X[:, 1] / X[:, 0]
    F_mat = F_mat.reshape(X.shape[0], 1)
    F_mat = np.log(F_mat)
    F_mat = np.nan_to_num(F_mat, True, 0, 0, 0)
    F_mat = F_mat / (time_points[1] - time_points[0])
    for i in range(X.shape[1] - 2):
        tmp_F_vec = X[:, i + 2] / X[:, i + 1]
        tmp_F_vec = tmp_F_vec.reshape(X.shape[0], 1)
        tmp_F_vec = np.log(tmp_F_vec)
        tmp_F_vec = np.nan_to_num(tmp_F_vec, True, 0, 0, 0)
        tmp_F_vec = tmp_F_vec / (time_points[i + 2] - time_points[i + 1])
        F_mat = np.concatenate((F_mat, tmp_F_vec), axis=1)
    return F_mat

def coef_matrix_by_paper_trick_ridge(Y, F, lambda_M, lambda_mue, lambda_E):
    """
    gets the Y and F matrices as numpy arrays and a lambda as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388#s4
    returns the coef. matrix as in the paper without using sklearn (F*Y^T(YY^T+D_lambda)^-1)
    """
    num_of_spec = F.shape[0]
    num_of_pert = Y.shape[0] - num_of_spec - 1
    lambda_for_D = [lambda_M for i in range(num_of_spec)]
    lambda_for_D.append(lambda_mue)
    for i in range(num_of_pert):
        lambda_for_D.append(lambda_E)
    D = np.diagflat(lambda_for_D)
    Y_mult = np.matmul(Y, Y.T)
    Y_add = np.add(Y_mult, D)
    Y_add_inv = np.linalg.inv(Y_add)
    coef_mat = np.matmul(F, np.matmul(Y.T, Y_add_inv))
    return coef_mat

def LIMITS(F, data, error_threshold, repeats, med_or_avg=0, perturbation_matrix_by_time=None):
    all_M = []
    all_mue = []
    for j in range(repeats):

        # partion the data to train and test
        data_nums = [i for i in range(F.shape[1])]
        scipy.random.shuffle(data_nums)
        train_nums = data_nums[: int(F.shape[1] * 0.5)]
        train_nums.sort()
        test_nums = data_nums[int(F.shape[1] * 0.5):]
        test_nums.sort()

        train_F = F[:, train_nums[0]: train_nums[0] + 1]
        train_data = data[:, train_nums[0]: train_nums[0] + 1]
        for i in range(1, len(train_nums)):
            train_F = np.concatenate((train_F, F[:, train_nums[i]: train_nums[i] + 1]), axis=1)
            train_data = np.concatenate((train_data, data[:, train_nums[i]: train_nums[i] + 1]), axis=1)

        test_F = F[:, test_nums[0]: test_nums[0] + 1]
        test_data = data[:, test_nums[0]: test_nums[0] + 1]
        for i in range(1, len(test_nums)):
            test_F = np.concatenate((test_F, F[:, test_nums[i]: test_nums[i] + 1]), axis=1)
            test_data = np.concatenate((test_data, data[:, test_nums[i]: test_nums[i] + 1]), axis=1)

        tmp_mue = np.zeros((F.shape[0], 1))
        tmp_M = np.zeros((F.shape[0], F.shape[0]))

        for main_spec in range(F.shape[0]):

            cur_spec_train_F = train_F[main_spec: main_spec + 1, :].T
            inactive_list = [i for i in range(F.shape[0])]

            # calculating mue
            cur_spec = np.ones((cur_spec_train_F.shape[0], 1))
            tmp_regression_model = linear_model.LinearRegression(fit_intercept = False).fit(cur_spec, cur_spec_train_F)
            tmp_mue[main_spec][0] = tmp_regression_model.coef_[0][0]
            cur_spec_train_F += - (cur_spec * tmp_mue[main_spec][0])

            # calculating the current F on the test set based only on mue
            cur_spec_test = np.ones((test_F.shape[1], 1))
            cur_spec_test_F = cur_spec_test * tmp_mue[main_spec][0]

            # calculate self interaction
            inactive_list.remove(main_spec)
            cur_spec = train_data[main_spec: main_spec + 1, :].T
            tmp_regression_model = linear_model.LinearRegression(fit_intercept = False).fit(cur_spec, cur_spec_train_F)
            tmp_M[main_spec][main_spec] = tmp_regression_model.coef_[0][0]
            cur_spec_train_F += - (cur_spec * tmp_M[main_spec][main_spec])

            # calculating error based on self interactions
            cur_spec_test = test_data[main_spec: main_spec + 1, :].T
            cur_spec_test_F += cur_spec_test * tmp_M[main_spec][main_spec]
            error = sklearn.metrics.mean_squared_error(cur_spec_test_F, test_F[main_spec: main_spec + 1, :].T)

            improve_model = True
            while improve_model == True:
                best_error = np.inf
                for spec_index in inactive_list:
                    cur_spec = train_data[spec_index: spec_index + 1, :].T
                    tmp_regression_model = linear_model.LinearRegression(fit_intercept = False).fit(cur_spec, cur_spec_train_F)
                    tmp_cur_coeff = tmp_regression_model.coef_[0][0]

                    cur_spec_test = test_data[spec_index: spec_index + 1, :].T * tmp_cur_coeff
                    tmp_test_F = cur_spec_test_F + cur_spec_test
                    tmp_error = sklearn.metrics.mean_squared_error(tmp_test_F,
                                                                   test_F[main_spec: main_spec + 1, :].T)
                    if tmp_error < best_error:
                        best_error = tmp_error
                        best_spec = spec_index
                        best_spec_coeff = tmp_cur_coeff

                if (error - best_error) / error >= error_threshold:
                    tmp_M[main_spec][best_spec] = best_spec_coeff
                    inactive_list.remove(best_spec)
                    error = best_error

                    cur_spec = train_data[best_spec: best_spec + 1, :].T
                    cur_spec_train_F += - (cur_spec * best_spec_coeff)

                    cur_spec_test = test_data[best_spec: best_spec + 1, :].T
                    cur_spec_test_F += cur_spec_test * best_spec_coeff
                else:
                    improve_model = False

                if inactive_list == []:
                    improve_model = False

        all_mue.append(tmp_mue)
        all_M.append(tmp_M)

    final_mue = np.zeros((F.shape[0], 1))
    final_M = np.zeros((F.shape[0], F.shape[0]))

    for k in range(F.shape[0]):
        for l in range(F.shape[0]):
            current_M_list = []
            for r in range(repeats):
                current_M_list.append(all_M[r][k][l])
                if med_or_avg == 0:
                    final_M[k][l] = np.median(current_M_list)
                else:
                    final_M[k][l] = np.mean(current_M_list)

    for k in range(F.shape[0]):
        current_mue_list = []
        for r in range(repeats):
            current_mue_list.append(all_mue[r][k][0])
            if med_or_avg == 0:
                final_mue[k][0] = np.median(current_mue_list)
            else:
                final_mue[k][0] = np.mean(current_mue_list)
    return final_mue, final_M

def MSE_gLV(F, Y_mat):
    """
    gets an F matrix and the corresponding Y_matrix (as described earlier)
    returns M, mue and perturbations matrices as a list using linear regression
    """

    tmp_coef = np.matmul(Y_mat, Y_mat.T)
    tmp_coef = np.linalg.inv(tmp_coef)
    tmp_coef = np.matmul(tmp_coef, Y_mat)
    tmp_coef = np.matmul(tmp_coef, F.T)
    m = F.shape[0]
    M = tmp_coef[:m, :]
    M = M.T
    mue = tmp_coef[m: m + 1, :].reshape(1, tmp_coef.shape[1])
    mue = mue.T
    pert_coef = None
    if Y_mat.shape[0] > F.shape[0] + 1:
        pert_coef = tmp_coef[m + 1:, :]
        pert_coef = pert_coef.T
    return M, mue, pert_coef

def coef_matrix_by_sklearn_ridge(Y, F, lambda_for_ridge):
    """
    gets the Y and F matrices as numpy arrays and a lambda as in https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003388#s4
    returns the coef. matrix (where the last column is the self growth rate)
    """
    clf = sklearn.linear_model.Ridge(alpha=lambda_for_ridge)
    clf.fit(Y.T, F.T)
    coef_mat = clf.coef_
    return coef_mat.T


