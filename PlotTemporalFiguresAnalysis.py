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
import copy

def plot_time_difference_by_method_bc_dist(times, all_methods_bc,
                                           min_samples_to_draw=10, subplot_length=5, subplot_width=3):
    time_differece = [times[i] - times[i - 1] for i in range(1, len(times))]
    time_differece_sorted = copy.deepcopy(time_differece)
    time_differece_sorted.sort()
    time_differece_sorted = set(time_differece_sorted)

    time_dif_dict = {}

    for i in set(time_differece):
        time_dif_dict[i] = []

    fig = plt.figure()

    cur_subplot = 0
    for method in all_methods_bc:
        cur_subplot += 1
        tmp_time_dif_dict = copy.deepcopy(time_dif_dict)
        cur_tmp_bc_results = all_methods_bc[method]
        for i in range(len(cur_tmp_bc_results)):
            tmp_time_dif_dict[time_differece[i]].append(cur_tmp_bc_results[i])

        time_difs_to_draw = []
        for i in time_differece_sorted:
            if len(tmp_time_dif_dict[i]) >= min_samples_to_draw:
                time_difs_to_draw.append(i)

        cur_plt = fig.add_subplot(subplot_length, subplot_width, cur_subplot)
        sns.boxplot(data=[tmp_time_dif_dict[time_dif] for time_dif in time_difs_to_draw],
                    showfliers=False, whis=[10, 90],
                    )
        plt.xticks([i for i in range(len(time_difs_to_draw))], time_difs_to_draw)
        plt.title(method)
        plt.ylim(0, 1)
    fig.set_size_inches(21.5, 18.5)
    sns.set()
    plt.show()

def plot_ADF_vs_average_bc(bray_curtis_per_patient, adf_per_patient, subplot_length = 5, subplot_width = 3):
    fig = plt.figure()
    cur_subplot = 0
    all_patients = list(bray_curtis_per_patient.keys())
    for method in bray_curtis_per_patient[all_patients[0]]:
        cur_subplot += 1
        cur_plt = fig.add_subplot(subplot_length, subplot_width, cur_subplot)
        for patient in all_patients:
            if "donors" in patient:
                plt.plot(adf_per_patient[patient], np.average(bray_curtis_per_patient[patient][method]),
                         marker=".", color="red")
            elif "trosvic" in patient:
                plt.plot(adf_per_patient[patient], np.average(bray_curtis_per_patient[patient][method]),
                         marker=".", color="orange")
            else:
                plt.plot(adf_per_patient[patient], np.average(bray_curtis_per_patient[patient][method]),
                         marker=".", color="blue")
        plt.ylim(0, 1)
        plt.xlim(-9, -2)
        plt.title(method)
        sns.set()

    fig.set_size_inches(21.5, 18.5)
    plt.show()
