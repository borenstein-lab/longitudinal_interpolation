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


def moving_picrtures_parsing(location_M3, location_F4, number_of_top_spec, replace_zero = True):
    """

    :param location_M3:
    :param location_F4:
    :param number_of_top_spec:
    :param replace_zero:
    :return: relative abundances of M3 and F4 (as a tuple) from "moving  pictures of the human microbiome.
    zeros replaced by minimum abundance on the whole dataset if replace_zero == True
    """
    data = pd.read_csv(location_M3)

    data_columns = [int(data.columns[i]) for i in range(1, len(data.columns))]
    data_columns.insert(0, data.columns[0])
    data = pd.read_csv(location_M3, names=data_columns)

    data = data.rename(index=data.iloc[:, 0])
    data = data.iloc[1:, 1:]

    if replace_zero:
        minimum_value = np.min(np.array(data)[np.nonzero(np.array(data))])
        data[data == 0] = minimum_value

    sum_columns = data.sum(axis=0)
    data = data.divide(sum_columns)

    num_of_top_spec = number_of_top_spec

    sum_by_specie = np.sum(data, axis=1)
    sum_by_specie = sum_by_specie.sort_values(ascending=False)
    top_species = data.loc[sum_by_specie.index[0]]
    for i in range(1, data.shape[0]):
        top_species = pd.concat((top_species, data.loc[sum_by_specie.index[i]]), axis=1)
    top_species = top_species.T
    top_species = top_species.iloc[:num_of_top_spec, :]
    others = []
    for i in range(top_species.shape[1]):
        others.append(1 - np.sum(top_species.iloc[:, i]))
    others = pd.DataFrame(others, columns=["others"], index=top_species.columns).T
    top_species_M3 = pd.concat((top_species, others))

    data = pd.read_csv(location_M3)

    data_columns = [int(data.columns[i]) for i in range(1, len(data.columns))]
    data_columns.insert(0, data.columns[0])
    data = pd.read_csv(location_M3, names=data_columns)

    data = data.rename(index=data.iloc[:, 0])
    data = data.iloc[1:, 1:]

    if replace_zero:
        minimum_value = np.min(np.array(data)[np.nonzero(np.array(data))])
        data[data == 0] = minimum_value

    sum_columns = data.sum(axis=0)
    data = data.divide(sum_columns)

    num_of_top_spec = number_of_top_spec

    sum_by_specie = np.sum(data, axis=1)
    sum_by_specie = sum_by_specie.sort_values(ascending=False)
    top_species = data.loc[sum_by_specie.index[0]]
    for i in range(1, data.shape[0]):
        top_species = pd.concat((top_species, data.loc[sum_by_specie.index[i]]), axis=1)
    top_species = top_species.T
    top_species = top_species.iloc[:num_of_top_spec, :]
    others = []
    for i in range(top_species.shape[1]):
        others.append(1 - np.sum(top_species.iloc[:, i]))
    others = pd.DataFrame(others, columns=["others"], index=top_species.columns).T
    top_species_M3 = pd.concat((top_species, others))

    ## from here F4
    data = pd.read_csv(location_F4)

    data_columns = [int(data.columns[i]) - 1 for i in range(1, len(data.columns))]
    data_columns.insert(0, data.columns[0])
    data = pd.read_csv(location_F4, names=data_columns)

    data = data.rename(index=data.iloc[:, 0])
    data = data.iloc[1:, 1:]

    if replace_zero:
        minimum_value = np.min(np.array(data)[np.nonzero(np.array(data))])
        data[data == 0] = minimum_value

    sum_columns = data.sum(axis=0)
    data = data.divide(sum_columns)

    num_of_top_spec = number_of_top_spec

    sum_by_specie = np.sum(data, axis=1)
    sum_by_specie = sum_by_specie.sort_values(ascending=False)
    top_species = data.loc[sum_by_specie.index[0]]
    for i in range(1, data.shape[0]):
        top_species = pd.concat((top_species, data.loc[sum_by_specie.index[i]]), axis=1)
    top_species = top_species.T
    top_species = top_species.iloc[:num_of_top_spec, :]
    others = []
    for i in range(top_species.shape[1]):
        others.append(1 - np.sum(top_species.iloc[:, i]))
    others = pd.DataFrame(others, columns=["others"], index=top_species.columns).T
    top_species_F4 = pd.concat((top_species, others))

    return top_species_M3, top_species_F4

def donors_parsing(data_location, meta_data_loc, number_of_top_spec, replace_zero = True):
    """

    :param data_location:
    :param meta_data_loc:
    :param number_of_top_spec:
    :param replace_zero:
    :return: a dictionary: every individual is a key and his abundance is a df.
    zeros replaced by minimum abundance on each individual if replace_zero is True
    """
    ###bg-0047 collection date is missing. should remove from the metadata file

    data = pd.read_csv(data_location, header=0, delimiter="\t", index_col=0)
    metadata = pd.read_csv(meta_data_loc, sep="\t")
    metadata["Collection_Date"] = pd.to_datetime(metadata["Collection_Date"], format="%d-%m-%Y")
    num_of_top_spec = number_of_top_spec

    patients = []
    patients = [patient[:2] for patient in data.columns if patient[:2] not in patients]
    patients = set(patients)
    data_per_patient = {}
    for patient in patients:
        tmp_data = data.filter(regex=(patient + ".*"))
        tmp_metadata = metadata[metadata["SubjectID"] == patient]
        tmp_dates = []

        if replace_zero:
            minimum_value = np.min(np.array(tmp_data)[np.nonzero(np.array(tmp_data))])
            tmp_data[tmp_data == 0] = minimum_value

        sum_by_specie = np.sum(tmp_data, axis=1)
        sum_by_specie = sum_by_specie.sort_values(ascending=False)
        top_species = tmp_data.loc[sum_by_specie.index[0]]
        for i in range(1, data.shape[0]):
            top_species = pd.concat((top_species, tmp_data.loc[sum_by_specie.index[i]]), axis=1)
        top_species = top_species.T
        top_species = top_species.iloc[:num_of_top_spec, :]
        others = []
        for i in range(top_species.shape[1]):
            others.append(1 - np.sum(top_species.iloc[:, i]))
        others = pd.DataFrame(others, columns=["others"], index=top_species.columns).T
        top_species = pd.concat((top_species, others))
        tmp_data = top_species

        for i in range(0, tmp_data.shape[1]):
            tmp_dates.append(
                tmp_metadata[tmp_metadata["SampleID"] == tmp_data.columns[i]]["Collection_Date"].iloc[0])
        tmp_data.columns = tmp_dates
        tmp_data.sort_index(axis=0, ascending=True, inplace=True)
        tmp_data = tmp_data.groupby(by=tmp_data.columns, axis=1).mean()

        times_since_t0 = [0]
        t0 = tmp_data.columns[0]
        for i in range(1, tmp_data.shape[1]):
            times_since_t0.append(int((tmp_data.columns[i] - t0) / np.timedelta64(1, 'D')))
        tmp_data.columns = times_since_t0
        data_per_patient[patient] = tmp_data
    return data_per_patient


def trosvic_parsing(metadata_loc, feature_table_loc , data_loc , num_of_top_spec, replace_zero = True):
    trosvic_metadata = pd.read_csv(metadata_loc, sep="\t")
    feature_table = pd.read_csv(feature_table_loc, sep="\t")
    all_data = pd.read_csv(data_loc, sep="\t")

    all_data = all_data.merge(feature_table[["Feature ID", "Taxon"]], left_on="OTU ID", right_on="Feature ID",
                              how="left")
    all_data.index = all_data["Taxon"]
    all_data = all_data.drop(["Feature ID", "Taxon", "OTU ID"], axis=1)
    all_data = all_data.groupby(all_data.index, axis=0).sum()

    if replace_zero:
        minimum_value = np.min(np.array(all_data)[np.nonzero(np.array(all_data))])
        all_data[all_data == 0] = minimum_value

    all_data_per_pat = {}

    for i in range(len(all_data.columns)):
        tmp_sample_id = all_data.columns[i]
        tmp_pat_id = trosvic_metadata[trosvic_metadata["Sample_name"] == tmp_sample_id]["Subject_ID"].iloc[0]
        tmp_sample = all_data.iloc[:, i]
        tmp_sample.name = trosvic_metadata[trosvic_metadata["Sample_name"] == tmp_sample_id]["Age_at_Collection"].iloc[
            0]

        if tmp_pat_id in all_data_per_pat.keys():
            all_data_per_pat[tmp_pat_id] = pd.concat((all_data_per_pat[tmp_pat_id], tmp_sample), axis=1)
        else:
            all_data_per_pat[tmp_pat_id] = tmp_sample

    for patient in all_data_per_pat.keys():
        all_data_per_pat[patient].sort_index(axis=1, inplace=True)
        tmp_cols = all_data_per_pat[patient].columns - all_data_per_pat[patient].columns[0]
        all_data_per_pat[patient].columns = tmp_cols
        tmp_tp_sum = all_data_per_pat[patient].sum(axis=0)
        all_data_per_pat[patient] = all_data_per_pat[patient] / tmp_tp_sum

        tmp_spec_sum = all_data_per_pat[patient].sum(axis=1)
        tmp_spec_sum = tmp_spec_sum.sort_values(ascending=False)
        top_spec_cur_pat = all_data_per_pat[patient][all_data_per_pat[patient].index == tmp_spec_sum.index[0]]
        for i in range(1, num_of_top_spec):
            tmp_cur_spec_for_cur_pat = all_data_per_pat[patient][
                all_data_per_pat[patient].index == tmp_spec_sum.index[i]]
            top_spec_cur_pat = pd.concat((top_spec_cur_pat, tmp_cur_spec_for_cur_pat), axis=0)

        top_spec_cur_pat = top_spec_cur_pat.T
        top_spec_cur_pat["others"] = list(1 - top_spec_cur_pat.sum(axis=1))
        top_spec_cur_pat = top_spec_cur_pat.T
        all_data_per_pat[patient] = top_spec_cur_pat
    return all_data_per_pat

