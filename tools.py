import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.cluster import contingency_matrix
import warnings
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import os
import platform
import json
from shutil import copyfile


def f_get_minibatch(mb_size, x, y):
    idx = range(np.shape(x)[0])
    idx = random.sample(idx, mb_size)

    x_mb   = x[idx].astype(float)
    y_mb   = y[idx].astype(float)

    return x_mb, y_mb


def f_get_prediction_scores(y_true_, y_pred_):
    if np.sum(y_true_) == 0:  # no label for running roc_auc_curves
        auroc_ = -1.
        auprc_ = -1.
    else:
        auroc_ = roc_auc_score(y_true_, y_pred_)
        auprc_ = average_precision_score(y_true_, y_pred_)
    return (auroc_, auprc_)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def build_labels(main_path, patientData4Visits):
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    scores = clinical_score.iloc[:, 26:49]  # 21:26
    scores = pd.DataFrame(scores)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(scores)
    np.set_printoptions(precision=3)  # Setting precision for the output
    scores = rescaledX
    scores = pd.DataFrame(scores)
    scores.columns = ['RAVLT_immediate', 'RAVLT_learning', 'RAVLT_forgetting', 'RAVLT_perc_forgetting', 'LDELTOTAL',
                      'DIGITSCOR', 'TRABSCOR', 'FAQ', 'MOCA', 'EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan',
                      'EcogPtOrgan', 'EcogPtDivatt', 'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat',
                      'EcogSPPlan', 'EcogSPOrgan', 'EcogSPDivatt', 'EcogSPTotal']
    header = clinical_score.iloc[:, :15]
    scores = pd.concat([header, scores], axis=1)
    labels = []
    for i in range(len(patientData4Visits)):
        patient = []
        patientIndex = patientData4Visits[i][0][3]
        print("PTID = {} ".format(patientIndex), end="")
        for j in range(2):  # range(4)
            visit = []
            if j == 0:  # first
                viscode = clinical_score.loc[
                    (clinical_score["PTID"] == patientIndex) & (pd.notnull(clinical_score["EcogPtMem"]))][
                    "VISCODE"].values[0]
                print("FIRST = {} ".format(viscode), end="")
            else:  # last
                viscode = clinical_score.loc[
                    (clinical_score["PTID"] == patientIndex) & (pd.notnull(clinical_score["EcogPtMem"]))][
                    "VISCODE"].values[-1]
                print("LAST = {}".format(viscode))

            columns = ['EcogPtMem', 'EcogPtLang', 'EcogPtVisspat', 'EcogPtPlan', 'EcogPtOrgan', 'EcogPtDivatt',
                       'EcogPtTotal', 'EcogSPMem', 'EcogSPLang', 'EcogSPVisspat', 'EcogSPPlan', 'EcogSPOrgan',
                       'EcogSPDivatt', 'EcogSPTotal']
            for index in range(14):  # for index in range(14):
                a = scores.loc[(clinical_score["PTID"] == patientIndex) & (clinical_score["VISCODE"] == viscode)][
                    columns[index]]  # ENZE updated
                a = a.values[0] if len(a) > 0 else np.nan  # ENZE updated
                visit.append(a)
            patient.append(visit)
        labels.append(patient)
    return np.asarray(labels)


def get_data_y(data_y):
    for i in range(len(data_y)):
        for k in range(14):
            scores = []
            notnan = []
            for j in range(2):
                scores.append(data_y[i][j][k])
            for l in range(2):
                if not math.isnan(scores[l]):
                    notnan.append(l)
            for p in range(2):
                if math.isnan(scores[p]):
                    mindiff = 10
                    minindex = 0
                    for h in notnan:
                        if abs(h - p) < mindiff:
                            mindiff = abs(h - p)
                            minindex = h
                    data_y[i][p][k] = data_y[i][minindex][k]
    for i in range(len(data_y)):
        for j in range(2):
            for k in range(14):
                if math.isnan(data_y[i][j][k]):
                    data_y[i][j][k] = 0.5
    return np.asarray(data_y)


def get_data_x(patientData4Visits):
    data_x = []
    # data_y = []

    for i in patientData4Visits:
        # print(len(i))
        patientArray = []
        patientLabels = []
        for j in range(len(i)):
            if j == 0:
                timeDiff = 0
            else:
                timeDiff = minus_to_month(i[j - 1][2], i[j][2])
            featureArray = np.insert(i[j][0], 0, timeDiff)
            patientArray.append(featureArray)
            patientLabels.append(i[j][1])
        patientArray = np.asarray(patientArray)
        data_x.append(patientArray)
    return np.asarray(data_x)


def string_to_stamp(string, string_format="%Y%m%d"):
  string = str(string)
  return time.mktime(time.strptime(string, string_format))


def minus_to_month(str1, str2):
  return (string_to_stamp(str2) - string_to_stamp(str1)) / 86400 / 30


def load_data(main_path, file_name):
    return np.load(main_path + file_name, allow_pickle=True)


def draw_heat_map(data, s=2):
    data = np.asarray(data)
    data_normed = np.abs((data - data.mean(axis=0)) / data.std(axis=0))
    data_normed = data_normed / s
    xlabels = ["MMSE", "CDRSB", "ADAS13"]
    ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    plt.figure()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(data_normed, interpolation='nearest', cmap=plt.cm.hot, vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45)
    plt.yticks(np.arange(len(ylabels)), ylabels)
    plt.title('DPS-Net')
    plt.show()


def draw_heat_map_2(data1, data2, s=2):
    data1 = np.asarray(data1)
    data1_normed = np.abs((data1 - data1.mean(axis=0)) / data1.std(axis=0))
    data1_normed = data1_normed / s
    data2 = np.asarray(data2)
    data2_normed = np.abs((data2 - data2.mean(axis=0)) / data2.std(axis=0))
    data2_normed = data2_normed / s
    xlabels = ["MMSE", "CDRSB", "ADAS13"]
    ylabels = ["Subtype #{0}".format(i) for i in range(1, 6)]
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title("k-means")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    im = ax.imshow(data1_normed, cmap=plt.cm.hot, vmin=0, vmax=1)
    cb = plt.colorbar(im, shrink=0.7)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Intra-cluster variance", fontdict={"rotation": 270})
    ax = fig.add_subplot(122)
    ax.set_title("DPS-Net")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    im = ax.imshow(data2_normed, cmap=plt.cm.hot, vmin=0, vmax=1)
    cb = plt.colorbar(im, shrink=0.7)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])
    cb.set_label("Intra-cluster variance", fontdict={"rotation": 270})
    plt.tight_layout()
    plt.show()


def get_engine():
    if platform.system().lower() == "linux":
        return "openpyxl"
    elif platform.system().lower() == "windows":
        return None
    return None


def get_heat_map_data(K, patient_data, label, data_path):
    dim_0 = len(patient_data)
    dim_1 = len(patient_data[0])
    label_match = np.asarray(label).reshape(dim_0 * dim_1)
    patient_data_match = []
    for i in range(dim_0):
        for j in range(dim_1):
            patient_data_match.append([patient_data[i][j][3], patient_data[i][j][2]])
    data = pd.read_excel(data_path, engine=get_engine())  # main_path + 'DPS_ATN/MRI_information_All_Measurement.xlsx'
    target_labels = ['EcogPtMem','EcogPtLang','EcogPtVisspat','EcogPtPlan','EcogPtOrgan','EcogPtDivatt','EcogPtTotal','EcogSPMem','EcogSPLang','EcogSPVisspat','EcogSPPlan','EcogSPOrgan','EcogSPDivatt','EcogSPTotal'] #["MMSE", "CDRSB", "ADAS13"]
    data = data[["PTID", "EXAMDATE"] + target_labels]
    # data["EXAMDATE"] = data["EXAMDATE"].astype(str)
    result = []
    for i in range(K):
        dic = dict()
        for one_target_label in target_labels:
            dic[one_target_label] = []
        for j in range(dim_0 * dim_1):
            if label_match[j] != i:
                continue
            for one_target_label in target_labels:
                tmp = data.loc[(data["PTID"] == patient_data_match[j][0]) & (data["EXAMDATE"] == patient_data_match[j][1])][one_target_label].values[0]
                if math.isnan(tmp):
                    print("bad in matching PTID = '{}'".format(patient_data_match[j][0]), " EXAMDATE = '{}'".format(patient_data_match[j][1]))
                    continue
                tmp_list = dic.get(one_target_label)
                tmp_list.append(float(tmp))
                dic[one_target_label] = tmp_list
        result.append([np.var(np.asarray(dic[one_target_label])) for one_target_label in target_labels])
    return result


def judge_good_train(labels, heat_map_data, const_cn_ad_labels):
    dic = dict()
    for i in range(5):
        dic[i] = 0
    for row in labels:
        for item in row:
            dic[item if (type(item) == int or type(item) == np.int32) else item[0]] += 1
    distribution = np.asarray([dic.get(i) for i in range(5)])
    label_strings = create_label_string(labels, const_cn_ad_labels)
    distribution_string = "/".join(["{}({})".format(x, y) for x, y in zip(distribution, label_strings)])
    param_cluster_std = distribution.std()
    fourteen_sums = np.asarray(heat_map_data).sum(axis=0) # three_sums = np.asarray(heat_map_data).sum(axis=0)

    judge = 0
    param_dic = dict()
    param_dic["Cluster_std"] = param_cluster_std
    for i, one_label in enumerate(['EcogPtMem','EcogPtLang','EcogPtVisspat','EcogPtPlan','EcogPtOrgan','EcogPtDivatt','EcogPtTotal','EcogSPMem','EcogSPLang','EcogSPVisspat','EcogSPPlan','EcogSPOrgan','EcogSPDivatt','EcogSPTotal']):
        param_dic[one_label + "_var"] = fourteen_sums[i]
    return judge, param_dic, distribution_string


def save_record(main_path, index, distribution_string, judge, judge_params, comments, params):
    with open(main_path + "/record/record.csv", "a") as f:
        f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(
            index,
            judge,
            distribution_string,
            judge_params.get("Cluster_std"),
            judge_params.get("EcogPtMem_var"),
            judge_params.get("EcogPtLang_var"),
            judge_params.get("EcogPtVisspat_var"),
            judge_params.get("EcogPtPlan_var"),
            judge_params.get("EcogPtOrgan_var"),
            judge_params.get("EcogPtDivatt_var"),
            judge_params.get("EcogPtTotal_var"),
            judge_params.get("EcogSPMem_var"),
            judge_params.get("EcogSPLang_var"),
            judge_params.get("EcogSPVisspat_var"),
            judge_params.get("EcogSPPlan_var"),
            judge_params.get("EcogSPOrgan_var"),
            judge_params.get("EcogSPDivatt_var"),
            judge_params.get("EcogSPTotal_var"),
            comments
        ))
        f.write(",".join([str(params.get(one_key)) for one_key in list(params.keys())]))
        f.write("\n")


def get_k_means_result(main_path):
    atn_kmeans_cluster = np.load(main_path + 'data/atn_kmeans_cluster.npy')
    atn_kmeans_cluster = np.asarray(atn_kmeans_cluster)
    enze_patient_data = np.load(main_path + "data/enze_patient_data_new.npy", allow_pickle=True)
    enze_patient_data = np.asarray(enze_patient_data)
    res1 = get_heat_map_data(5, enze_patient_data, atn_kmeans_cluster, main_path + 'data/MRI_information_All_Measurement.xlsx')
    judge, params, distribution_string = judge_good_train(atn_kmeans_cluster, res1)
    # print(judge, params, distribution_string)
    return res1


def get_start_index(main_path):
    df = pd.read_csv(main_path + "record/record.csv")
    start_index = sorted(list(df["Id"]), key=lambda x: x)[-1] + 1
    return start_index


def get_ac_tpc_result(main_path, index):
    labels = np.load(main_path + 'saves/{}/proposed/trained/results/labels.npy'.format(index))
    enze_patient_data = np.load(main_path + "data/enze_patient_data_new.npy", allow_pickle=True)
    res = get_heat_map_data(5, enze_patient_data, labels, main_path + 'data/MRI_information_All_Measurement.xlsx')
    return res


def get_cn_ad_labels(main_path, pt_id_list):
    clinical_score = pd.read_excel(main_path + 'data/MRI_information_All_Measurement.xlsx', engine=get_engine())
    cn_ad_labels = []
    dic = dict()
    for i in range(320):  # [148*148，[label tuple]，VISCODE，patientID]
        first_scan_label = list(clinical_score[clinical_score["PTID"] == pt_id_list[i]]["DX"])[0]
        last_scan_label = list(clinical_score[clinical_score["PTID"] == pt_id_list[i]]["DX"])[-1]
        for scan_label in [first_scan_label, last_scan_label]:
            if scan_label not in dic:
                dic[scan_label] = 1
            else:
                dic[scan_label] += 1
        cn_ad_labels.append([first_scan_label, last_scan_label])
    # print(dic)
    return np.asarray(cn_ad_labels)


def create_label_string(cluster_labels, const_cn_ad_labels):
    dic_list = []
    for i in range(5):
        dic = dict()
        dic["AD"] = 0
        dic["CN"] = 0
        dic["Other"] = 0
        dic_list.append(dic)

    for i in range(320):
        for j in range(2):
            tmp_cluster_id = cluster_labels[i][j] if type(cluster_labels[i][j]) == int else cluster_labels[i][j][0]
            if const_cn_ad_labels[i][j] == "AD":
                dic_list[tmp_cluster_id]["AD"] += 1
            elif const_cn_ad_labels[i][j] == "CN":
                dic_list[tmp_cluster_id]["CN"] += 1
            else:
                dic_list[tmp_cluster_id]["Other"] += 1
    # for dic in dic_list:
    #     print(dic)
    return ["{}+{}+{}".format(dic.get("CN"), dic.get("AD"), dic.get("Other")) for dic in dic_list]


def initial_record():
    if not os.path.exists("record/record.csv"):
        copyfile("record/record_0.csv", "record/record.csv")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main_path = os.path.dirname(os.path.abspath("__file__")) + "/"
    enze_patient_data = np.load(main_path + "data/enze_patient_data_new.npy", allow_pickle=True)
    pt_id_list = [item[0][3] for item in enze_patient_data]
    # print(pt_id_list)
    cn_ad_labels = get_cn_ad_labels(main_path, pt_id_list)
    labels = np.load(main_path + 'saves/{}/proposed/trained/results/labels.npy'.format(1146))
    print(create_label_string(labels, cn_ad_labels))
    # p = {
    #     "Cluster_std": 30,
    #     "MMSE_var": 50,
    #     "CDRSB_var": 20,
    #     "ADAS_var": 40
    # }
    # save_record(main_path, 10, 0, p, "test")
    # res1 = get_k_means_result(main_path)
    # res2 = get_ac_tpc_result(main_path, 1146)
    # draw_heat_map_2(res1, res2)
    # data = pd.read_excel("data/MRI_information_All_Measurement.xlsx", engine="openpyxl")
    # target_labels = ["MMSE", "CDRSB", "ADAS13"]
    # data = data[["PTID", "EXAMDATE"] + target_labels]
    # print(data)
    # print(data.dtypes)
    # data["PTID"] = data["PTID"].astype(str)
    # data["EXAMDATE"] = data["EXAMDATE"].astype(str)
    # print(data)
    # print(data.dtypes)
    # print(data.loc[(data["PTID"] == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])
    # print(data[(str(data["PTID"]) == "013_S_2389") & (data["EXAMDATE"] == int("20171130"))]["MMSE"])

    # tmp = list(data.loc[(data["PTID"] == "002_S_0413")]["EXAMDATE"])
    # print("'{}'".format(tmp[-1]), type(tmp[-1]))
    pass










