import os
import zipfile
import requests
import tqdm
import numpy as np
import pandas as pd
from scipy import stats

SAVE_DIR = os.environ["HOME"] + "/.SensorSignalDatasets/"


def fetch_mhealth(extract=True, url=None):
    if not url:
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00319/MHEALTHDATASET.zip"

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = url.split("/")[-1]

    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    print("Downloading UCI MHEALTH Dataset ...")
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)


def __load_per_label(original, label, window_size, overlap_rate=0.5):
    data = original[original[23] == label].iloc[:, :23]
    targets = original[original[23] == label].iloc[:, 23]

    start_idx = 0
    end_idx = window_size
    data_array = []
    target_array = []

    while True:
        d = data[start_idx:end_idx]
        t = targets[start_idx:end_idx]
        t = stats.mode(t)[0]

        if len(data) == end_idx:
            break

        start_idx = int(window_size * overlap_rate) + start_idx
        end_idx = start_idx + window_size

        if end_idx > len(data):
            start_idx = start_idx - (end_idx - len(data))
            end_idx = len(data)

        data_array.append(d)
        target_array.append(t)

    data_array = np.dstack(data_array).transpose((2, 0, 1))  # [samples, time-steps, features]
    target_array = np.dstack(target_array).transpose((2, 1, 0))
    target_array = target_array.reshape(-1, target_array.shape[1])

    return data_array, target_array


def __load_train_mhealth(targets=None, window_size=128):
    save_dir = SAVE_DIR + "MHEALTHDATASET/"

    if not targets:
        targets = ["mHealth_subject1.log", "mHealth_subject2.log", "mHealth_subject3.log", "mHealth_subject4.log",
                   "mHealth_subject5.log", "mHealth_subject6.log", "mHealth_subject7.log"]

    tmp = [save_dir + i for i in targets]
    dataframes = [pd.read_csv(log_file, header=None, delim_whitespace=True) for log_file in tmp]

    all_data = []
    all_labels = []
    for i in dataframes:
        i = i[i[23] != 0]

        for label in range(1, 13):
            X, y = __load_per_label(i, label, window_size=window_size)
            all_data.append(X)
            all_labels.append(y)

    all_data = np.vstack(all_data)
    all_labels = np.vstack(all_labels)

    return all_data, all_labels


def load_mhealth(window_size=128):
    save_dir = SAVE_DIR + "MHEALTHDATASET/"

    if not os.path.isdir(save_dir):
        fetch_mhealth()

    x_train, y_train = __load_train_mhealth(window_size=window_size)
    # TODO
    # 訓練用は 0 ~ 13 が出来るようオプションをつける。
    x_test, y_test = __load_train_mhealth(
        window_size=window_size,
        targets=["mHealth_subject8.log", "mHealth_subject9.log", "mHealth_subject10.log"]
    )
    return (x_train, y_train), (x_test, y_test)


def fetch_har(extract=True, url=None):
    """
    Fetch UCI HAR Dataset
    :param extract: Bool, if True, extract downloaded file
    :param url: String, for debug
    :return:
    """

    if not url:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    filename = url.split("/")[-1]

    file_size = int(requests.head(url).headers["content-length"])

    r = requests.get(url, stream=True)
    pbar = tqdm.tqdm(total=file_size, unit="B", unit_scale=True)

    print("Downloading UCI HAR Dataset ...")
    with open(filename, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))

        pbar.close()

    if extract:
        with zipfile.ZipFile(filename) as zfile:
            zfile.extractall(SAVE_DIR)

    os.remove(filename)


def __load_file(filepath):
    df = pd.read_csv(filepath, header=None, delim_whitespace=True)
    return df.values


def __load_group(filenames):
    loaded = []
    for name in filenames:
        data = __load_file(name)
        loaded.append(data)

    loaded = np.dstack(loaded)
    return loaded


def __loaded_per_types(dirpath):
    types = ["train", "test"]
    path_to_raw = "Inertial Signals/"

    def load_raw_data(file_type):
        datasets = [
            "total_acc_x_" + file_type + ".txt", "total_acc_y_" + file_type + ".txt", "total_acc_z_" + file_type+".txt",
            "body_acc_x_" + file_type + ".txt", "body_acc_y_" + file_type + ".txt", "body_acc_z_" + file_type+".txt",
            "body_gyro_x_" + file_type + ".txt", "body_gyro_y_" + file_type + ".txt", "body_gyro_z_" + file_type+".txt"
        ]

        tmp = [dirpath + file_type + "/" + path_to_raw + i for i in datasets]

        X = __load_group(tmp)
        return X

    raw_datasets = []
    for mode in types:
        data = load_raw_data(mode)
        target = __load_file(dirpath + mode + "/y_" + mode + ".txt")

        raw_datasets.append((data, target))

    return raw_datasets[0], raw_datasets[1]


def load_har(raw=False):
    save_dir = SAVE_DIR + "UCI HAR Dataset/"

    if not os.path.isdir(save_dir):
        fetch_har()

    if raw:
        (x_train, y_train), (x_test, y_test) = __loaded_per_types(save_dir)

    else:
        x_train = __load_file(save_dir + "train/X_train.txt")
        y_train = __load_file(save_dir + "train/y_train.txt")

        x_test = __load_file(save_dir + "test/X_test.txt")
        y_test = __load_file(save_dir + "test/y_test.txt")

    return (x_train, y_train), (x_test, y_test)
