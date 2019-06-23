import os
import zipfile
import requests
import tqdm
import numpy as np
import pandas as pd


def fetch_har(extract=True, url=None):
    """
    Fetch UCI HAR Dataset
    :param extract: Bool, if True, extract downloaded file
    :param url: String, for debug
    :return:
    """
    save_dir = os.environ["HOME"] + "/.SensorSignalDatasets/"

    if not url:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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
            zfile.extractall(save_dir)

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

    save_dir = os.environ["HOME"] + "/.SensorSignalDatasets/UCI HAR Dataset/"

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
