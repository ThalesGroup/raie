import neurokit2 as nk
import pandas as pd
import sleepecg
import tempbeat as tb

from raie.methods import (
    christov2004,
    elgendi2010,
    engzeemod2012,
    gamboa2008,
    hamilton2002,
    kalidas2017,
    martinez2004,
    neurokit,
    pantompkins1985,
    rodrigues2020,
    sleepecg,
    tempbeat,
)


def concatenate():

    # Load ECGs
    ecgs = [
        "./data/gudb/ECGs.csv",
        "./data/mit_arrhythmia/ECGs.csv",
        "./data/mit_normal/ECGs.csv",
        "./data/ludb/ECGs.csv",
        "./data/fantasia/ECGs.csv",
    ]

    # Load True R-peaks location
    rpeaks = [
        pd.read_csv("./data/gudb/Rpeaks.csv"),
        pd.read_csv("./data/mit_arrhythmia/Rpeaks.csv"),
        pd.read_csv("./data/mit_normal/Rpeaks.csv"),
        pd.read_csv("./data/ludb/Rpeaks.csv"),
        pd.read_csv("./data/fantasia/Rpeaks.csv"),
    ]
    return ecgs, rpeaks


def study_1(output_path):
    results = []
    for method in [
        neurokit,
        pantompkins1985,
        hamilton2002,
        martinez2004,
        christov2004,
        gamboa2008,
        elgendi2010,
        engzeemod2012,
        kalidas2017,
        rodrigues2020,
        tempbeat,
        sleepecg,
    ]:
        print("method: " + method.__name__)
        for i in range(len(rpeaks)):
            print("i: " + str(i))
            data_ecg = pd.read_csv(ecgs[i])
            result = nk.benchmark_ecg_preprocessing(method, data_ecg, rpeaks[i])
            result["Method"] = method.__name__
            results.append(result)
            print(result)

    results = pd.concat(results).reset_index(drop=True)

    results.to_csv(output_path, index=False)


def none(ecg, sampling_rate):
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def mean_detrend(ecg, sampling_rate):
    ecg = nk.signal_detrend(ecg, order=0)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def standardize(ecg, sampling_rate):
    ecg = nk.standardize(ecg)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def study_2(output_path):
    results = []
    for method in [none, mean_detrend, standardize]:
        print("method: " + method.__name__)
        for i in range(len(rpeaks)):
            print("i: " + str(i))
            data_ecg = pd.read_csv(ecgs[i])
            result = nk.benchmark_ecg_preprocessing(method, data_ecg, rpeaks[i])

            result["Method"] = method.__name__
            results.append(result)
            print(result)
    results = pd.concat(results).reset_index(drop=True)

    results.to_csv(output_path, index=False)


# Detrending-based
def polylength(ecg, sampling_rate):
    length = len(ecg) / sampling_rate
    ecg = nk.signal_detrend(ecg, method="polynomial", order=int(length / 2))
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def tarvainen(ecg, sampling_rate):
    ecg = nk.signal_detrend(ecg, method="tarvainen2002")
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def locreg(ecg, sampling_rate):
    ecg = nk.signal_detrend(ecg, method="locreg", window=1 / 0.5, stepsize=0.02)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def rollingz(ecg, sampling_rate):
    ecg = nk.standardize(ecg, window=sampling_rate * 2)
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


# Filtering-based


def study_3(output_path):
    results = []
    for method in [none, polylength, tarvainen, locreg, rollingz]:
        print("method: " + method.__name__)
        result = nk.benchmark_ecg_preprocessing(method, ecgs, rpeaks)
        result["Method"] = method.__name__
        print(result)
        results.append(result)
    results = pd.concat(results).reset_index(drop=True)

    results.to_csv(output_path, index=False)


if __name__ == "__main__":
    ecgs, rpeaks = concatenate()
