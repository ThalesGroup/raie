import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# function to plot database
def plot_ecg_peaks(path_to_ecg, path_to_peaks, slice_length):
    """**Plot ECG signal and annotated peaks.**
    The function plots a slice of ECG signal with its annotated peaks.

    Parameters
    ----------
    path_to_ecg : string
        String containing the path to the ECG data.
    path_to_peaks : string
        String containing the path to the ECG peaks annotation.
    slice_length : int
        Integer indicating the length of the data slice in seconds.
    """
    # fetching the sampling rate
    sampling_rate = pd.read_csv(path_to_ecg, nrows=1)
    sampling_rate = sampling_rate["Sampling_Rate"][0]

    # fetching a slice of data
    data_slice = slice_length * sampling_rate
    ecg = pd.read_csv(path_to_ecg, nrows=data_slice + 1)
    rpeaks = pd.read_csv(path_to_peaks, nrows=data_slice + 1)
    ecg = ecg.iloc[0:data_slice]
    rpeaks = rpeaks.iloc[0:data_slice].loc[
        (rpeaks["Rpeaks"] < data_slice)
        & (rpeaks["Database"] == rpeaks["Database"][0])
        & (rpeaks["Participant"] == rpeaks["Participant"][0])
    ]

    # fetching corresponding peaks
    ecgs = ecg["ECG"]
    peaks = []
    peak_time = []

    for i in range(rpeaks["Rpeaks"].shape[0]):
        peaks.append(
            (
                float(
                    ecg["ECG"].loc[
                        (ecg["Participant"] == rpeaks["Participant"][i])
                        & (ecg["Database"] == rpeaks["Database"][i])
                        & (ecg["Sample"] == rpeaks["Rpeaks"][i])
                    ]
                )
            )
        )
        peak_time.append(float(rpeaks["Rpeaks"][i] * (1 / sampling_rate)))

    # plotting
    time = np.arange(0, slice_length, 1 / sampling_rate)
    plt.plot(time, ecgs, c="blue")
    plt.scatter(peak_time, peaks, c="red", marker="D")
    for i, j in zip(peak_time, peaks):
        plt.annotate(str(j), xy=(i, j))
    plt.show()


if __name__ == "__main__":

    # specify your path
    path = "../../data/mit_arrhythmia"
    # Edit the below variables and run the code

    path_to_ecg = path + "/ECGs.csv"
    path_to_peaks = path + "/Rpeaks.csv"
    slice_length = 30  # ECG signal length in seconds

    plot_ecg_peaks(path_to_ecg, path_to_peaks, slice_length)
