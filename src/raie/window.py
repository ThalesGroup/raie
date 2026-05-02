import pandas as pd
from raie.real_time import workflow


def windowing(
    data, peaks, window_length, step_size, edge_threshold, outputFile, detection_method
):
    """**Adapting the benchmarking function to windows of ECG signal**
    The function relies on indices to delineate the window:
        - start_index: refers to the first sample in the window
        - end_index: refers to the last sample in the window
        - limit_index: refers to the last sample pertaining to the same participant in the same database
    The number of samples in a window is total_samples. It is calculated by multiplying the sample rate by the window length.
    THe function picks ECG slices based on these rules:
        - Start_index should not exceed the number of samples.
        - If the end_index is less or eual to the limit_index, the window is full length.
        - If the end_index exceeds the limit_index while the start_index is still less than the limit_index, the limit_index becomes the end_index and the window is less than the full length.
        This window contains the remaining of the samples from the participant that were not included by the previous window but too few to form a full window.
        - The true Rpeaks are picked from the full ground truth by identifying the peaks that fall in the window.
        - When a window is delineated, the benchmarking function is called and it is given the ECG and Rpeak slices.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing ECG signal. If the input is a slice of a DataFrame, reset_index(drop=True) must be used.
    peaks : pd.DataFrame
        DataFrame containing Rpeak signal
    window_length : int
        Window length in seconds
    step_size : float
        Number of seconds between the start of two consecutive windows, for example:
        If step_size = 1, and window_size = 10, the windows will have 9 seconds overlap.
        If step_size = 10,and window_size = 10, there is no overlap.
        If step_size = 20,and window_size = 10, the next window will skip 10 seconds of data after the end of the previous window.
    edge_threshold : float
        Indicates the time in seconds of the current window after which the found peaks are discarded.
        Example: if the window length is 10 seconds and the edge threshold is 9 seconds, then the peaks after 9 seconds from the start of the window are discarded.
                Only the peaks included in the 9 seconds are kept.
    outputFile : string
        Path to a directory to save a CSV file containing the results.
    detection_method : string
        Indicates the peak detection algorithm adopted. The algorithms available are:
        christov2004, elgendi2010, engzeemod2012, gamboa2008, hamilton2002, kalidas2017, martinez2004, neurokit, pantompkins1985, rodrigues2020, sleepecg, tempbeat.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing the results of the windowed benchmarking.

    Example
    --------
    dataFile = pd.read_csv("ECG_PATH.csv")  #ECG dataframe
    rpeaksFile = pd.read_csv("RPEAK_PATH.csv") #Rpeaks annotation dataframe
    results_windowing = windowing(dataFile, rpeaksFile, window_length=10, step_size=1, edge_threshold=9)
    results_windowing.to_csv("OUTPUT_PATH.csv", index=False)

    """

    count = 0
    start_index = 0
    last_peak_found = 0
    total_samples = window_length * data.Sampling_Rate[start_index]
    end_index = start_index + total_samples
    limit_index = (
        data["Sample"]
        .loc[
            (data["Participant"] == data.Participant[start_index])
            & (data["Database"] == data.Database[start_index])
        ]
        .idxmax()
    )

    all_windows = []
    while limit_index < data.shape[0]:

        while end_index <= limit_index:

            ecg_slice = data.iloc[start_index:end_index]
            rpeaks_slice = peaks.loc[
                (peaks["Participant"] == data.Participant[start_index])
                & (peaks["Database"] == data.Database[start_index])
                & (peaks["Rpeaks"] <= data.Sample[end_index])
                & (peaks["Rpeaks"] >= data.Sample[start_index])
            ]

            edge_threshold_samples = (
                start_index + edge_threshold * data.Sampling_Rate[start_index]
            )
            window_result, last_peak_found = workflow(
                ecgs=ecg_slice,
                rpeaks=rpeaks_slice,
                window_length=window_length,
                step_size=step_size,
                count=count,
                last_peak_found=last_peak_found,
                edge_threshold=edge_threshold_samples,
                algorithm=detection_method,
            )
            all_windows.append(window_result)
            count = count + 1

            start_index = start_index + int(step_size * data.Sampling_Rate[start_index])
            end_index = start_index + total_samples

        if start_index < limit_index:

            end_index = limit_index
            ecg_slice = data.iloc[start_index : limit_index + 1]
            rpeaks_slice = peaks.loc[
                (peaks["Participant"] == data.Participant[start_index])
                & (peaks["Database"] == data.Database[start_index])
                & (peaks["Rpeaks"] <= data.Sample[end_index])
                & (peaks["Rpeaks"] >= data.Sample[start_index])
            ]

            edge_threshold_samples = (
                start_index + edge_threshold * data.Sampling_Rate[start_index]
            )
            window_result, last_peak_found = workflow(
                ecgs=ecg_slice,
                rpeaks=rpeaks_slice,
                window_length=window_length,
                step_size=step_size,
                count=count,
                last_peak_found=last_peak_found,
                edge_threshold=edge_threshold_samples,
                algorithm=detection_method,
            )
            count = count + 1
            all_windows.append(window_result)

            if limit_index + 1 < data.shape[0]:
                start_index = limit_index + 1
                total_samples = window_length * data.Sampling_Rate[start_index]
                end_index = start_index + total_samples
                limit_index = (
                    data["Sample"]
                    .loc[
                        (data["Participant"] == data.Participant[start_index])
                        & (data["Database"] == data.Database[start_index])
                    ]
                    .idxmax()
                )
                last_peak_found = 0
            else:
                limit_index = data.shape[0] + 1

    all_windows = pd.concat(all_windows).reset_index(drop=True)
    all_windows.to_csv(outputFile, index=False)

    return all_windows
