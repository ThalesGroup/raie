from math import ceil
from statistics import mean, median
import time

import numpy as np
import pandas as pd

from benchmark import benchmark_ecg_preprocessing
from methods import (
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


def find_Rpeaks(ecgs, rpeaks, method):
    """Detect Rpeaks in ECG data.
    The function detects Rpeaks in ECG data using the given peak detection algorithm.

    Parameters
    ----------
    ecgs : pd.DataFrame
        DataFrame containing ECG signal. If the input is a slice of a DataFrame, reset_index(drop=True) must be used.
    rpeaks : pd.DataFrame
        DataFrame containing Rpeak signal
    method : function
        Function indicating the peak detection algorithm to be used.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the results of the Rpeaks detection.
    """
    result = benchmark_ecg_preprocessing(method, ecgs, rpeaks)
    result["Method"] = method.__name__

    # Map the found Rpeaks to the actual sample number
    ecgs.reset_index(drop=True, inplace=True)

    predicted = []
    if isinstance(result["Found"][0], np.float64):
        predicted.append(np.nan)
    elif len(result["Found"][0]) == 1:
        if np.isnan(result["Found"][0]):
            predicted.append(np.nan)
    else:
        for x in result["Found"][0]:
            predicted.append(ecgs["Sample"][x])
    result["Predicted"] = [predicted]
    return result

def add_remove(predicted, add, remove):
    """Add and remove lists from a list.
    This function adds the element of a list and removes the elements of another list.

    Parameters
    ----------
    predicted: list
        List to add to and to remove from.
    add: list
        List containing all the elements to add to predicted.
    remove: list
        List containing all the elements to remove from predicted.


    Returns
    --------
    list
        A final list after the modifications
    """
    result = list(set(predicted) - set(remove))
    result = result + add
    result.sort()
    return result

def insert_peaks(first, last, interval):
    """Insert peaks.
    This function insert peaks according to an interval.

    Parameters
    ----------
    first: int
        Starting peak.
    last: int
        Ending peak.
    interval: float
        The R-R interval used to add peaks..

    Returns
    --------
    list
        A list containing the peaks to add in the large gap.
    list
        A list containing the peaks to remove.
    """
    add = []
    remove = []
    interval_int = int(ceil(interval))
    for i in range(first, int(ceil(last + 1 - interval)), interval_int):
        add.append(int(i + interval))
    remove.append(last)
    return add, remove

def find_small_gaps(predicted, threshold):
    """Find small gaps.
    This function finds small gaps in ECG data and determines the peaks to remove in case of small gaps.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    threshold: float
        Indicates the value of the largest small gap rejected.

    Returns
    --------
    list
        A list contanining the peaks to remove due to small gaps.
    """
    remove = []
    for i in range(0, len(predicted) - 1):
        gap = abs(predicted[i + 1] - predicted[i])
        if gap <= threshold:
            remove.append(predicted[i + 1])
            i = i + 1
    return remove

def find_large_gaps(predicted, threshold, interval):
    """Find large gaps.
    This function finds large gaps in ECG data. Then, it determines the peaks to add by deviding the large gap by the given R-R interval value. 
    Also, it indicates that the last peak should be removed by returning it in the remove list.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    threshold: float
        Indicated the value of the smallest large gap rejected.
    interval: float
        The R-R interval used to add peaks in the large gaps.

    Returns
    --------
    list
        A list containing the peaks to add in the large gap.
    list
        A list containing the peaks to remove.
    """
    add = []
    remove = []

    for i in range(0, len(predicted) - 1):
        gap = predicted[i + 1] - predicted[i]
        if gap >= threshold:
            insert, delete = insert_peaks(predicted[i], predicted[i + 1], interval)
            add = add + insert
            remove = remove + delete
    return add, remove

def fix_gaps(predicted, small_gap, large_gap, interval):
    """Fix small and lare gaps.
    This function removes small gaps in ECG data and adds the necessary peaks in case of large gaps.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    small_gap: float
        Indicates the value of the largest small gap rejected.
    large_gap: float
        Indicated the value of the smallest large gap rejected.
    interval: float
        The R-R interval used to add peaks in the large gaps.

    Returns
    --------
    list
        A final list after the modifications
    """
    delete = find_small_gaps(predicted, small_gap)
    predicted = add_remove(predicted, [], delete)

    add, remove = find_large_gaps(predicted, large_gap, interval)
    predicted_fixed = add_remove(predicted, add, remove)
    return predicted_fixed

def get_stat_info(predicted, mode="mean"):
    """Compute statistical information.
    THis function computes the mean or the median of R-R intervals depending to the mode chosen.
    Parameters
    ----------
    predicted: list
        List of detected ECG R-R intervals.
    mode: string
        A string indicating whether to calculate the mean or the median of predicted.

    Returns
    --------
    float
        The value of the median or the mean of a list.
    """
    if predicted == []:
        return 0

    if mode == "mean":
        return mean(predicted)
    elif mode == "median":
        return median(predicted)

def get_RR_intervals(predicted):
    """Compute R-R intervals.
    This function compute R-R intervals from a list of R-peaks.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.

    Returns
    --------
    list
        A list containing the R-R intervals.
    """
    RR = []
    for i in range(0, len(predicted) - 1):
        RR.append(abs(predicted[i + 1] - predicted[i]))
    return RR

def find_previous_peaks(predicted, last_peak_found):
    """Find peaks preceding a given peak.
    This function finds previous peaks that happened before the last peak found in the previous window.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    last_peak_found: int
        The sample number of the last peak found in the previous window.

    Returns
    --------
    list
        A list containing the peaks to remove.
    """
    peaks_to_remove = []
    for i in range(0, len(predicted)):
        if predicted[i] <= last_peak_found:
            peaks_to_remove.append(predicted[i])

    return peaks_to_remove

def find_peaks_beyond_edge(predicted, edge_threshold):
    """Find peaks exceeding a given edge threshold.
    This function finds previous peaks that happened after an edge threshold.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    edge_thereshold: float
        Indicates the time in seconds of the current window after which the found peaks are discarded.
        Example: if the window length is 10 seconds and the edge threshold is 9 seconds, then the peaks after 9 seconds from the start of the window are discarded.
                Only the peaks included in the 9 seconds are kept.

    Returns
    --------
    list
        A list containing the peaks to remove.
    """
    peaks_to_remove = []
    for i in range(0, len(predicted)):
        if predicted[i] > edge_threshold:
            peaks_to_remove.append(predicted[i])

    return peaks_to_remove

def remove_previous_peaks(predicted, last_peak_found):
    """Remove the previous peaks and add the last peak found.
    This function removes the previous peaks found in the find_previous_peaks function. To make sure that the R-R interval add up at the end, the fucntion adds the last peak found in the last window to the current window.

    Parameters
    ----------
    predicted: list
        List of detected ECG R-peaks.
    last_peak_found: int
        The sample number of the last peak found in the previous window.

    Returns
    --------
    list
        A list containing the new peaks.
    """
    # Find peaks with timestamp less than or equal to last peak found
    remove_previous = find_previous_peaks(predicted, last_peak_found)

    # Remove previous peaks and add the timestamp of last peak found
    if last_peak_found != 0:
        predicted = add_remove(predicted, [last_peak_found], remove_previous)
    return predicted

def workflow(ecgs, rpeaks, window_length, step_size, count, last_peak_found, edge_threshold, algorithm = "neurokit"):
    """Construct the real-time workflow.
    This function gets called in the windowing function. It contains all the steps for the real-time workflow.
    You can edit and perform all the operations that you want in the current window here.
    Here are the steps followed in the workflow below:
    1- Find peaks in the current window.
    2- Compute R-R intervals.
    3- Compute median (or mean) of the R-R intervals.
    4- Remove peaks detected in the previous window.
    6- Remove peaks beyond the threshold determined,
    5- Fix small and large gaps.
    6- Compute new R-R intervals.
    7- Compute execution time.
    8- Save last peak found in the current window so that you can add it in the next window.

    Parameters
    ----------
    ecgs : pd.DataFrame
        DataFrame containing ECG signal. If the input is a slice of a DataFrame, reset_index(drop=True) must be used.
    rpeaks : pd.DataFrame
        DataFrame containing Rpeak signal
    window_length: int
        Integer indicating the length of the window in seconds.
    Step_size: int
        Integer indicating the step size in the windowing function.
    count: int
        Integer indicating the number of windows found.
    last_peal_found: int
        The sample number of the last peak found in the previous window.
    edge_thereshold: float
        Indicates the time in seconds of the current window after which the found peaks are discarded.
    algorithm : string
        Indicates the peak detection algorithm used.


    Returns
    --------
    list
        A list containing the R-R intervals.
    int
        An integer indicating the sample number of the last peak found in the current window.
    """
    start = time.time()

    # Detect Peaks based on the algorithm chosen
    match algorithm:
        case "neurokit":
            result = find_Rpeaks(ecgs, rpeaks, method=neurokit)
        case "christov2004":
            result = find_Rpeaks(ecgs, rpeaks, method=christov2004)
        case "elgendi2010":
            result = find_Rpeaks(ecgs, rpeaks, method=elgendi2010)
        case "engzeemod2012":
            result = find_Rpeaks(ecgs, rpeaks, method=engzeemod2012)
        case "gamboa2008":
            result = find_Rpeaks(ecgs, rpeaks, method=gamboa2008)
        case "hamilton2002":
            result = find_Rpeaks(ecgs, rpeaks, method=hamilton2002)
        case "kalidas2017":
            result = find_Rpeaks(ecgs, rpeaks, method=kalidas2017)
        case "martinez2004":
            result = find_Rpeaks(ecgs, rpeaks, method=martinez2004)
        case "pantompkins1985":
            result = find_Rpeaks(ecgs, rpeaks, method=pantompkins1985)
        case "rodrigues2020":
            result = find_Rpeaks(ecgs, rpeaks, method=rodrigues2020)
        case "sleepecg":
            result = find_Rpeaks(ecgs, rpeaks, method=sleepecg)
        case "tempbeat":
            result = find_Rpeaks(ecgs, rpeaks, method=tempbeat)


    predicted = result["Predicted"][0]
    sampling_rate = result["Sampling_Rate"][0]

    # Find initial R-R intervals
    RR_intial = get_RR_intervals(predicted)

    # Compute Median or Mean
    median = get_stat_info(RR_intial, mode="median")

    # Remove previous peaks and add last peak
    predicted_new = remove_previous_peaks(predicted, last_peak_found)

    # Remove peaks beyong edge threshold
    predicted_new_trimmed = add_remove(predicted_new, [], find_peaks_beyond_edge(predicted_new, edge_threshold))

    # Fix small and large gaps : small_gap and large_gap values to be changed asneeded
    predicted_fixed = fix_gaps(predicted_new_trimmed, small_gap = 0.1 * sampling_rate, large_gap = 2 * sampling_rate, interval = median)
    #predicted_fixed = predicted_new_trimmed
    # Find new R-R intervals
    RR_new = get_RR_intervals(predicted_fixed)

    end = time.time()
    
    result["FixedPeaks"] = [predicted_fixed]
    result["R-R Intervals (initial)"] = [RR_intial]
    result["PreviousLastPeak"] = last_peak_found

    if len(predicted_fixed) == 0:
        result["NewLastPeak"] = 0
    else:
        result["NewLastPeak"] = predicted_fixed[-1]
    result["R-R Intervals (new)"] = [RR_new]
    
    execution_time = end - start
    result["ExecutionTime"] = execution_time
    return result, result["NewLastPeak"][0]
