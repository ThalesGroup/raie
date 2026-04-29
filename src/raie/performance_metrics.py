import numpy as np
from neurokit2.signal import signal_period


def true_positive(predicted, actual, threshold=None, sampling_rate=None):
    if threshold is None:
        return len(set(predicted) & set(actual))
    else:
        TP = 0
        samples = int(sampling_rate * threshold)
        actual = list(actual)
        for i in predicted:
            for j in range(0, samples + 1):
                if (i + j) in actual:
                    TP = TP + 1
                    actual.remove(i + j)
                    break
                elif (i - j) in actual:
                    TP = TP + 1
                    actual.remove(i - j)
                    break
        return TP


def false_negative(predicted, actual, threshold=None, sampling_rate=None):
    return len(actual) - true_positive(predicted, actual, threshold, sampling_rate)


def false_positive(predicted, actual, threshold=None, sampling_rate=None):
    return len(predicted) - true_positive(predicted, actual, threshold, sampling_rate)


def true_negative(predicted, actual, total, threshold=None, sampling_rate=None):
    return (
        total
        - len(predicted)
        - false_negative(predicted, actual, threshold, sampling_rate)
    )


def sensitivity(predicted, actual, threshold=None, sampling_rate=None):
    TP = true_positive(predicted, actual, threshold, sampling_rate)
    FN = false_negative(predicted, actual, threshold, sampling_rate)

    if (TP + FN) != 0:
        return TP / (TP + FN)
    else:
        return 0


def positive_predictive_value(predicted, actual, threshold=None, sampling_rate=None):
    TP = true_positive(predicted, actual, threshold, sampling_rate)
    FP = false_positive(predicted, actual, threshold, sampling_rate)

    if (TP + FP) != 0:
        return TP / (TP + FP)
    else:
        return 0


def accuracy(predicted, actual, total, threshold=None, sampling_rate=None):
    TP = true_positive(predicted, actual, threshold, sampling_rate)
    TN = true_negative(predicted, actual, total, threshold, sampling_rate)

    if total == 0:
        return 0
    else:
        return (TP + TN) / total


def benchmark_ecg_compareRpeaks(true_rpeaks, found_rpeaks, sampling_rate=250):
    """**Calculates error**
    The function calculatres the error from the ECG peaks.

    Parameters
    ----------
    true_peaks : list
        List of annotated peaks.
    found_peaks : list
        List of detected peaks.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    float
        A float indicating the computed error.

    """
    # Failure to find sufficient R-peaks
    if len(found_rpeaks) <= 3:
        return np.nan, "R-peaks detected <= 3"

    length = np.max(np.concatenate([true_rpeaks, found_rpeaks]))

    true_interpolated = signal_period(
        true_rpeaks,
        sampling_rate=sampling_rate,
        desired_length=length,
        interpolation_method="linear",
    )
    found_interpolated = signal_period(
        found_rpeaks,
        sampling_rate=sampling_rate,
        desired_length=length,
        interpolation_method="linear",
    )

    return np.mean(np.abs(found_interpolated - true_interpolated)), "None"
