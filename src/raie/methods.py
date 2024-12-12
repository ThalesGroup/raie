import neurokit2 as nk
from sleepecg import detect_heartbeats
from tempbeat.extraction.heartbeat_extraction import hb_extract
from tempbeat.utils.timestamp import sampling_rate_to_sig_time, timestamp_to_samp


def neurokit(ecg, sampling_rate):
    """Run NeuroKit peak detection algorithm.
    The function detects Rpeaks in ECG data using NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="neurokit")
    return info["ECG_R_Peaks"]


def pantompkins1985(ecg, sampling_rate):
    """Runs Pan-Tompkins 1985 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Pan-Tompkins (1985) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="pantompkins1985")
    return info["ECG_R_Peaks"]


def hamilton2002(ecg, sampling_rate):
    """Run Hamilton 2002 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Hamilton (2002) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="hamilton2002")
    return info["ECG_R_Peaks"]


def martinez2004(ecg, sampling_rate):
    """Run Martinez 2004 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Martinez (2004) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="martinez2004")
    return info["ECG_R_Peaks"]


def christov2004(ecg, sampling_rate):
    """Run Christov 2004 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Christov (2004) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="christov2004")
    return info["ECG_R_Peaks"]


def gamboa2008(ecg, sampling_rate):
    """Run Gamboa 2008 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Gamboa (2008) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="gamboa2008")
    return info["ECG_R_Peaks"]


def elgendi2010(ecg, sampling_rate):
    """Run Elgendi 2010 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Elgendi (2010) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="elgendi2010")
    return info["ECG_R_Peaks"]


def engzeemod2012(ecg, sampling_rate):
    """Run Engzeemod 2012 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Engzeemod (2012) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="engzeemod2012")
    return info["ECG_R_Peaks"]


def kalidas2017(ecg, sampling_rate):
    """Run Kalidas 2017 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Kalidas (2017) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="kalidas2017")
    return info["ECG_R_Peaks"]


def rodrigues2020(ecg, sampling_rate):
    """Run Rodrigues 2020 as implemented in NeuroKit2.
    The function detects Rpeaks in ECG data using Rodrigues (2020) as implemented in NeuroKit algorithm.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    signal, info = nk.ecg_peaks(ecg, sampling_rate=sampling_rate, method="rodrigues2020")
    return info["ECG_R_Peaks"]


def sleepecg(ecg, sampling_rate):
    """ Run SleepECG as peak detection algorithm.
    The function detects Rpeaks in ECG data using SleepECG.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    return detect_heartbeats(ecg, sampling_rate)


def tempbeat(ecg, sampling_rate):
    """ Run TempBeat as peak detection algorithm
    The function detects Rpeaks in ECG data using TempBeat.

    Parameters
    ----------
    ecg : pd.DataFrame
        DataFrame containing ECG signal.
    sampling_rate : int
        Integer specifying the sampling rate of the ECG signal.

    Returns
    --------
    pd.DataFrame
        A DataFrame containing information about the peaks found.

    """
    peak_time = hb_extract(ecg, sampling_rate=sampling_rate, method="temp")
    return timestamp_to_samp(peak_time, sampling_rate=sampling_rate, sig_time=sampling_rate_to_sig_time(ecg, sampling_rate=sampling_rate))
