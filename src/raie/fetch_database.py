import os
import pathlib

import ecg_gudb_database
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb


def download_and_format_gudb(output_path="./"):
    # -*- coding: utf-8 -*-
    """Script for downloading, formatting and saving the GUDB database (https://github.com/berndporr/ECG-GUDB).

    It contains ECGs from 25 subjects. Each subject was recorded performing 5 different tasks for two minutes:
    - sitting
    - a maths test on a tablet
    - walking on a treadmill
    - running on a treadmill
    - using a hand bike

    The sampling rate is 250Hz for all experiments.

    Credits and citation:
    - Howell, L., & Porr, B. (2018). High precision ECG Database with annotated R peaks,
    recorded and filmed under realistic conditions.
    """

    dfs_ecg = []
    dfs_rpeaks = []

    for participant in range(25):
        print("Participant: " + str(participant + 1) + "/25")
        for i, experiment in enumerate(ecg_gudb_database.GUDb.experiments):
            print("  - Condition " + str(i + 1) + "/5")
            # creating class which loads the experiment
            ecg_class = ecg_gudb_database.GUDb(participant, experiment)

            # Chest Strap Data - only donwload if R-peaks annotations are available
            if ecg_class.anno_cs_exists:

                data = pd.DataFrame({"ECG": ecg_class.cs_V2_V1})
                data["Participant"] = "GUDB_%.2i" % (participant)
                data["Sample"] = range(len(data))
                data["Sampling_Rate"] = 250
                data["Database"] = "GUDB_" + experiment

                # getting annotations
                anno = pd.DataFrame({"Rpeaks": ecg_class.anno_cs})
                anno["Participant"] = "GUDB_%.2i" % (participant)
                anno["Sampling_Rate"] = 250
                anno["Database"] = "GUDB_" + experiment

                # Store with the rest
                dfs_ecg.append(data)
                dfs_rpeaks.append(anno)

            # Einthoven leads
    #        if ecg_class.anno_cables_exists:
    #            cables_anno = ecg_class.anno_cables
    #            einthoven_i = ecg_class.einthoven_I
    #            einthoven_ii = ecg_class.einthoven_II
    #            einthoven_iii = ecg_class.einthoven_III

    # Save
    df_ecg = pd.concat(dfs_ecg).to_csv(output_path + "ECGs.csv", index=False)
    dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv(output_path + "Rpeaks.csv", index=False)


def download_and_format_mit_arrhythmia(database_path="./", output_path="./"):
    # -*- coding: utf-8 -*-
    """Script for formatting the MIT-Arrhythmia database

    Steps:
        1. Download the ZIP database from https://alpha.physionet.org/content/mitdb/1.0.0/
        2. Open it with a zip-opener (WinZip, 7zip).
        3. Extract the folder of the same name (named 'mit-bih-arrhythmia-database-1.0.0') to the same folder as this script.
        4. Run this script.

    Credits:
        https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
    """

    database_path = database_path

    # Check if expected folder exists
    if not os.path.exists(database_path):
        url = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
        download_successful = nk.download_zip(url, database_path)
        if not download_successful:
            raise ValueError(
                "NeuroKit error: download of MIT-Arrhythmia database failed. "
                "Please download it manually from https://alpha.physionet.org/content/mitdb/1.0.0/ "
                "and unzip it in the same folder as this script."
            )

    data_files = [database_path + file for file in os.listdir(database_path) if ".dat" in file]

    dfs_ecg = []
    dfs_rpeaks = []

    for participant, file in enumerate(data_files):

        print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

        data, anno = read_file(file, participant)

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

        # Store additional recording if available
        if "x_" + file.replace(database_path, "") in os.listdir(database_path + "x_mitdb/"):
            print("  - Additional recording detected.")
            data, anno = read_file(database_path + "/x_mitdb/" + "x_" + file.replace(database_path, ""), participant)
            # Store with the rest
            dfs_ecg.append(data)
            dfs_rpeaks.append(anno)

    # Save
    df_ecg = pd.concat(dfs_ecg).to_csv(output_path + "ECGs.csv", index=False)
    dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv(output_path + "Rpeaks.csv", index=False)

    # Quick test
    # import neurokit2 as nk
    # nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1002])


def read_file(file, participant):
    """Utility function"""
    # Get signal
    data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 0]})
    data["Participant"] = "MIT-Arrhythmia_%.2i" % (participant)
    data["Sample"] = range(len(data))
    data["Sampling_Rate"] = 360
    data["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    # getting annotations
    anno = wfdb.rdann(file[:-4], "atr")
    anno = np.unique(
        anno.sample[np.in1d(anno.symbol, ["N", "L", "R", "B", "A", "a", "J", "S", "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q", "?"])]
    )
    anno = pd.DataFrame({"Rpeaks": anno})
    anno["Participant"] = "MIT-Arrhythmia_%.2i" % (participant)
    anno["Sampling_Rate"] = 360
    anno["Database"] = "MIT-Arrhythmia-x" if "x_mitdb" in file else "MIT-Arrhythmia"

    return data, anno


def download_and_format_mit_normal(output_path="./"):
    # -*- coding: utf-8 -*-
    """Script for formatting the MIT-Normal Sinus Rhythm Database

    Steps:
        1. Download the ZIP database from https://physionet.org/content/nsrdb/1.0.0/
        2. Open it with a zip-opener (WinZip, 7zip).
        3. Extract the folder of the same name (named 'mit-bih-normal-sinus-rhythm-database-1.0.0') to the same folder as this script.
        4. Run this script.

    Credits:
        https://github.com/berndporr/py-ecg-detectors/blob/master/tester_MITDB.py by Bernd Porr
    """

    os.listdir("./")
    data_files = [
        "./data/mit-bih-normal-sinus-rhythm-database-1.0.0/" + file
        for file in os.listdir("./data/mit-bih-normal-sinus-rhythm-database-1.0.0/")
        if ".dat" in file
    ]

    dfs_ecg = []
    dfs_rpeaks = []

    for participant, file in enumerate(data_files):

        print("Participant: " + str(participant + 1) + "/" + str(len(data_files)))

        # Get signal
        data = pd.DataFrame({"ECG": wfdb.rdsamp(file[:-4])[0][:, 1]})
        data["Participant"] = "MIT-Normal_%.2i" % (participant)
        data["Sample"] = range(len(data))
        data["Sampling_Rate"] = 128
        data["Database"] = "MIT-Normal"

        # getting annotations
        anno = wfdb.rdann(file[:-4], "atr")
        anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
        anno = pd.DataFrame({"Rpeaks": anno})
        anno["Participant"] = "MIT-Normal_%.2i" % (participant)
        anno["Sampling_Rate"] = 128
        anno["Database"] = "MIT-Normal"

        # Select only 1h of recording (otherwise it's too big)
        data = data[460800 : 460800 * 3].reset_index(drop=True)
        anno = anno[(anno["Rpeaks"] > 460800) & (anno["Rpeaks"] <= 460800 * 2)].reset_index(drop=True)
        anno["Rpeaks"] = anno["Rpeaks"] - 460800

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

    # Save
    df_ecg = pd.concat(dfs_ecg).to_csv(output_path + "ECGs.csv", index=False)
    dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv(output_path + "Rpeaks.csv", index=False)

    # Quick test
    # import neurokit2 as nk
    # nk.events_plot(anno["Rpeaks"][anno["Rpeaks"] <= 1000], data["ECG"][0:1001])


def download_and_format_ludb(database_path="./", output_path="./"):
    # -*- coding: utf-8 -*-
    """Script for formatting the Lobachevsky University Electrocardiography Database

    The database consists of 200 10-second 12-lead ECG signal records representing different morphologies of the ECG signal. The ECGs were collected from healthy volunteers and patients, which had various cardiovascular diseases. The boundaries of P, T waves and QRS complexes were manually annotated by cardiologists for all 200 records.

    Steps:
        1. Download zipped data base from https://physionet.org/content/ludb/1.0.1/
        2. Unzip the folder so that you have a `lobachevsky-university-electrocardiography-database-1.0.1/` folder'
        3. Run this script.
    """

    dfs_ecg = []
    dfs_rpeaks = []

    for participant in range(200):
        filename = str(participant + 1)

        data, info = wfdb.rdsamp(database_path + "lobachevsky-university-electrocardiography-database-1.0.1/data/" + filename)

        # Get signal
        data = pd.DataFrame(data, columns=info["sig_name"])
        data = data[["i"]].rename(columns={"i": "ECG"})
        data["Participant"] = "LUDB_%.2i" % (participant + 1)
        data["Sample"] = range(len(data))
        data["Sampling_Rate"] = info["fs"]
        data["Database"] = "LUDB"

        # Get annotations
        anno = wfdb.rdann(database_path + "lobachevsky-university-electrocardiography-database-1.0.1/data/" + filename, "i")
        anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
        anno = pd.DataFrame({"Rpeaks": anno})
        anno["Participant"] = "LUDB_%.2i" % (participant + 1)
        anno["Sampling_Rate"] = info["fs"]
        anno["Database"] = "LUDB"

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

    # Save
    pd.concat(dfs_ecg).to_csv(output_path + "ECGs.csv", index=False)
    dfs_rpeaks = pd.concat(dfs_rpeaks).to_csv(output_path + "Rpeaks.csv", index=False)


def download_and_format_fantasia(database_path="./", output_path="./"):
    # -*- coding: utf-8 -*-
    """Script for formatting the Fantasia Database

    The database consists of twenty young and twenty elderly healthy subjects.
    All subjects remained in a resting state in sinus rhythm while watching the movie Fantasia (Disney, 1940) to help maintain wakefulness.
    The continuous ECG signals were digitized at 250 Hz.
    Each heartbeat was annotated using an automated arrhythmia detection algorithm, and each beat annotation was verified by visual inspection.

    Steps:
        1. Download the ZIP database from https://physionet.org/content/fantasia/1.0.0/
        2. Open it with a zip-opener (WinZip, 7zip).
        3. Extract the folder of the same name (named 'fantasia-database-1.0.0') to the same folder as this script.
        4. Run this script.
    """

    database_path = database_path + "fantasia-database-1.0.0/"

    # Check if expected folder exists
    if not os.path.exists(database_path):
        url = "https://physionet.org/static/published-projects/fantasia/fantasia-database-1.0.0.zip"
        download_successful = nk.download_zip(url, database_path)
        if not download_successful:
            raise ValueError(
                "NeuroKit error: download of Fantasia database failed. "
                "Please download it manually from https://physionet.org/content/fantasia/1.0.0/ "
                "and unzip it in the same folder as this script."
            )

    files = os.listdir(database_path)
    files = [s.replace(".dat", "") for s in files if ".dat" in s]

    dfs_ecg = []
    dfs_rpeaks = []

    for i, participant in enumerate(files):

        data, info = wfdb.rdsamp(str(pathlib.Path(database_path, participant)))

        # Get signal
        data = pd.DataFrame(data, columns=info["sig_name"])
        data = data[["ECG"]]
        data["Participant"] = "Fantasia_" + participant
        data["Sample"] = range(len(data))
        data["Sampling_Rate"] = info["fs"]
        data["Database"] = "Fantasia"

        # Get annotations
        anno = wfdb.rdann(str(pathlib.Path(database_path, participant)), "ecg")
        anno = anno.sample[np.where(np.array(anno.symbol) == "N")[0]]
        anno = pd.DataFrame({"Rpeaks": anno})
        anno["Participant"] = "Fantasia_" + participant
        anno["Sampling_Rate"] = info["fs"]
        anno["Database"] = "Fantasia"

        # Store with the rest
        dfs_ecg.append(data)
        dfs_rpeaks.append(anno)

    # Save
    pd.concat(dfs_ecg).to_csv(output_path + "ECGs.csv", index=False)
    pd.concat(dfs_rpeaks).to_csv(output_path + "Rpeaks.csv", index=False)

download_and_format_mit_normal()