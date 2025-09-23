Real-time pipeline to extract ECG R-peaks and compute R-R intervals.
Steps to run the code:
1- Fetch the data by running fetch_database.py script. Add the appropriate output paths that indicate where to store the data.
2- Edit the workflow function in real_time.py script as needed. This function contains all the steps to process the data.
3- Run the function windowing in window.py script to launch the pipeline. This function will extract the data in windows and calls workflow function to perform the required operations.
4- Analyze the results using the validation methods in performance_metrics.py script.
5- Algorithms can be compared using the studies.py script based on previous work: https://github.com/neuropsychology/NeuroKit/tree/master/studies/ecg_benchmark
