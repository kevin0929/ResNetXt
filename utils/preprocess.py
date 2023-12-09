import pandas as pd
import numpy as np

from scipy.signal import butter, lfilter


def butter_bandpass_filter(
    data: pd.Series, lowcut: float, highcut: float, fs: float, order: int = 5
) -> list:
    """
    lowcut: lowerbound frequency of filter
    highcut: upperbound frequency of filter
    fs: sample frequency of filter
    order: gradient
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, data)

    return y


def filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    filter raw data to conclude odd value
    """
    df["angle"] = np.arctan(df["acc_x"] / df["acc_y"])
    df["acc_vec"] = (df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2) ** 0.5
    df["gyro_vec"] = (df["gyro_x"] ** 2 + df["gyro_y"] ** 2 + df["gyro_z"] ** 2) ** 0.5

    # set range for bandpass filter
    lowcut = 0.3
    highcut = 2.4
    sampling_fs = 5.0

    # filter signal
    axises = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
    for axis in axises:
        input_signal = df[axis]
        filtered_signal = butter_bandpass_filter(
            input_signal, lowcut, highcut, sampling_fs, order=6
        )
        bd_value = np.array(filtered_signal)
        df[axis] = bd_value

    return df
