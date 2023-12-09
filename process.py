import pandas as pd
import numpy as np
import json
import warnings

from utils.preprocess import filter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings("ignore")


def load_data(data_path: str, type: str) -> None:
    df = pd.read_csv(data_path)

    # load time_series json
    data_name = data_path.split(".")[0].split("/")[-1]
    json_path = f"json/{data_name}.json"

    # load data json file to divide different dataframe
    with open(json_path, "r") as file:
        data_dict = json.load(file)

    # sensor list
    sensor_list = df["node_address"].unique()

    # dataframe rename and reserve columns which we need
    df = df.rename(
        columns={
            "tickstotimestamp": "timestamp",
            "accelerometerx": "acc_x",
            "accelerometery": "acc_y",
            "accelerometerz": "acc_z",
            "gyroscopex": "gyro_x",
            "gyroscopey": "gyro_y",
            "gyroscopez": "gyro_z",
            "標記轉換": "label",
        }
    )

    df = df[
        [
            "node_address",
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "label",
        ]
    ]

    df.dropna(inplace=True)

    print(df)

    # make sliding window for 12 second
    window_size = 60
    features = []
    labels = []

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["label"] = df["label"].astype(int)

    for sensor in sensor_list:
        sensor_df = df.loc[df["node_address"] == sensor]

        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
        sensor_df.sort_values("timestamp", ascending=True, inplace=True)

        time_series_list = data_dict[sensor]
        for time_series in time_series_list:
            start_time = time_series[0]
            end_time = time_series[1]

            sub_df = sensor_df.loc[
                (sensor_df["timestamp"] >= start_time)
                & (sensor_df["timestamp"] <= end_time)
            ]

            data_set = sub_df.drop(["node_address", "timestamp"], axis=1)
            y_data = data_set["label"]
            x_data = data_set.drop(["label"], axis=1)

            # normalize input data
            scaler = MinMaxScaler()
            x_data = scaler.fit_transform(x_data)

            columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            x_data = pd.DataFrame(x_data, columns=columns)

            # record input data amount
            amount = len(sub_df) - window_size + 1

            for idx in range(amount):
                sensor_data = x_data.iloc[idx : idx + window_size]

                feature_vector = (
                    sensor_data[
                        ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
                    ]
                    .values.flatten()
                    .tolist()
                )

                label_values = y_data.iloc[idx : idx + window_size].values[-1]

                features.append(feature_vector)
                labels.append(label_values)

    # convert to numpy format
    features = np.array(features)
    labels = np.array(labels)

    if type == "train":
        # split dataset into train / test data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, shuffle=False
        )

        # save data into numpy type
        np.savez("data/train_data_no_shuffle.npz", x_train=X_train, y_train=y_train)
        np.savez("data/test_data_no_shuffle.npz", x_test=X_test, y_test=y_test)

    elif type == "valid":
        np.savez(f"data/{data_name}.npz", x_valid=features, y_valid=labels)


if __name__ == "__main__":
    data_path = "dataset/train_dataset.csv"
    load_data(data_path, "train")
