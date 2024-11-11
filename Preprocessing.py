
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def moving_average_filter(data, window_size=5):
    df = pd.DataFrame(data)
    smoothed_data = df.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    return smoothed_data


def normalize_data(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def preprocess_group(group):
    preprocessed_data = None
    for dset_name in group:
        data = group[dset_name][()]

        # Assuming acceleration data starts from column 1 (0 is time)
        acc_data = data[:, 1:]

        # Apply moving average filter with rolling
        smoothed_data = moving_average_filter(acc_data)

        # Normalize data (excluding time column)
        normalized_data, _ = normalize_data(smoothed_data)

        # Combine time column back with normalized data
        preprocessed_segment = np.column_stack((data[:, 0], normalized_data))

        # Concatenate all segments
        if preprocessed_data is None:
            preprocessed_data = preprocessed_segment
        else:
            preprocessed_data = np.vstack((preprocessed_data, preprocessed_segment))

    # Sort the concatenated data by time, in case it's not in order
    preprocessed_data = preprocessed_data[preprocessed_data[:, 0].argsort()]

    return preprocessed_data


def visualize_preprocessed_data(preprocessed_data, name):
    time = preprocessed_data[:, 0]
    accX = preprocessed_data[:, 1]
    accY = preprocessed_data[:, 2]
    accZ = preprocessed_data[:, 3]

    plt.figure(figsize=(10, 6))
    plt.plot(time, accX, label='AccX (Normalized)')
    plt.plot(time, accY, label='AccY (Normalized)')
    plt.plot(time, accZ, label='AccZ (Normalized)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Acceleration')
    plt.title(name +' Preprocessed Acceleration Data Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()


def main(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        # Choose whether to visualize training or test data
        group_name = 'Test'  # Change to 'Train' if you want to visualize training data
        data_group = hdf5_file[group_name]
        preprocessed_data = preprocess_group(data_group)

        # Choose whether to visualize training or test data
        group_name2 = 'Train'  # Change from 'Test' to 'Train' to visualize training data
        data_group2 = hdf5_file[group_name2]
        preprocessed_data2 = preprocess_group(data_group2)

        # Visualize the concatenated preprocessed data
        if preprocessed_data.size > 0:
            visualize_preprocessed_data(preprocessed_data, group_name)
            visualize_preprocessed_data(preprocessed_data2,group_name2)
        else:
            print(f"No data to visualize in the {group_name} group.")

if __name__ == "__main__":
    hdf5_file_path = 'dataset.hdf5'
    main(hdf5_file_path)
