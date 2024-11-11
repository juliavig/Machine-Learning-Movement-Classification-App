import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_acceleration_data(group_data, title):
    concatenated_data = None
    for dset_name in group_data:
        data = group_data[dset_name][()]
        if concatenated_data is None:
            concatenated_data = data
        else:
            concatenated_data = np.vstack((concatenated_data, data))

    concatenated_data = concatenated_data[concatenated_data[:, 0].argsort()]

    time_adjusted = concatenated_data[:, 0] - concatenated_data[0, 0]
    time_adjusted = time_adjusted[time_adjusted <= 400]

    time_diff = np.diff(time_adjusted)
    gap_indices = np.where(time_diff > 1)[0] + 1

    for idx in reversed(gap_indices):
        time_adjusted = np.insert(time_adjusted, idx, np.nan)
        concatenated_data = np.insert(concatenated_data, idx, np.nan, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(time_adjusted, concatenated_data[:len(time_adjusted), 1], label='AccX')
    plt.plot(time_adjusted, concatenated_data[:len(time_adjusted), 2], label='AccY')
    plt.plot(time_adjusted, concatenated_data[:len(time_adjusted), 3], label='AccZ')

    plt.xlim(0, 420)
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

with h5py.File('dataset.hdf5', 'r') as hdf5_file:
    test_group = hdf5_file['Test']

    train_group = hdf5_file['Train']

    plot_acceleration_data(test_group, 'Test Data Acceleration Over Time from HDF5 File')
    plot_acceleration_data(train_group, 'Train Data Acceleration Over Time from HDF5 File')
