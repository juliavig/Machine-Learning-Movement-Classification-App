import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Function to calculate sample rate
def calculate_sample_rate(df):
    return int(round(1 / (df['Time (s)'][1] - df['Time (s)'][0])))

# Function to encode labels to integers
def fit_encode_labels(dfs, label_columns):
    label_encoders = {}
    all_data = pd.concat(dfs, axis=0)
    for col in label_columns:
        le = LabelEncoder()
        le.fit(all_data[col])
        label_encoders[col] = le
        for df in dfs:
            df[col] = le.transform(df[col])
    return dfs, label_encoders

# Function to segment the data into windows
def segment_data(df, window_size, sample_rate):
    rows_per_segment = sample_rate * window_size
    return [df.iloc[i:i + rows_per_segment] for i in range(0, len(df), rows_per_segment) if i + rows_per_segment <= len(df)]

# Load the datasets
file_paths = {
    'Aytu_Walking_Hand': 'Aytu_Walking_Hand.csv',
    'Aytu_Jumping_Hand': 'Aytu_Jumping_Hand.csv',
    'Connor_Walking_BackPocket': 'Connor_Walking_BackPocket.csv',
    'Connor_Jumping_BackPocket': 'Connor_Jumping_BackPocket.csv',
    'Julia_Walking_FrontPocket': 'Julia_Walking_FrontPocket.csv',
    'Julia_Jumping_FrontPocket': 'Julia_Jumping_FrontPocket.csv'
}

dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Columns that need label encoding (object type columns)
label_columns = ['Activity', 'Position']

# Fit LabelEncoders on all possible labels and transform the datasets
dfs_list, label_encoders = fit_encode_labels([df for df in dfs.values()], label_columns)

# Calculate the sample rate (assuming all datasets have the same sample rate)
sample_rate = calculate_sample_rate(next(iter(dfs.values())))

# Segment the datasets
all_segments = []
for df in dfs.values():
    all_segments += segment_data(df, window_size=5, sample_rate=sample_rate)

# Shuffle the combined segments
np.random.shuffle(all_segments)

# Split the combined segments into training and testing sets (90% training, 10% testing)
train_segments, test_segments = train_test_split(all_segments, test_size=0.1)

# Create an HDF5 file with the dataset organized as per the provided structure
with h5py.File('dataset.hdf5', 'w') as hdf5_file:
    # Create groups for training and testing
    train_group = hdf5_file.create_group('Train')
    test_group = hdf5_file.create_group('Test')

    # Save segmented data into training and testing groups
    for i, segment in enumerate(train_segments):
        train_group.create_dataset(f'train_{i}', data=segment.values)

    for i, segment in enumerate(test_segments):
        test_group.create_dataset(f'test_{i}', data=segment.values)

    # Add datasets for each activity, position, and member
    for name, df in dfs.items():
        member_name = name.split('_')[0]  # Assuming the format "Name_Activity_Position"
        if member_name not in hdf5_file:
            member_group = hdf5_file.create_group(member_name)
        else:
            member_group = hdf5_file[member_name]
        for (activity, position), group_df in df.groupby(['Activity', 'Position']):
            dataset_name = f'{label_encoders["Activity"].inverse_transform([activity])[0]}_' \
                           f'{label_encoders["Position"].inverse_transform([position])[0]}'
            member_group.create_dataset(dataset_name, data=group_df.values)
