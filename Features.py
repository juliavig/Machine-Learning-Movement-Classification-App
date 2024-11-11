import pandas as pd
from scipy.stats import skew, kurtosis, iqr, variation
from Store import all_segments
from sklearn.preprocessing import StandardScaler

def extract_features(segment):
    features = {}
    for axis in ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']:
        axis_name = axis.split(' ')[2]
        # Max
        features[f'{axis_name}_max'] = segment[axis].max()
        # Min
        features[f'{axis_name}_min'] = segment[axis].min()
        # Range
        features[f'{axis_name}_range'] = features[f'{axis_name}_max'] - features[f'{axis_name}_min']
        # Mean
        features[f'{axis_name}_mean'] = segment[axis].mean()
        # Median
        features[f'{axis_name}_median'] = segment[axis].median()
        # Variance
        features[f'{axis_name}_var'] = segment[axis].var()
        # Skewness
        features[f'{axis_name}_skew'] = skew(segment[axis])
        # Kurtosis
        features[f'{axis_name}_kurt'] = kurtosis(segment[axis])
        # Interquartile Range
        features[f'{axis_name}_iqr'] = iqr(segment[axis])
        # Coefficient of Variation
        features[f'{axis_name}_cv'] = variation(segment[axis])
    return features

all_features = []
for segment in all_segments:
    features = extract_features(segment)
    all_features.append(features)

# Create a DataFrame with all features
feature_df = pd.DataFrame(all_features)

print(feature_df.head())

# NORMAILZING IT
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the feature DataFrame and transform the data
normalized_features = scaler.fit_transform(feature_df)

normalized_feature_df = pd.DataFrame(normalized_features, columns=feature_df.columns)
print(normalized_feature_df.head())
