import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from Features import extract_features
from Store import all_segments
import joblib


all_features = []
labels = []

for segment in all_segments:
    features = extract_features(segment)
    all_features.append(features)
    activity_label = segment['Activity'].iloc[0]  # Adjust this as per your data structure
    labels.append(activity_label)

feature_df = pd.DataFrame(all_features)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

X = feature_df.values
y = y_encoded

# Create a pipeline that includes scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

# Define StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')

print(f'CV Accuracy Scores: {cv_scores}')
print(f'CV Accuracy: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}')

# Train-test split for final model evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Fit the model to the training set
pipeline.fit(X_train, y_train)

# Predict and evaluate on the test set
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Set Accuracy: {accuracy:.2f}')

# Graphical Analysis of CV Scores
plt.figure(figsize=(10, 6))
plt.plot(cv_scores, label='CV Accuracy Scores', marker='o')
plt.axhline(y=np.mean(cv_scores), color='r', linestyle='--', label='Mean CV Accuracy')
plt.fill_between(range(len(cv_scores)), np.mean(cv_scores) - np.std(cv_scores), np.mean(cv_scores) + np.std(cv_scores), color='r', alpha=0.2)
plt.title('Cross-Validation Accuracy Scores')
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained pipeline to a file
joblib.dump(pipeline, '/Users/juliaviger/PycharmProjects/ELEC292_FinalProject/trainedModel.joblib')

# Save the LabelEncoder for later use in predictions
joblib.dump(label_encoder, '/Users/juliaviger/PycharmProjects/ELEC292_FinalProject/label_encoder.joblib')
