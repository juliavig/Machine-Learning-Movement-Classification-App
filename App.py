import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import joblib
from scipy.stats import skew, kurtosis, iqr, variation

class ActivityClassifierApp(tk.Tk):
    def __init__(self, model_path, label_encoder_path):
        super().__init__()
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.figure_canvas = None
        self.model = joblib.load(model_path)
        self.data = None
        self.initialize_ui()

    def initialize_ui(self):
        self.title("Activity Classifier")
        self.geometry("600x400")

        tk.Button(self, text="Select CSV File", command=self.load_csv).pack()
        tk.Button(self, text="Classify", command=self.classify_activities).pack()
        self.figure_canvas = None

    def load_csv(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = pd.read_csv(file_path)
            print("CSV Loaded")

    def classify_activities(self):
        if self.data is None:
            messagebox.showerror("Error", "Please load a CSV file first.")
            return

        self.segments = self.segment_data(self.data)
        features_list = [self.extract_features(segment) for segment in self.segments]
        features_df = pd.DataFrame(features_list)

        expected_feature_order = ['Linear Acceleration x (m/s^2)_max', 'Linear Acceleration x (m/s^2)_min',
                                  'Linear Acceleration x (m/s^2)_range', 'Linear Acceleration x (m/s^2)_mean',
                                  'Linear Acceleration x (m/s^2)__median', 'Linear Acceleration x (m/s^2)__var',
                                  'Linear Acceleration x (m/s^2)__skew', 'Linear Acceleration x (m/s^2)__kurt',
                                  'Linear Acceleration x (m/s^2)__iqr', 'Linear Acceleration x (m/s^2)__cv',
                                  'Linear Acceleration y (m/s^2)_max', 'Linear Acceleration y (m/s^2)_min',
                                  'Linear Acceleration y (m/s^2)_range', 'Linear Acceleration y (m/s^2)_mean',
                                  'Linear Acceleration y (m/s^2)__median', 'Linear Acceleration y (m/s^2)__var',
                                  'Linear Acceleration y (m/s^2)__skew', 'Linear Acceleration y (m/s^2)__kurt',
                                  'Linear Acceleration y (m/s^2)__iqr', 'Linear Acceleration y (m/s^2)__cv',
                                  'Linear Acceleration z (m/s^2)_max', 'Linear Acceleration z (m/s^2)_min',
                                  'Linear Acceleration z (m/s^2)_range', 'Linear Acceleration z (m/s^2)_mean',
                                  'Linear Acceleration z (m/s^2)__median', 'Linear Acceleration z (m/s^2)__var',
                                  'Linear Acceleration z (m/s^2)__skew', 'Linear Acceleration z (m/s^2)__kurt',
                                  'Linear Acceleration z (m/s^2)__iqr', 'Linear Acceleration z (m/s^2)__cv']

        if not set(expected_feature_order).issubset(features_df.columns):
            missing_cols = set(expected_feature_order) - set(features_df.columns)
            messagebox.showerror("Error", f"Missing expected features: {missing_cols}")
            return

        features = features_df[expected_feature_order].values
        numeric_predictions = self.model.predict(features)
        decoded_predictions = ["jumping" if pred == 0 else "walking" for pred in numeric_predictions]

        self.save_predictions_to_csv(decoded_predictions)
        self.plot_results(decoded_predictions)

    def save_predictions_to_csv(self, labeled_predictions):
        output_file_name = 'activity_predictions.csv'
        segment_start_times = [segment.iloc[0]['Time (s)'] for segment in self.segments]

        results_df = pd.DataFrame({
            'Start Time (s)': segment_start_times,
            'Predicted Activity': labeled_predictions
        })

        results_df.sort_values(by='Start Time (s)', inplace=True)
        results_df.to_csv(output_file_name, index=False)

        messagebox.showinfo("Success", f"Predictions saved to {output_file_name}")

    def segment_data(self, data, window_size=5):
        timestamps = data['Time (s)']
        sample_rate = round(1 / (timestamps.iloc[1] - timestamps.iloc[0]))
        rows_per_segment = sample_rate * window_size
        self.segments = []
        for start in range(0, len(data), rows_per_segment):
            end = start + rows_per_segment
            if end <= len(data):
                segment = data.iloc[start:end]
                self.segments.append(segment)
        return self.segments

    def extract_features(self, segment):
        features = {}
        for axis in ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']:
            prefix = f'{axis}_'  # Prefix for naming the features: x_, y_, z_
            features[f'{prefix}max'] = segment[f'{axis}'].max()
            features[f'{prefix}min'] = segment[f'{axis}'].min()
            features[f'{prefix}range'] = features[f'{prefix}max'] - features[f'{prefix}min']
            features[f'{prefix}mean'] = segment[f'{axis}'].mean()
            features[f'{prefix}_median'] = segment[axis].median()
            features[f'{prefix}_var'] = segment[axis].var()
            features[f'{prefix}_skew'] = skew(segment[axis], nan_policy='omit')  # Handling NaNs
            features[f'{prefix}_kurt'] = kurtosis(segment[axis], nan_policy='omit')  # Handling NaNs
            features[f'{prefix}_iqr'] = iqr(segment[axis])
            features[f'{prefix}_cv'] = variation(segment[axis])
        return features

    def map_predictions_to_labels(self, predictions):
        label_mapping = {0: "walking", 1: "jumping"}
        labeled_predictions = [label_mapping[pred] for pred in predictions]
        return labeled_predictions

    def plot_results(self, predictions):
        fig = Figure(figsize=(6, 4), dpi=100)
        plot = fig.add_subplot(1, 1, 1)
        plot.plot(predictions, 'ro-')
        if self.figure_canvas:
            self.figure_canvas.get_tk_widget().destroy()

        self.figure_canvas = FigureCanvasTkAgg(fig, master=self)
        self.figure_canvas.draw()
        self.figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    app = ActivityClassifierApp('/Users/juliaviger/PycharmProjects/ELEC292_FinalProject/trainedModel.joblib',
                                '/Users/juliaviger/PycharmProjects/ELEC292_FinalProject/label_encoder.joblib')
    app.mainloop()
