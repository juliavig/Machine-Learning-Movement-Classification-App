import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
import numpy as np
from scipy.stats import skew, kurtosis, iqr, variation
import joblib

def extract_features(segment):
    features = {}
    for axis in ['Linear Acceleration x (m/s^2)', 'Linear Acceleration y (m/s^2)', 'Linear Acceleration z (m/s^2)']:
        axis_name = axis.split(' ')[2]
        features[f'{axis_name}_max'] = segment[axis].max()
        features[f'{axis_name}_min'] = segment[axis].min()
        features[f'{axis_name}_range'] = features[f'{axis_name}_max'] - features[f'{axis_name}_min']
        features[f'{axis_name}_mean'] = segment[axis].mean()
        features[f'{axis_name}_median'] = segment[axis].median()
        features[f'{axis_name}_var'] = segment[axis].var()
        features[f'{axis_name}_skew'] = skew(segment[axis])
        features[f'{axis_name}_kurt'] = kurtosis(segment[axis])
        features[f'{axis_name}_iqr'] = iqr(segment[axis])
        features[f'{axis_name}_cv'] = variation(segment[axis])
    return features
    pass

# Assuming the trained model is saved as 'trained_model.pkl'
# Make sure to train and save your model using joblib.dump() beforehand
pipeline = joblib.load('model.pkl')

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Walking or Jumping Classifier")

        upload_button = tk.Button(root, text="Upload CSV File", command=self.upload_csv)
        upload_button.pack()

        classify_button = tk.Button(root, text="Classify CSV File", command=self.classify)
        classify_button.pack()

        self.figure = plt.Figure(figsize=(6, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self.root)
        self.canvas.get_tk_widget().pack()

    def upload_csv(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not self.filepath:
            return
        self.data = pd.read_csv(self.filepath)
        messagebox.showinfo("File uploaded", "Successfully uploaded")

    def classify(self):
        if not hasattr(self, 'data'):
            messagebox.showerror("Error", "Please upload a CSV file first")
            return

        # Here, adapt this part to match how your data needs to be processed
        # and features extracted to be compatible with your model's expected input
        segments = self.segment_data(self.data)  # You need to implement this based on your model's requirements
        segment_features = [extract_features(segment) for segment in segments]
        features_df = pd.DataFrame(segment_features)

        predictions = pipeline.predict(features_df)

        output_filepath = self.filepath.replace('.csv', '_labeled.csv')
        result_df = pd.DataFrame(predictions, columns=['Activity'])
        result_df.to_csv(output_filepath, index=False)

        self.ax.clear()
        self.ax.plot(result_df['Activity'], label='Activity')
        self.ax.legend()
        self.canvas.draw()

        messagebox.showinfo("Classification complete", f"Labeled file saved as {output_filepath}")

    def segment_data(self, data):
        # Implement this method based on your data segmentation logic
        pass

root = tk.Tk()
app = App(root)
root.mainloop()
