import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)

# File paths and names
file_paths = {
     " yolov5": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\contrast\yolov5\results.csv",
    " yolov6": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\contrast\yolov6\results.csv",
    " yolov8n": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\contrast\yolov8n\results.csv",
    " yolov9t": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\contrast\yolov9t\results.csv",
    " yolov10n": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\contrast\yolov10n\results.csv",
    " yolov11": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\1_base\results.csv",
    " Ours": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\5_regnet_imse_impiou\results.csv",
}

# Load and clean data
results = {}
for name, path in file_paths.items():
    df = pd.read_csv(path)
    clean_column_names(df)
    results[name] = df

# Interpolation function for smoother curves
def smooth_curve(x, y, num_points=200):
    f = interp1d(x, y, kind='cubic')
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_smooth = f(x_new)
    return x_new, y_smooth

# Plot smoothed recall curves for all models
plt.figure()

# Iterate through results and plot
for name, df in results.items():
    if 'metrics/recall(B)' not in df.columns:
        print(f"Skipping {name}: 'metrics/recall(B)' column not found.")
        continue

    epochs = np.arange(min(len(df['metrics/recall(B)']), 200))
    x_smooth, y_smooth = smooth_curve(epochs, df['metrics/recall(B)'][:len(epochs)])
    plt.plot(x_smooth, y_smooth, label=name)

# Plot settings
plt.xlabel("Epoch")
plt.ylabel("Recall (B)")
plt.xlim(0, 200)
plt.legend()

plt.grid(True)
plt.savefig("smoothed_recall_comparison_models.png")

# Show plot
plt.show()
