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
    " YOLOv11(Baseline)": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\1_base\results.csv",
    " YOLOv11+Regnety": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\2_regnety\results.csv",
    " YOLOv11+SDSE": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\3_imse\results.csv",
    " YOLOv11+ImPIoU": r"C:\Users\黄文全\Desktop\ultralytics-main-regnet-crack\ultralytics-main-regnet-crack\runs\weights\4_impiou\results.csv",
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

# Plot smoothed mAP@0.5 curves for all models
plt.figure()

# Iterate through results and plot
for name, df in results.items():
    if 'metrics/mAP50-95(B)' not in df.columns:
        print(f"Skipping {name}: 'metrics/mAP50-95(B)' column not found.")
        continue

    epochs = np.arange(min(len(df['metrics/mAP50-95(B)']), 200))
    x_smooth, y_smooth = smooth_curve(epochs, df['metrics/mAP50-95(B)'][:len(epochs)])
    plt.plot(x_smooth, y_smooth, label=name)

# Plot settings
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5:0.95")
plt.xlim(0, 200)
plt.legend()

plt.grid(True)
plt.savefig("smoothed_mAP50-95_comparison_models.png")

# Show plot
plt.show()
