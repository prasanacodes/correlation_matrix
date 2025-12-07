import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load the data
file_path = 'q-excel-correlation-heatmap.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
    exit(1)

# Select relevant columns for correlation
numerical_cols = [
    'Supplier_Lead_Time',
    'Inventory_Levels',
    'Order_Frequency',
    'Delivery_Performance',
    'Cost_Per_Unit'
]

# Ensure cols exist
missing_cols = [col for col in numerical_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing columns in dataset: {missing_cols}")
    exit(1)

df_corr = df[numerical_cols]

# Calculate correlation matrix
correlation_matrix = df_corr.corr()

# Export correlation matrix to CSV
correlation_matrix.to_csv('correlation.csv')
print("correlation.csv saved.")

# Generate Heatmap
plt.figure(figsize=(5, 5))
# Target is 400x400 to 512x512.
# 5 inches * 100 dpi = 500x500 pixels.

# Define custom colormap: Red (low) - White - Green (high)
# Seaborn's 'RdYlGn' is close, but has Yellow. User asked for Red-White-Green.
# We can use LinearSegmentedColormap
from matplotlib.colors import LinearSegmentedColormap

colors = ["red", "white", "green"]
cmap = LinearSegmentedColormap.from_list("custom_rwg", colors, N=256)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    cbar_kws={"shrink": .8}
)

plt.title('Supply Chain Correlation Matrix')
plt.tight_layout()

# Save as png
plt.savefig('heatmap.png', dpi=100, bbox_inches='tight')
print("heatmap.png saved.")

# Check dimensions
import cv2
img = cv2.imread('heatmap.png')
if img is not None:
    h, w, _ = img.shape
    print(f"Heatmap dimensions: {w}x{h}")
else:
    print("Could not verify heatmap dimensions with cv2.")
