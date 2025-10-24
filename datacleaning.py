import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns

# Load dataset
dataset = pd.read_csv(r"c:\Users\kumar\Downloads\raw_student_data.csv")

# Check missing values
print(dataset.isnull().sum())
print("\nIn percentage:")
print((dataset.isnull().sum() / dataset.shape[0]) * 100)

# Plot missing value count
missing = dataset.isnull().sum()
missing = missing[missing > 0]

plt.figure(figsize=(8, 5))
sns.barplot(x=missing.index, y=missing.values, palette="rocket")
plt.title("Missing Values Count")
plt.ylabel("Count")
plt.xlabel("Columns")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
