import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns

# Load dataset
dataset = pd.read_csv(r"c:\Users\kumar\Downloads\raw_student_data.csv")

print(dataset.head(50))

missing=dataset.isnull().sum()
print(missing)

print("SHape:",dataset.shape)

print("\nin percentage:")
print((dataset.isnull().sum() / dataset.shape[0]) * 100)

print("")
print("after:\n")


clean=dataset.drop(columns=["Gender"],inplace=True)
dataset.dropna(inplace=True)
print(dataset.head(50))

print("Percentage_cleaning:",((55-50)/55)*100)

missing=dataset.isnull().sum()
print(missing)

print("SHape:",dataset.shape)

print("\nin percentage:")
print((dataset.isnull().sum() / dataset.shape[0]) * 100)

sns.heatmap(dataset.isnull())
plt.show()