import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
dataset = pd.read_csv("king.txt")
#print("🔹 First 50 rows:\n", dataset.head(50))

missing=dataset.isna().sum()
print(missing)
print(dataset.describe())

dataset["Income"]=dataset["Income"].fillna(dataset["Income"].mean())
dataset["Age"]=dataset["Age"].fillna(dataset["Age"].mean())
missing=dataset.isna().sum()
print(missing)
print(dataset.describe())

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.title("Before Standardization Scaling-Age")
sns.histplot(dataset["Age"],kde=True)

plt.subplot(1, 2, 2)
plt.title("Before Standardization Scaling-Income")
sns.histplot(dataset["Income"],kde=True)

# scaling
from sklearn.preprocessing import StandardScaler
la=StandardScaler()

ar=la.fit_transform(dataset[["Income"]])
dataset[["Income"]]=pd.DataFrame(ar,columns=["Income"])
print(dataset.head(50))

from sklearn.preprocessing import StandardScaler
la1=StandardScaler()
ar1=la1.fit_transform(dataset[["Age"]])
dataset[["Age"]]=pd.DataFrame(ar1,columns=["Age"])


print(dataset.describe())

plt.figure(figsize=(10,4))

plt.subplot(1, 2, 1)
plt.title("After Standardization Scaling-Age")
sns.histplot(dataset["Age"],kde=True)

plt.subplot(1,2,2)
plt.title("After Standardization scaling-Income")
sns.histplot(dataset["Income"],kde=True)
plt.tight_layout()
plt.show()


