import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

# Load dataset
dataset = pd.read_csv("king.txt")
print("🔹 First 50 rows:\n", dataset.head(50))

missing=dataset.isna().sum()
print(missing)

dataset.select_dtypes(include="object").columns
from sklearn.impute import SimpleImputer
la=SimpleImputer(strategy="most_frequent")
ar=la.fit_transform(dataset[["Name","Date_of_Birth","State","District"]])
dataset[["Name","Date_of_Birth","State","District"]]=pd.DataFrame(ar,columns=["Name","Date_of_Birth","State","District"])


dataset.select_dtypes(include="int64").columns
from sklearn.impute import SimpleImputer
la=SimpleImputer(strategy="mean")
ar=la.fit_transform(dataset[["Income","Age"]])
dataset[["Income","Age"]]=pd.DataFrame(ar,columns=["Income","Age"])
print(dataset.head(50))
missing=dataset.isna().sum()
print(missing)

#finding outliear
sns.boxplot(x="Income",data=dataset)
plt.show()

print(dataset.describe())
sns.distplot(dataset["Income"])
plt.show()