import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns

dataset=pd.read_csv("king.txt")
print(dataset.head(50))

missing=dataset.isnull().sum()
print(missing)
print("")
print("Name_mpode:",dataset["Name"].mode()[0])
print("Age_mean:",dataset["Age"].mean())
print("Date_of_Birth_Mode",dataset["Date_of_Birth"].mode()[0])
print("Income_Mean:",dataset["Income"].mean())
print("State_mode:",dataset["State"].mode()[0])
print("DISTRICT_Mode:",dataset["District"].mode()[0])

dataset.info()

dataset.select_dtypes(include="object").columns

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy="most_frequent")


ar = si.fit_transform(dataset[['Name', 'State','Date_of_Birth', 'District']])

dataset[['Name','Date_of_Birth','State', 'District']] = pd.DataFrame(ar, columns=['Name','Date_of_Birth', 'State', 'District'])


dataset.select_dtypes(include="float64").columns

from sklearn.impute import SimpleImputer

si = SimpleImputer(strategy="mean")


ar = si.fit_transform(dataset[['Age', 'Income']])

dataset[['Age', 'Income']] = pd.DataFrame(ar, columns=['Age', 'Income'])


print(dataset.head(50))

missing=dataset.isnull().sum()
print(missing)

en_data=dataset[["Age","Name","Date_of_Birth","Income","State","District"]]
encoding=pd.get_dummies(en_data)
print("encoding")
print(encoding.head(50))

from sklearn.preprocessing import OneHotEncoder

# Select only categorical columns for encoding
categorical = dataset[["Name", "Date_of_Birth", "State", "District"]]

# Initialize encoder
ohe = OneHotEncoder(sparse_output=False)


# Fit and transform
ar = ohe.fit_transform(categorical)

# Get proper column names
cols = ohe.get_feature_names_out(["Name", "Date_of_Birth", "State", "District"])

# Make DataFrame
encoded_df = pd.DataFrame(ar, columns=cols)

# Combine with numerical data
final_df = pd.concat([dataset[["Age", "Income"]].reset_index(drop=True), encoded_df], axis=1)

# Print output
print(final_df.head(50))
