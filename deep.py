import numpy as np
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns

dataset=pd.read_csv(r"c:\Users\kumar\Downloads\customer_churn_synthetic.csv")
print(dataset.head(10))

print("Size:",dataset.size)

missing=dataset.isnull().sum()
print(missing)
print("")

print(dataset.info())

dataset.drop(columns=['CustomerID','Balance','HasCreditCard','EstimatedSalary'],inplace=True)

print(dataset.head(10))

print("Uniq:",dataset["Gender"].unique())

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
la.fit_transform(dataset["Gender"])

dataset["Gender"]=la.transform(dataset["Gender"])

print(dataset.head(5))
print("")
print("Duplicates:",dataset.duplicated())

X=dataset.iloc[:,:-1]
Y=dataset["Churn"]

from imblearn.over_sampling import RandomOverSampler
ra=RandomOverSampler()
ra_x,ra_y=ra.fit_resample(X,Y)

print(ra_y.value_counts())


from sklearn.preprocessing import StandardScaler
sa=StandardScaler()
ar=sa.fit_transform(ra_x)
ra_x=pd.DataFrame(ar,columns=X.columns)
print("")
print("Ra_x",ra_x)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(ra_x,ra_y,random_state=42,test_size=0.25)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(5,)))          # pehle input layer define karo
model.add(Dense(32, activation="sigmoid"))
model.add(Dense(16, activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))
print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='Adam')
model.fit(x_train,y_train,epochs=100)
print("")
print("Wighta and baias layesr 0\n")
print(model.layers[0].get_weights())
print("")
print("Wighta and baias layesr 1\n")
print(model.layers[1].get_weights())

print("")
print("Wighta and baias layesr 2\n")

print(model.layers[2].get_weights())

#print("Prediction:",model.predict(x_test))

y_log=model.predict(x_test)

y_prd=np.where(y_log>0.5,1,0) 
#print(y_prd)

print("Enter Valid_Information\n")

a=int(input("Enter The Age:"))
b=int(input("ENter the Gender (Male=1/Female=0):"))
c=int(input("Enter the tenure:"))
d=int(input("Enter the numproduct:"))
e=int(input("Enter isActivemember(yes=1/no=0):"))

input=pd.DataFrame([[a,b,c,d,e]],columns=["Age","Gender","Tenure","NumOfProducts","IsActiveMember"])

input_scaled=sa.transform(input)

prd=model.predict(input_scaled)

from sklearn.metrics import accuracy_score
print("Accuiracy:",accuracy_score(y_test,y_prd)*100)

#print(prd)

print("")
print("Prediction:-")
if prd[0] > 0.5:
    print("Employ Stay in Company:")
    print("")
    print("Thanks to Kumar_Aditya_Raj")
    print("")
    
else:
    print("Empoly leave the company:")
    print("")
    print("Thanks to Kumar_Aditya_Raj")
    print("")
    


