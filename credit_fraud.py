import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"c:\Users\kumar\OneDrive\Desktop\Daatasheet\creditcard.csv")
print(dataset.head(6))
print("\n")
print(dataset.tail())

missing=dataset.isnull().sum()
print(missing)

print(dataset.info())

print(dataset.describe())
print("\n")

print(dataset.size)
print(dataset['Class'].value_counts())

legit=dataset[dataset.Class==0]
fraud=dataset[dataset.Class==1]

print(legit.shape)
print(fraud.shape)

print(legit.describe())
print(fraud.describe())

print("Compare to dataset yes with no \n")

print(dataset.groupby('Class').mean())

legit_sample=legit.sample(n=492)

print(legit_sample.shape)

new_datset=pd.concat([legit_sample,fraud],axis=0)

print(dataset.head(5))
print("Shape:",new_datset.shape)
print(new_datset["Class"].value_counts())

x=new_datset.iloc[:,:-1]
y=new_datset["Class"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.preprocessing import StandardScaler
sta=StandardScaler()
x_train_scaled=sta.fit_transform(x_train)
x_test_scales=sta.transform(x_test)

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
la=LogisticRegression()
la.fit(x_train_scaled,y_train)
y_pred=la.predict(x_test_scales)

print("Accuracy in Logistic:",la.score(x_test_scales,y_pred)*100,"%")


import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model=Sequential()

model.add(Dense(32,activation='relu',input_dim=30))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False
)

history = model.fit(x_train_scaled, y_train, validation_data=(x_test_scales, y_test), epochs=3500, callbacks=callback)

y_pd=model.predict(x_test_scales)
y_pd_classes = (y_pd > 0.5).astype(int) 

print("Accuracy in deep learning:",accuracy_score(y_test,y_pd_classes)*100,"%")


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
