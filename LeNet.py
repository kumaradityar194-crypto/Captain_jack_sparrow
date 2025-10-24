import tensorflow
from tensorflow import keras
from keras.layers import Dense,Conv2D , Flatten, AveragePooling2D
from keras import Sequential
from keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model=Sequential()

model.add(Conv2D(6,kernel_size=(5,5),padding='valid',activation='relu',input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2,padding='valid'))
model.add(Conv2D(16,kernel_size=(5,5),padding='valid',activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(120,activation='tanh'))
model.add(Dense(84,activation='tanh'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

pred=model.predict(x_test)

pred_classes = pred.argmax(axis=1)
print(pred_classes[:10]) 
