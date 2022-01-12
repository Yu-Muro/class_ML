# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import tensorflow as tf
import tensorflow.keras.datasets
import tensorflow.keras.utils
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Reshape, Flatten, Conv2D, MaxPooling2D

(data_train, label_train), (data_test, label_test) = tf.keras.datasets.cifar10.load_data()
data_train = data_train.astype( 'float32' ) / 255
data_test  = data_test.astype( 'float32' ) / 255
label_train = label_train.astype( 'int32' )
label_test  = label_test.astype( 'int32' )

label_num = 10
one_hot_train = tf.keras.utils.to_categorical( label_train, label_num ) #ytrain
one_hot_test  = tf.keras.utils.to_categorical( label_test,  label_num ) #ytest

#モデルを構築
model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(data_train, one_hot_train, batch_size=128, epochs=20, verbose=1, validation_data=(data_test, one_hot_test))

#モデルと重みを保存
json_string=model.to_json()
open('cifar10_cnn.json', "w").write(json_string)
model.save_weights('cifar10_cnn.h5')

#モデルの表示
model.summary()

#評価
score=model.evaluate(data_test, one_hot_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# ここで CNN を評価、混同行列を求める
label_pred = model.predict( data_test ).argmax( axis=1 )
print( sklearn.metrics.confusion_matrix( label_test, label_pred ) )

