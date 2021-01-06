# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('depression_dataset.csv')

X = dataset.iloc[:, 2:21].values
y = dataset.iloc[:, 22].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(output_dim=12, input_dim =19, kernel_initializer = 'uniform', activation = 'relu' ))



classifier.add(Dense(output_dim=12, kernel_initializer = 'uniform', activation = 'relu' ))


 
classifier.add(Dense(output_dim=1, kernel_initializer = 'uniform', activation = 'sigmoid' ))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)



from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
