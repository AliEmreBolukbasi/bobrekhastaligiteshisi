# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:14:21 2021

@author: USER
"""
#1. adım kütüphane yükle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2. adım  veri setini yükle

dataset = pd.read_csv('veriseti.csv')

X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)

#3. adım eksik verileri doldur

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer=imputer.fit(X[:,:])
X[:,:]=imputer.transform(X[:,:])
print(X)

#4.adım kategorik verilerin düzenlenmesi

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_y=LabelEncoder()
Y=labelEncoder_y.fit_transform(Y)
print(Y)

#5. adım veri setini eğitim ve test olarak böl

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#6. adım özellik ölçeklendirme

from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X_train=sc_X.fit_transform(X_train)
print(X_train)
X_test=sc_X.transform(X_test)
print(X_test)

#7.adım lineer regresyon uygula
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#8.adım test veri setinden tahminlemede bulun
Y_pred=regressor.predict(X_test)

print(Y_pred)
print(Y_test)

#9. adım Grafik çizelim
plt.scatter(X_train[:,0],Y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')

plt.title("eğitim veri seti")
plt.xlabel("yaş")
plt.ylabel("hastalık")
plt.show()
plt.scatter(X_test[:,0],Y_test,color='red')
plt.plot(X_test, regressor.predict(X_test),color='blue')
plt.title("eğitim veri seti")
plt.xlabel("yaş")
plt.ylabel("hastalık")
plt.show()


#YAPAY SİNİR AĞI

#1. adım kütüphane yükle
import keras
from  keras.models import Sequential
from  keras.layers import Dense

#2. adım YSA başlat

classifier =Sequential()

#3. adım girdi katmanı ve ilk gizli katmanı oluştur

classifier.add(Dense(units=6,input_dim=12,activation='relu',kernel_initializer="uniform"))

#4. adım ikinci gizli katmanı oluştur

classifier.add(Dense(units=6,activation='relu',kernel_initializer="uniform"))

#5.adım çıktı katmanı

classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer="uniform"))

#6.adım ysa derle

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#7.adım ysa eğitmeye başla

classifier.fit(X_train,Y_train,batch_size=10,epochs=150)

#TEST KISMI

#1. adım test verileri için tahminlemeleri hesapla

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
print(y_pred)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(Y_test,y_pred)
print(cm)





