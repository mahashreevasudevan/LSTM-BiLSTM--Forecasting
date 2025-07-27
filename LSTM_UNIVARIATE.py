import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.4f' % x)
import seaborn as sns
sns.set_context("paper", font_scale=1.3)
sns.set_style('white')
import warnings
warnings.filterwarnings('ignore')
from time import time
import matplotlib.ticker as tkr
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
from statsmodels.tsa.stattools import pacf


import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

df=pd.read_csv('Dataset_path')
visdata = df.VISIBILITY.values 
visdata = visdata.astype('float32')
visdata = np.reshape(visdata, (-1, 1))

#normalizing the variable 
scale = MinMaxScaler(feature_range=(0, 1))
visdata = scale.fit_transform(visdata)

#splitting the dataset into training and testing data 
train_size = int(len(visdata) * #specify the training percentage)
test_size = len(visdata) - train_size
train, test = visdata[0:train_size,:], visdata[train_size:len(visdata),:]

#converting array of values into dataset matrix 

def vis_dataset(visdata, look_back=1):
    X, Y = [], []
    for i in range(len(visdata)-look_back-1):
        a = visdata[i:(i+look_back), 0]
        X.append(a)
        Y.append(visdata[i + look_back, 0])
    return np.array(X), np.array(Y)
    
look_back = #how much of the previous day data to be considered
X_train, Y_train = vis_dataset(train, look_back)
X_test, Y_test = vis_dataset(test, look_back)

# reshape input to be 3D [samples, time steps, features] 
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
     



#Defining the model
model = Sequential()
model.add(LSTM(#number of units, input_shape=(X_train.shape[#specify value], X_train.shape[#specify value])))
model.add(Dropout(#dropout percentage))
model.add(Dense(#number of unit in output layer))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(X_train, Y_train, epochs=#number of epoch, batch_size= #batch size, validation_data=(X_test, Y_test), 
                    callbacks=[EarlyStopping(monitor='val_loss', patience= #Specify patience)], verbose=1, shuffle=False)

model.summary()
     



train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# invert predictions
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])
print('Train Mean Absolute Error:', mean_absolute_error(Y_train[0], train_predict[:,0]))
print('Train Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0])))
print('Test Mean Absolute Error:', mean_absolute_error(Y_test[0], test_predict[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0])))



plt.figure(figsize=(8,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')
plt.show();



aa=[x for x in range(#sample size)]
plt.figure(figsize=(8,4))
plt.plot(aa, Y_test[0][:#sample size], marker='.', label="actual")
plt.plot(aa, test_predict[:,0][:#sample size], 'r', label="prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('VISIBILITY', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)
plt.show();
