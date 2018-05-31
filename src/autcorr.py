#!/usr/bin/env python

from presage_functions import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import plot_model

from matplotlib import pyplot

from pandas import DataFrame
from pandas import concat

from numpy import concatenate

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

### Read the data
DON = read_datacsv("BF_DON","2017","02","02")
s450 = read_datacsv("BF_450","2017","02","02")
GRN = read_datacsv("BF_GRN","2017","02","02")

DON = read_month("BF_DON","2017","01")
s450 = read_month("BF_450","2017","01")
GRN = read_month("BF_GRN","2017","01")

DON_MTR_12 = DON.loc[DON['DI_DEVICE_NAME'] == "BF_DON_MTR-12IN"]
DON_MTR_12 = DON_MTR_12.loc[DON_MTR_12['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
DON_MTR_12 = DON_MTR_12[["TD_TAG_VALUE"]]
DON_MTR_12.columns = ["DON_FLOW"]

s450_MTR = s450.loc[s450['DI_DEVICE_NAME'] == "BF_450_MTR-OUTGOING"]
s450_MTR = s450_MTR.loc[s450_MTR['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
s450_MTR = s450_MTR[["TD_TAG_VALUE"]]
s450_MTR.columns = ["450_FLOW"]

GRN_MTR = GRN.loc[GRN['DI_DEVICE_NAME'] == "BF_GRN_MTR-STATION"]
GRN_MTR = GRN_MTR.loc[GRN_MTR['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
GRN_MTR = GRN_MTR[["TD_TAG_VALUE"]]
GRN_MTR.columns = ["GRN_FLOW"]

### Moving average
DON_MTR_12 = mov_avg(DON_MTR_12,"10s","1min")
s450_MTR = mov_avg(s450_MTR,"10s","1min")
GRN_MTR = mov_avg(GRN_MTR,"10s","1min")

both = DON_MTR_12.join(s450_MTR).fillna(0)
both = both.join(GRN_MTR).fillna(0)
values = both.values
encoder = LabelEncoder()
values[:,1] =  encoder.fit_transform(values[:,1])
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 48, 1)


for i in range(1,48):
    behind = str(i)
    reframed = reframed.drop('var3(t-'+behind+')',axis=1)

ref = series_to_supervised(s450_MTR, 12, 1)
reframed = reframed.join(ref)
reframed = reframed.join(GRN_MTR)

values = reframed.values
n_train_hours = len(values)
n_train_hours = 7500 * 30
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1500, batch_size=7, validation_data=(test_X, test_y), verbose=2, shuffle=False)





# otro modelo
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),\
    return_sequences=True))
#  model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
filepath = "../data/best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(train_X, train_y, epochs=1000, batch_size=250,\
        validation_data=(test_X, test_y), verbose=2, shuffle=False,\
        callbacks=callbacks_list)



# make a prediction
yhat = model.predict(train_X)
pred = DataFrame(yhat)
pred.columns = ["pred"]

plt.figure()
pred.plot(figsize=(30,3));
plt.savefig("../yhatm.png")


plt.figure()
real.plot(figsize=(30,3));
plt.savefig("../real.png")

#Aqui abajo cambie cosas pero no debe ser train_y
real = DataFrame(train_y)
real.columns = ["real"]

ambas = pred.join(real)
plt.figure()
ambas.plot(figsize=(30,5),title="Prediction vs real 02/02/2017");
plt.savefig("../pred_vs_real_no_grn.png")


mean_squared_error(yhat,train_y)


test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
