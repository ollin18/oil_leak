#!/usr/bin/env python3

from presage_functions import *

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils import plot_model

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot
import matplotlib.pyplot as plt

from pandas import DataFrame
from pandas import concat

from numpy import concatenate
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--yi",type=str, help="The initial year")
parser.add_argument("--yf",type=str, help="The final year")
parser.add_argument("--mi",type=str, help="The initial month")
parser.add_argument("--mf",type=str, help="The final month")

args = parser.parse_args()

DON  = read_month("BF_DON",args.yi,args.mi)
s450 = read_month("BF_450",args.yi,args.mi)
GRN  = read_month("BF_GRN",args.yi,args.mi)

yeari = int(args.yi)
yearf = int(args.yf)
monthi = int(args.mi)
monthf = int(args.mf)

for year in np.arange(yeari,yearf+1):
    for month in np.arange(monthi,monthf):
        DON=DON.append(read_month("BF_DON",str(year),str(month).zfill(2)))
        s450=s450.append(read_month("BF_450",str(year),str(month).zfill(2)))
        GRN=GRN.append(read_month("BF_GRN",str(year),str(month).zfill(2)))


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
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
joblib.dump(scaler, "/data/scaler.save")
scaled = scaler.fit_transform(values)

reframed = series_to_supervised(scaled, 180, 1)

for i in range(1,181):
    behind = str(i)
    reframed = reframed.drop('var3(t-'+behind+')',axis=1)

values = reframed.values
n_train_hours = int(len(values)*0.80)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Keras Network
model = Sequential()
model.add(Dense(180, input_shape=(train_X.shape[1], train_X.shape[2]),\
        activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(180,activation='relu'))
model.add(Dense(180,activation='relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
filepath = "/data/best_weights3m.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

train_y = np.expand_dims(np.expand_dims(train_y,-1),-1)
test_y  = np.expand_dims(np.expand_dims(test_y,-1),-1)

history = model.fit(train_X, train_y, epochs=1000, batch_size=500,\
        validation_data=(test_X, test_y), verbose=2, shuffle=False,\
        callbacks=callbacks_list)

# make a prediction
yhat = model.predict(test_X)
yhat = yhat[:,0]
allset = np.concatenate((yhat,yhat),axis=1)
allset = np.concatenate((allset,yhat),axis=1)
yhat = scaler.inverse_transform(allset)[:,[2]]

pred = DataFrame(yhat)
pred.columns = ["pred"]
real = GRN_MTR.iloc[(len(GRN_MTR)-len(pred)):,:]
real.columns = ["real"]
ambas = pd.concat([pred.set_index(real.index),real],axis=1,ignore_index=False)
ambas['diff'] = ambas.real - ambas.pred
ambas = DataFrame(ambas)
ambas.columns = ["pred","real","diff"]

# Save results
plt.figure()
plot = ambas.plot(figsize=(30,5),title="Prediction vs real "+"test "\
        +args.yi+"/"+args.yf+" "+args.mi+"/"+args.mf);
fig = plot.get_figure()
fig.savefig("/figs/Prediction_vs_real_test_"+args.yi+"-"+args.yf\
        +"_"+args.mi+"-"+args.mf+".png")

ambas.to_csv("/data/Prediction_vs_real_test_"+args.yi+\
        "_"+args.yf+"_"+args.mi+"_"+args.mf+".csv",index=False)
