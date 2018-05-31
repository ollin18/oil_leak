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

### Read the data
DON = read_datacsv("BF_DON","2017","01","01")
s450 = read_datacsv("BF_450","2017","01","01")
GRN = read_datacsv("BF_GRN","2017","01","01")


DON = read_month("BF_DON","2017","01")
s450 = read_month("BF_450","2017","01")
GRN = read_month("BF_GRN","2017","01")

### Get only the required data
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

#  GRN_MTR = GRN_MTR[["TD_TAG_VALUE"]]
#  GRN_MTR = GRN_MTR[["TI_TAG_DESCRIPTION","TD_TAG_VALUE"]]
#  GRN_MTR = GRN[["DI_DEVICE_NAME","TI_TAG_DESCRIPTION","TD_TAG_VALUE"]]

### Moving average
DON_MTR_12 = mov_avg(DON_MTR_12,"10s","1min")
s450_MTR = mov_avg(s450_MTR,"10s","1min")
GRN_MTR = mov_avg(GRN_MTR,"10s","1min")

### Define invalance
imbalance = DON_MTR_12+s450_MTR-GRN_MTR

### Resample for all the event's codes
event_don = DON[["ML_EVENTCODE"]]
event_don = event_don.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')
#  event_don.columns = ["1_DON_EV","2_DON_EV","3_DON_EV"]
subevent_don = DON[["ML_SUBEVENTCODE"]]
subevent_don = subevent_don.groupby('ML_SUBEVENTCODE').resample('10s').count().unstack('ML_SUBEVENTCODE')

event_450 = s450[["ML_EVENTCODE"]]
event_450 = event_450.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')
#  event_450.columns = ["1_450_EV","2_450_EV","3_450_EV"]
subevent_450 = s450[["ML_SUBEVENTCODE"]]
subevent_450 = subevent_450.groupby('ML_SUBEVENTCODE').resample('10s').count().unstack('ML_SUBEVENTCODE')

event_GRN = GRN[["ML_EVENTCODE"]]
event_GRN = event_GRN.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')
#  event_GRN.columns = ["1_GRN_EV","2_GRN_EV","3_GRN_EV"]
subevent_GRN = GRN[["ML_SUBEVENTCODE"]]
subevent_GRN = subevent_GRN.groupby('ML_SUBEVENTCODE').resample('10s').count().unstack('ML_SUBEVENTCODE')

#  cuales=GRN_MTR.groupby('DI_DEVICE_NAME').resample('10s').mean().unstack('DI_DEVICE_NAME')
#  cuales=cuales.rolling("1min").mean()
the_data = pd.concat([DON_MTR_12, s450_MTR, event_don, subevent_don,\
        event_450, subevent_450,\
        event_GRN, subevent_GRN,\
        GRN_MTR],axis=1).fillna(0)







the_data = pd.concat([DON_MTR_12, GRN_MTR],axis=1).fillna(0)



#  plt.figure()
#  cuales.plot(subplots=True, figsize=(24, 6),style='.');
#  plt.savefig("figurasavg.png")
#  plt.show()
#
#  plt.figure()
#  #  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
#  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
#  plt.show()
#
#  plt.figure()
#  #  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
#  GRN_MTR.plot(y='TD_TAG_VALUE',figsize=(20,8));
#  plt.savefig("../figuraa.png")
#  plt.show()
#
#  plt.figure()
#  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3),style='.');
#  plt.show()
#
#  ### Graficar todo
#  plot_all("2017","01","01","10s","1min")

values = the_data.values
#  n_train_hours = int((24 * 60 * 30) * 0.8)
n_train_hours = int(6912)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, 0:-1], train[:, -1]
test_X, test_y = test[:, 0:-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



# Using past time









##### New reshape
values=the_data.values
seq_length = 30
train_X, train_y = train[:, 0:-1], train[:, -1]
dataX = []
dataY = []
for i in range(0, len(values) - seq_length):
    seq_in = train_X[i:i+seq_length]
    seq_out = train_y[i:i+seq_length]
    dataX.append([e for e in seq_in])
    dataY.append([e for e in seq_out])

n_train_hours = int(6912)
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]



##### New network
model = Sequential()
model.add(LSTM(64,\
        activation='relu',
        batch_input_shape=(80,30,1),\
        stateful=True,\
        return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mae',optimizer='adam')


# design network
model = Sequential()
model.add(LSTM(24, input_shape=(train_X.shape[1], train_X.shape[2]),\
    return_sequences=True))
#  model.add(Dropout(0.2))
model.add(LSTM(24))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
filepath = "../data/best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(train_X, train_y, epochs=1000, batch_size=10,\
        validation_data=(test_X, test_y), verbose=2, shuffle=False,\
        callbacks=callbacks_list)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
plt.savefig("../lost.png")

# plot model
plot_model(model, to_file='../model.png')


# prediction

DON = read_datacsv("BF_DON","2017","01","02")
s450 = read_datacsv("BF_450","2017","01","02")
GRN = read_datacsv("BF_GRN","2017","01","02")

DON = read_month("BF_DON","2017","02")
s450 = read_month("BF_450","2017","02")
GRN = read_month("BF_GRN","2017","02")

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

to_forecast = pd.concat([DON_MTR_12\
        #  event_450, subevent_450,\
        #  event_GRN, subevent_GRN\
        ],axis=1).fillna(0)


val = to_forecast.values
val = val.reshape((val.shape[0],1,val.shape[1]))
print(val.shape)


#  n_train_hours = int((24 * 60 * 30) * 0.8)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


filename = "../data/best_weights.hdf5"
model.load_weights(filename)
model.compile(loss='mae', optimizer='rmsprop')
prediction = model.predict(val, verbose=0)



GRN_MTR
prediction
pre=pd.DataFrame(prediction)

plt.figure()
pre.plot(figsize=(30,3));
GRN_MTR.plot(y='GRN_FLOW',figsize=(30,3));
plt.savefig("../foreca.png")





from pandas import DataFrame
from pandas import concat

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


raw = DataFrame()
raw['ob1'] = [x for x in range(10)]
raw['ob2'] = [x for x in range(50, 60)]
values = raw.values
data = series_to_supervised(DON_MTR_12, 128, 1)
the_data = data.join(GRN_MTR)
print(data)




val = data.values
val = val.reshape((val.shape[0],1,val.shape[1]))
