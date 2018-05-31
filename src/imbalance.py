#!/usr/bin/env python

from presage_functions import *

DON = read_datacsv("BF_DON","2017","01","01")
s450 = read_datacsv("BF_450","2017","01","01")
GRN = read_datacsv("BF_GRN","2017","01","04")

DON_MTR_12 = DON.loc[DON['DI_DEVICE_NAME'] == "BF_DON_MTR-12IN"]
DON_MTR_12 = DON_MTR_12.loc[DON_MTR_12['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
DON_MTR_12 = DON_MTR_12[["TD_TAG_VALUE"]]

#  DON_MTR_STATION = DON.loc[DON['DI_DEVICE_NAME'] == "BF_DON_MTR-STATION"]
#  DON_MTR_THBIRD = DON.loc[DON['DI_DEVICE_NAME'] == "BF_DON_MTR-THBIRD_INCOMING"]

s450_MTR = s450.loc[s450['DI_DEVICE_NAME'] == "BF_450_MTR-OUTGOING"]
s450_MTR = s450.loc[s450['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
s450_MTR = s450_MTR[["TD_TAG_VALUE"]]

GRN_MTR = GRN.loc[GRN['DI_DEVICE_NAME'] == "BF_GRN_PT-INCOMING"]
GRN_MTR = GRN.loc[GRN['DI_DEVICE_NAME'] == "BF_GRN_MTR-STATION"]
GRN_MTR = GRN.loc[GRN['TI_TAG_ID'] == 406132]
GRN_MTR = GRN.loc[GRN['TI_TAG_DESCRIPTION'] == "Meter flow rate"]

GRN_MTR = GRN_MTR[["TD_TAG_VALUE"]]
GRN_MTR = GRN_MTR[["TI_TAG_DESCRIPTION","TD_TAG_VALUE"]]
GRN_MTR = GRN[["DI_DEVICE_NAME","TI_TAG_DESCRIPTION","TD_TAG_VALUE"]]

DON_MTR_12 = mov_avg(DON_MTR_12,"10s","1min")
s450_MTR = mov_avg(s450_MTR,"10s","1min")
GRN_MTR = mov_avg(GRN_MTR,"10s","1min")

imbalance = DON_MTR_12+s450_MTR-GRN_MTR

event = DON[["ML_EVENTCODE"]]
event = event.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')

cuales=GRN_MTR.groupby('DI_DEVICE_NAME').resample('10s').mean().unstack('DI_DEVICE_NAME')
cuales=cuales.rolling("1min").mean()

s450_MTR.head(10)
GRN_MTR.head(25)
cuales.head()

plt.figure()
cuales.plot(subplots=True, figsize=(24, 6),style='.');
plt.savefig("figurasavg.png")
plt.show()

plt.figure()
#  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
plt.show()

plt.figure()
#  imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3));
GRN_MTR.plot(y='TD_TAG_VALUE',figsize=(20,8));
plt.savefig("../figuraa.png")
plt.show()

plt.figure()
imbalance.plot(y='TD_TAG_VALUE',figsize=(30,3),style='.');
plt.show()

### Graficar todo
plot_all("2017","01","01","10s","1min")
