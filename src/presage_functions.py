import numpy as np
import pandas as pd
import glob
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

def read_datacsv(place,year,month,day):
    firstm = int(month)
    firstm = str(firstm)
    dataf = pd.read_csv("/data/"+place+"/"+year+"/"\
            +place+"_"+year+"_"+firstm+"/"\
            +place+"_"+year+month+day+".csv",\
            parse_dates=[1])
    dataf = dataf.loc[dataf["THE_TIME_STAMP"] != '\x1a']
    dataf['THE_TIME_STAMP'] = pd.to_datetime(dataf['THE_TIME_STAMP'],format="%Y-%m-%d %H:%M:%S.%f")
    dataf = dataf.sort_values('THE_TIME_STAMP')
    dataf = dataf.set_index('THE_TIME_STAMP')
    return(dataf)

def read_month(place,year,month):
    month = int(month)
    month = str(month)
    path = r'/data/'+place+'/'+year+'/'+place+'_'+year+'_'+month+'/'
    files = glob.glob(path + "*.csv")
    dataf = pd.DataFrame()
    the_list = []
    for csv_file in files:
        dataf = pd.read_csv(csv_file,parse_dates=[1])
        the_list.append(dataf)

    dataf = pd.concat(the_list)
    dataf = dataf.loc[dataf["THE_TIME_STAMP"] != '\x1a']
    dataf['THE_TIME_STAMP'] = pd.to_datetime(dataf['THE_TIME_STAMP'],format="%Y-%m-%d %H:%M:%S.%f")
    dataf = dataf.sort_values('THE_TIME_STAMP')
    dataf = dataf.set_index('THE_TIME_STAMP')
    return(dataf)

def mov_avg(df,every,window):
    return(df.resample(every).mean().rolling(window).mean())


def plot_all(year,month,day,every,window):

    #  DON = read_datacsv("BF_DON",year,month,day)
    #  s450 = read_datacsv("BF_450",year,month,day)
    #  GRN = read_datacsv("BF_GRN",year,month,day)

    DON = read_month("BF_DON",year,month)
    s450 = read_month("BF_450",year,month)
    GRN = read_month("BF_GRN",year,month)

    DON_MTR_12 = DON.loc[DON['DI_DEVICE_NAME'] == "BF_DON_MTR-12IN"]
    DON_MTR_12 = DON_MTR_12.loc[DON_MTR_12['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
    DON_MTR_12 = DON_MTR_12[["TD_TAG_VALUE"]]

    s450_MTR = s450.loc[s450['DI_DEVICE_NAME'] == "BF_450_MTR-OUTGOING"]
    s450_MTR = s450_MTR.loc[s450_MTR['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
    s450_MTR = s450_MTR[["TD_TAG_VALUE"]]

    GRN_MTR = GRN.loc[GRN['DI_DEVICE_NAME'] == "BF_GRN_MTR-STATION"]
    GRN_MTR = GRN_MTR.loc[GRN_MTR['TI_TAG_DESCRIPTION'] == "Meter flow rate"]
    GRN_MTR = GRN_MTR[["TD_TAG_VALUE"]]

    DON_MTR_12 = mov_avg(DON_MTR_12,every,window)
    s450_MTR = mov_avg(s450_MTR,every,window)
    GRN_MTR = mov_avg(GRN_MTR,every,window)

    imbalance = DON_MTR_12+s450_MTR-GRN_MTR

    event_don = DON[["ML_EVENTCODE"]]
    event_don = event_don.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')

    event_450 = s450[["ML_EVENTCODE"]]
    event_450 = event_450.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')

    event_grn = GRN[["ML_EVENTCODE"]]
    event_grn = event_grn.groupby('ML_EVENTCODE').resample('10s').count().unstack('ML_EVENTCODE')

    fig, axes = plt.subplots(nrows=2,ncols=1,sharex=True)
    ax = DON_MTR_12.plot(y='TD_TAG_VALUE',figsize=(20,5),ax=axes[0],\
            title=year+" "+month+" "+day)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    s450_MTR.plot(ax=ax)
    GRN_MTR.plot(ax=ax)
    imbalance.plot(ax=ax)
    ax.legend(["DON","450","GRN","Imbalance"])

    event_don = event_don.replace(0,np.nan)
    event_450 = event_450.replace(0,np.nan)
    event_grn = event_grn.replace(0,np.nan)
    try:
        ax = event_don.plot(y="ML_EVENTCODE",style='.',ax=axes[1],title="EVENTCODE")
    except:
        print("No eventcode")
    try:
        event_450.plot(ax=ax,style='.')
    except:
        print("No eventcode")
    try:
        event_grn.plot(ax=ax,style='.')
    except:
        print("No eventcode")
    ax.legend(["DON","450","GRN"])
    plt.savefig(year+"_"+month+"_"+day+".png")
    #  plt.show()

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

