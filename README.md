# oil_leak
Leak detection and time series forecasting
### It is still on dev stage but already completely functional for daily forecasting

### Build docker image like this
```
docker build -t ollin18/presage:0.1 .
```

## Run the docker image this way
```
docker run -it --rm \
-v $(pwd)/data:/data \
-v $(pwd)/figs:/figs \
-v $(pwd)/src:/src \
ollin18/presage:0.1 /bin/bash
```

## Inside docker run:
### To train
```
src/train.py --yi "2017" --yf "2017" --mi "01" --mf "03"
```
You can change the parameters, they stand for initial and final year and month.

### To predict
```
/src/predict.py --y "2017" --m "12" --d "17"
```
The parameters stands for year, month, day.

TODO
* Add parameters such as epoch and batch size
* Connect to SQL database
* Real-time predictions
* CUDA integration
* Pass args through docker run
