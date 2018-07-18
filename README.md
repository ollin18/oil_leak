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
-p 1433:1433 \
ollin18/presage:0.1 /bin/bash
```

## Inside the container run:
### To train
```
/src/train.py --yi "2017" --yf "2017" --mi "01" --mf "03" --di "01" --df "31"
```
You can change the parameters. They stand for initial and final year and month.

### To predict
```
/src/predict.py --yi "2017" --yf "2017" --mi "04" --mf "04" --di "17" --df "19"
```
The parameters stand for year, month, day.

TODO
* Add parameters such as epoch and batch size
* Real-time predictions
* Pass args through docker run
