# stock-market-predictions-with-bayesian-lstm
Predict the trends for your favourite stocks using Bayesian LSTM neural network. THe idea is to make a combined model with several stocks in a particular sector and finally get predictions for each one individually.

usage: main.py [-h] [--tickers [TICKERS ...]] [--start_date START_DATE] [--end_date END_DATE] [--validation_days VALIDATION_DAYS] [--epochs EPOCHS] [--layers LAYERS] [--input_time_steps INPUT_TIME_STEPS] [--output_time_steps OUTPUT_TIME_STEPS]
               [--batch_size BATCH_SIZE] [--lr LR] [--output_dir OUTPUT_DIR]

## Example for usage:


main.py --output_dir new --tickers 'MSFT' 'AAPL' 'AMZN' --validation_days 365 --layers 2 tickers =  ['MSFT', 'AAPL', 'AMZN']



dates (start, end, validation) 2018-11-08 2023-11-07 2022-11-07
epochs, lr, batch_size =  1000 1e-04 32
time steps (input, output) =  120 120
Directory 'new/' already exists.
[*********************100%%**********************]  1 of 1 completed
new/MSFT.csv
[*********************100%%**********************]  1 of 1 completed
new/AAPL.csv
[*********************100%%**********************]  1 of 1 completed
new/AMZN.csv


Train all stocks as one combined model but training in sequentially..
new/MSFT.csv
tran dataset <bound method DataFrame.info of             Date        Open        High         Low       Close   Adj Close    Volume
0     2018-11-08  111.800003  112.209999  110.910004  111.750000  105.875481  25644100
1     2018-11-09  110.849998  111.449997  108.760002  109.570000  103.810089  32039200
2     2018-11-12  109.419998  109.959999  106.099998  106.870003  101.252022  33621800
3     2018-11-13  107.550003  108.739998  106.639999  106.940002  101.318329  35374600
4     2018-11-14  108.099998  108.260002  104.470001  104.970001   99.881554  39495100
...          ...         ...         ...         ...         ...         ...       ...
1000  2022-10-31  233.759995  234.919998  231.149994  232.130005  229.908890  28357300
1001  2022-11-01  234.600006  235.740005  227.330002  228.169998  225.986786  30592300
1002  2022-11-02  229.460007  231.300003  220.039993  220.100006  217.994019  38407000
1003  2022-11-03  220.089996  220.410004  213.979996  214.250000  212.199982  36633900
1004  2022-11-04  217.550003  221.589996  213.429993  221.389999  219.271667  36789100

[1005 rows x 7 columns]>
test dataset <bound method DataFrame.info of             Date        Open        High         Low       Close   Adj Close    Volume
1005  2022-11-07  221.990005  228.410004  221.279999  227.869995  225.689636  33498000
1006  2022-11-08  228.699997  231.649994  225.839996  228.869995  226.680084  28192500
1007  2022-11-09  227.369995  228.630005  224.330002  224.509995  222.361816  27852900
1008  2022-11-10  235.429993  243.330002  235.000000  242.979996  240.655075  46268000
1009  2022-11-11  242.990005  247.990005  241.929993  247.110001  244.745560  34620200
...          ...         ...         ...         ...         ...         ...       ...
1251  2023-10-31  338.850006  339.000000  334.690002  338.109985  338.109985  20265300
1252  2023-11-01  339.790009  347.420013  339.649994  346.070007  346.070007  28158800
1253  2023-11-02  347.239990  348.829987  344.769989  348.320007  348.320007  24348100
1254  2023-11-03  349.630005  354.390015  347.329987  352.799988  352.799988  23624000
1255  2023-11-06  353.450012  357.540009  353.350006  356.529999  356.529999  23811600

[251 rows x 7 columns]>
train/val samples  (1016, 120, 5) (1016, 120) (11, 120, 5) (11, 120)
Metal device set to: Apple M1 Pro

systemMemory: 16.00 GB
maxCacheSize: 5.33 GB

Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 120, 5)]          0         
                                                                 
 LSTMLayer1 (Bidirectional)  (None, 120, 20)           1280      
                                                                 
 LSTMLayer2 (Dropout)        (None, 120, 20)           0         
                                                                 
 LSTMLayer3 (Bidirectional)  (None, 120, 20)           2480      
                                                                 
 LSTMLayer4 (Dropout)        (None, 120, 20)           0         
                                                                 
 LSTMLayer5 (Bidirectional)  (None, 120, 20)           2480      
                                                                 
 LSTMLayer6 (Dropout)        (None, 120, 20)           0         
                                                                 
 LSTMLayer7 (Bidirectional)  (None, 20)                2480      
                                                                 
 LSTMLayer8 (Dropout)        (None, 20)                0         
                                                                 
 dense (Dense)               (None, 240)               5040      
                                                                 
 dist (DistributionLambda)   ((None, 120),             0         
                              (None, 120))                       
                                                                 
=================================================================
Total params: 13,760
Trainable params: 13,760
Non-trainable params: 0
_________________________________________________________________
