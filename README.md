# Stock market predictions using baysian LSTM networks

# Example Usage
python main.py --tickers 'AMZN' 'MSFT' --epochs 1000 --output_dir new


tickers =  ['AMZN', 'MSFT']
dates (start, end, validation) 2018-11-08 2023-11-07 2022-11-07
epochs, lr, batch_size layers =  1000 1e-04 32 4
time steps (input, output) =  120 120
Directory 'new/' created successfully.
# Downloading the data fron yahoo finanace
[*********************100%%**********************]  1 of 1 completed
new/AMZN.csv
[*********************100%%**********************]  1 of 1 completed
new/MSFT.csv

# information for training and validation samples
Train all stocks as one combined model but training in sequentially..
new/AMZN.csv
train dataset <bound method DataFrame.info of             
| index |Date   |     Open    |    High     |    Low     |  Close  | Adj Close |     Volume|
|0     |2018-11-08|   |87.750000   |89.199997   |86.255501   |87.745499   |87.745499  |130698000|
|1004  |2022-11-04|   |91.489998   |92.440002   |88.040001   |90.980003   |90.980003  |129101300|

[1005 rows x 7 columns]>
test dataset <bound method DataFrame.info of
| index |Date   |     Open    |    High     |    Low     |  Close  | Adj Close |     Volume|
1005  2022-11-07   91.949997   92.099998   89.040001   90.529999   90.529999   77495700
1255  2023-11-06  138.759995  140.729996  138.360001  139.740005  139.740005   44928800

[251 rows x 7 columns]>
train/val samples  (1016, 120, 5) (1016, 120) (11, 120, 5) (11, 120)

# Model summary
Model: "model"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 120, 5)]          0         
                                                                 
 LSTMLayer1 (Bidirectional)  (None, 120, 240)          120960    
                                                                 
 LSTMLayer2 (Dropout)        (None, 120, 240)          0         
                                                                 
 LSTMLayer3 (Bidirectional)  (None, 120, 240)          346560    
                                                                 
 LSTMLayer4 (Dropout)        (None, 120, 240)          0         
                                                                 
 LSTMLayer5 (Bidirectional)  (None, 120, 240)          346560    
                                                                 
 LSTMLayer6 (Dropout)        (None, 120, 240)          0         
                                                                 
 LSTMLayer7 (Bidirectional)  (None, 240)               346560    
                                                                 
 LSTMLayer8 (Dropout)        (None, 240)               0         
                                                                 
 dense (Dense)               (None, 240)               57840     
                                                                 
 dist (DistributionLambda)   ((None, 120),             0         
                              (None, 120))                       
                                                                 
=================================================================
Total params: 1,218,480
Trainable params: 1,218,480
Non-trainable params: 0
_________________________________________________________________

# training for a model with combined weights using all sotcks one by one

Epoch 1/1000
32/32 [==============================] - ETA: 0s - loss: 0.7893  
Epoch 00001: val_loss improved from inf to 0.48532, saving model to new/models/combined_model
32/32 [==============================] - 19s 376ms/step - loss: 0.7893 - val_loss: 0.4853



![losses](loss.jpg)
![predictions](pred.jpg)
