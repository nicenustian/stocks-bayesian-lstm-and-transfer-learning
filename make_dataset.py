import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def make_dataset(output_dir, validation_date, ticker, columns, previous_timesteps=120, output_timesteps=120):

    data_file_name = output_dir+ticker+".csv"

    X_train = []
    y_train = []
    X_val = []
    y_val = []
    data_train = []
    data_val = []
    xx = []
    
    print(data_file_name)
    df = pd.read_csv(data_file_name)  # Read the CSV file
    ##print('dataframe', df.info)
    
    xx.append(df[columns].values)
    
    # Select data for the last year
    df_val = df[(df['Date'] >= validation_date)]
    
    # Select training samples
    df_train = df[ (df['Date'] < validation_date) ]
    print('train dataset',df_train.info)
    print('test dataset', df_val.info)
    
    df_train = df_train.set_index('Date')
    data_train.append(df[columns].values)
    data_val.append(df_val[columns].values)
    
    xx = np.array(xx)

    data_train = np.array(data_train)
    data_val = np.array(data_val)

    features = len(columns)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_train = data_train.reshape((-1, features))
    scaler = scaler.fit(data_train)
    data_train = scaler.transform(data_train).reshape((1, -1, features))
    data_val = scaler.transform(data_val.reshape(( -1, features))).reshape((1, -1, features))
    
    
    for si in range(data_train.shape[0]):
        for i in range(previous_timesteps, data_train.shape[1]-output_timesteps):
            X_train.append(data_train[si, i-previous_timesteps:i])
            y_train.append(data_train[si, i:i+output_timesteps, 0])
    
    for si in range(data_val.shape[0]):        
        for i in range(previous_timesteps, data_val.shape[1]-output_timesteps):
            X_val.append(data_val[si, i-previous_timesteps:i])
            y_val.append(data_val[si, i:i+output_timesteps, 0])
    
    ###############################################################################
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print('train/val samples ',X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    
    df  = df[['Close', 'Date']]
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime (optional but recommended)
    
    return xx, df, scaler, X_train, y_train, X_val, y_val
