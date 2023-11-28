import numpy as np
from make_dataset import make_dataset
from train import train
import time
import os
import yfinance as yf
import argparse
from datetime import datetime, timedelta
from plot_hist import plot_hist
from plot_predictions import plot_predictions
from plot_animations import plot_animations
from predict import predict

###############################################################################

def main():
    
    tickers = [
        'AAPL',  # Apple Inc.
        'MSFT',  # Microsoft Corporation
        'AMZN',
        'GOOGL',
    ]
    
    columns = ['Close', 'Volume', 'High', 'Low', 'Open']
    
    offset_today_by_days = 0
    previous_years = 5
    patience_epochs = 10
    animation_output_steps = 10
                
    end_date =  (datetime.today() - timedelta(
        days=offset_today_by_days))
    
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    # take the last five years of data as default with offset provided
    # starting from today
    start_date =  (end_date - timedelta(days=365*previous_years))
    start_date_str = start_date.strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(description=("arguments"))
    parser.add_argument('--tickers', action='store', default=tickers,
                        type=str, nargs='*')
    parser.add_argument("--start_date", default=start_date_str)
    parser.add_argument("--end_date", default=end_date_str)
    # one year of past data as validation
    parser.add_argument("--validation_days", default="365")
    
    parser.add_argument("--epochs", default="1000")
    parser.add_argument("--layers", default="4")
    parser.add_argument("--input_time_steps", default="120")
    parser.add_argument("--output_time_steps", default="30")
    parser.add_argument("--batch_size", default="32")
    parser.add_argument("--lr", default="1e-4")
    parser.add_argument("--download", action="store_true", default=False)
    parser.add_argument("--output_dir", default="tech_stocks_combined/")
    
    args = parser.parse_args()
    
    tickers = args.tickers
    epochs = np.int32(args.epochs)
    batch_size = np.int32(args.batch_size)
    lr = np.float32(args.lr)
    validation_days = int(args.validation_days)

    
    
    batch_size = np.int32(args.batch_size)
    num_of_layers = np.int32(args.layers)
    input_time_steps = np.int32(args.input_time_steps)
    output_time_steps = np.int32(args.output_time_steps)
    output_dir = args.output_dir+'date'+start_date_str+\
    '_inout'+str(input_time_steps)+str(output_time_steps)+"/"


    validation_date = (end_date - timedelta(
        days=validation_days)).strftime("%Y-%m-%d")
    lstm_units = output_time_steps


    print('tickers = ', args.tickers)
    print('dates (start, end, validation)', start_date, end_date, validation_date)
    print('epochs, lr, batch_size layers = ', epochs, lr, batch_size, num_of_layers)
    print('time steps (input, output) = ', input_time_steps, output_time_steps)
    
    
    start_time = time.time()

    # Check if the directory exists, and if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir+'models/')
        print(f"Directory '{output_dir}' created successfully.")
    else:
        print(f"Directory '{output_dir}' already exists.")

    
    if args.download:
        # Download the data first into output directory
        for ticker in tickers:
                    
            # Download stock data with 1-day resolution
            time_series_data = yf.download(ticker, start=start_date, 
                                           end=end_date, interval="1d")
            
            # Print the downloaded data
            print(output_dir+ticker+".csv")
            time_series_data.to_csv(output_dir+ticker+".csv")
            
        
        
    print()
    print()
    print('Train all stocks as one combined model but training in sequentially..')
        
    
    # First run for a combined model for stocks    
    # read each stock and train model and save weights to combine model
    for ti, ticker in enumerate(tickers):
        
        print()
        print('===========================================')

        xx, _, scaler, X_train, y_train, X_val, y_val = make_dataset(output_dir,
                validation_date, ticker, columns, input_time_steps, output_time_steps)
        
        ticker_start_time = time.time()
        
        train(output_dir, ti, True, ticker, xx, X_train, y_train, X_val, y_val, scaler, 
                              epochs, batch_size, num_of_layers, lstm_units,
                              patience_epochs, lr)
        
        ticker_end_time = time.time()
            
        print('ticker ', ticker, (ticker_end_time - ticker_start_time)/60., '[min]')
    
    print()
    print()
    print('Now train each model again using combined model as a starting point..')
    
    
    #read combined model as starting point, train again on each
    #stock save each stock model after training
    for ti, ticker in enumerate(tickers):
        
        print()
        print('==============================================================')

        xx, df, scaler, X_train, y_train, X_val, y_val = make_dataset(output_dir,
                validation_date, ticker, columns, input_time_steps, output_time_steps)
        
        ticker_start_time = time.time()
        
        train(output_dir, ti, False, ticker, xx, X_train, y_train, X_val, y_val, scaler, 
                              epochs, batch_size, num_of_layers, lstm_units,
                              patience_epochs, lr)
        
        ticker_end_time = time.time()
                
        predict(output_dir, ticker, validation_date, num_of_layers, lstm_units, input_time_steps, 
                output_time_steps, df, xx, scaler, len(columns))
        
        print('ticker ', ticker, (ticker_end_time - ticker_start_time)/60., 
              '[min]')

    
    end_time = time.time()
    print('total time [min] = ', (end_time - start_time)/60.)
    
    plot_predictions(output_dir, input_time_steps, output_time_steps,
                    tickers, validation_date, columns)
   
    plot_hist(output_dir, input_time_steps, output_time_steps, 
                  tickers, columns)
        
    plot_animations(output_dir, input_time_steps, output_time_steps, 
                  animation_output_steps, tickers, validation_date, columns)
    
# ###############################################################################


if __name__ == "__main__":
    main()
