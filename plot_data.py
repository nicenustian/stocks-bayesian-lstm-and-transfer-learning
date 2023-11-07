from make_dataset import make_dataset
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from datetime import timedelta


font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)


def plot_data(output_dir, input_time_steps, output_time_steps, 
              validation_date, tickers, columns):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(wspace=0., hspace=0)
     
    fig2, ax2 = plt.subplots(len(tickers), 1, figsize=(20, 4*len(tickers)))
    fig2.subplots_adjust(wspace=0., hspace=0.2)
    
    for ti, ticker in enumerate(tickers):
    
        xx, df, _,_,_,_,_ = make_dataset(output_dir, validation_date, ticker, 
                                             columns, input_time_steps, output_time_steps)
            
        # Load the history from the JSON file
        history_file_name = output_dir+ticker+'_hist.json'
        
        with open(history_file_name, 'r') as file:
                history = json.load(file)
        
        pred_file_name = output_dir+ticker+'_pred.npy'
    
        with open(pred_file_name, 'rb') as f:
                mean = np.load(f)
                upper_1sigma = np.load(f)
                lower_1sigma = np.load(f)
                        
                
        # Plot training loss
        ax[0].plot(history['loss'], label=ticker)
        # Plot validation loss
        ax[1].plot(history['val_loss'])   
        
        
        ax2[ti].step(df['Date'], df['Close'], where='pre', color='black',  alpha=.6)
            
        # Convert the last date in the DataFrame to datetime
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        
        # Create a date range for the next 120 days and add it to the DataFrame
        next_120_days = pd.date_range(last_date + timedelta(days=1), 
                                          periods=output_time_steps, freq='D')
        df = pd.concat((df, pd.DataFrame({'Date': next_120_days, 'Value': 0})), 
                           ignore_index=True)
                
        ax2[ti].step(df['Date'], mean, color='red', linestyle='-', alpha=.6)
        ax2[ti].fill_between(df['Date'], upper_1sigma, y2=lower_1sigma, 
                            color='red', alpha=.2)
            
        ax2[ti].text(0.85, 0.1,  ticker, fontsize=32, transform = ax2[ti].transAxes, 
                           color='black')
    
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Vali. Loss')
    ax[0].set_xticks([])
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper center', fontsize=12, handlelength=.5, 
                 bbox_to_anchor=(0.5, 1.3), ncol=8, fancybox=True)
    
    fig.savefig(output_dir+'loss'+'.pdf', format='pdf', dpi=90, bbox_inches = 'tight')
    fig2.savefig(output_dir+'pred'+'.pdf', format='pdf', dpi=90, bbox_inches = 'tight')

    
    
