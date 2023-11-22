import pandas as pd
from datetime import timedelta



def plot_data(ax, output_dir, ticker, validation_date, output_time_steps, df, 
              mean, upper_1sigma, lower_1sigma):
                  
        # Convert the last date in the DataFrame to datetime
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        
        # Create a date range for the next 120 days and add it to the DataFrame
        next_120_days = pd.date_range(last_date + timedelta(days=1), 
                                          periods=output_time_steps, freq='D')
        df = pd.concat((df, pd.DataFrame({'Date': next_120_days, 'Value': 0})), 
                           ignore_index=True)
        validation_date_datetime = pd.to_datetime(validation_date) 

        df_train_plot = df[df['Date']<validation_date_datetime]
        df_validate_plot = df[df['Date']>=validation_date_datetime]
        

        ax.step(df_train_plot['Date'], df_train_plot['Close'], where='pre', 
                      color='black', alpha=1)

        ax.step(df_validate_plot['Date'], df_validate_plot['Close'], where='pre', 
                      color='red', alpha=1)
            
        ax.step(df_train_plot['Date'], mean[:len(df_train_plot)], 
                     color='black', linestyle='-', alpha=.6)
        ax.fill_between(df_train_plot['Date'], upper_1sigma[:len(df_train_plot)], 
                             y2=lower_1sigma[:len(df_train_plot)], 
                                 color='grey', alpha=.2)
        
        ax.step(df_validate_plot['Date'], mean[len(df_train_plot):], 
                     color='red', linestyle='-', alpha=.6)
        ax.fill_between(df_validate_plot['Date'], upper_1sigma[len(df_train_plot):], 
                             y2=lower_1sigma[len(df_train_plot):], 
                                 color='red', alpha=.2)        
        
        ax.text(0.85, 0.1,  ticker, fontsize=32, transform = ax.transAxes, 
                               color='black')
