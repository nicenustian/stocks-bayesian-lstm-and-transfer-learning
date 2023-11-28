import pandas as pd
from datetime import timedelta



def plot_data(ax, i, output_dir, ticker, validation_date, output_time_steps, df, 
              mean, upper_1sigma, lower_1sigma):
                  
        # Convert the last date in the DataFrame to datetime
        last_date = pd.to_datetime(df['Date'].iloc[-1])
                
        # Create a date range for the next output step days and add it to the DataFrame
        next_120_days = pd.date_range(last_date + timedelta(days=1), 
                                          periods=output_time_steps, freq='D')
        
        df = pd.concat((df, pd.DataFrame({'Date': next_120_days, 'Value': 0})), 
                           ignore_index=True)
        
        validation_date_datetime = pd.to_datetime(validation_date)
        
        #test_date_datetime = last_date - timedelta(days=120)
        
        df_train_plot = df[df['Date']<validation_date_datetime]
        
        df_validate_plot = df[(df['Date'] >= validation_date_datetime) #& 
                              #(df['Date'] < test_date_datetime)
                              ]

        
        #df_test_plot = df[(df['Date']>=test_date_datetime)]


        ##ax.set_title('Daily Closing Price for '+ticker)

        ax.step(df_train_plot['Date'], df_train_plot['Close'], where='pre', 
                      color='black', alpha=1, label='Training')

        ax.step(df_validate_plot['Date'], df_validate_plot['Close'], where='pre', 
                      color='red', alpha=1, label='Validation')

        # ax.step(df_test_plot['Date'], df_test_plot['Close'], where='pre', 
        #               color='red', alpha=1, label='Test')
        
        ax.step(df['Date'][:i], 
                mean[:i], color='orange', 
                linestyle='-', 
                  alpha=.6, label='Predictions')
        ax.fill_between(df['Date'][:i],
                        upper_1sigma[:i], 
                        y2=lower_1sigma[:i], 
                                  color='orange', alpha=.2)

        ax.step(df['Date'][i:], 
                        mean[i:], color='green', 
                        linestyle='-', 
                          alpha=.6, label='Forecast')
        ax.fill_between(df['Date'][i:], 
                                upper_1sigma[i:], 
                                y2=lower_1sigma[i:], 
                                          color='green', alpha=.2)
     
        ax.text(0.85, 0.1,  ticker, fontsize=32, transform = ax.transAxes, 
                           color='black')
