from make_dataset import make_dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_data import plot_data

font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)


def plot_animations(output_dir, input_time_steps, output_time_steps, animation_output_steps,
              tickers, validation_date, columns):
    
    

    df_list = []
    mean_sequence_list = []
    upper_1sigma_sequence_list = []
    lower_1sigma_sequence_list = []
    
    length_of_sequence = 0
    
    for ti, ticker in enumerate(tickers):
    
        _, df, _,_,_,_,_ = make_dataset(output_dir, validation_date, ticker, 
                                             columns, input_time_steps, 
                                             output_time_steps)

        pred_sequence_file_name = output_dir+ticker+'_pred_sequence.npy'
        length_of_sequence = len(df['Date'])
        
    
        with open(pred_sequence_file_name, 'rb') as f:
                mean_sequence = np.load(f)
                upper_1sigma_sequence = np.load(f)
                lower_1sigma_sequence = np.load(f)
        
        df_list.append(df)
        mean_sequence_list.append(mean_sequence)
        upper_1sigma_sequence_list.append(upper_1sigma_sequence)
        lower_1sigma_sequence_list.append(lower_1sigma_sequence)
    
    shape_for_predictions = (len(tickers), length_of_sequence + output_time_steps)
    mean = np.full(shape_for_predictions, np.nan)
    upper_1sigma = np.full(shape_for_predictions, np.nan)
    lower_1sigma = np.full(shape_for_predictions, np.nan)
    
    
    # the last index at which the input sequence exist for each prediction
    for i in range(input_time_steps, length_of_sequence):
        
        
        # plot the prediction sequence to a plot all stocks togethers
        for ti, ticker in enumerate(tickers):
                     
            mean[ti][i:i+output_time_steps] = mean_sequence_list[ti][i-input_time_steps]
            upper_1sigma[ti][i:i+output_time_steps] = upper_1sigma_sequence_list[ti][i-input_time_steps]
            lower_1sigma[ti][i:i+output_time_steps] = lower_1sigma_sequence_list[ti][i-input_time_steps]
            
                            
        if np.mod(np.int32(i-input_time_steps), animation_output_steps)==0:
            
                                
            fig, ax = plt.subplots(len(tickers), 1, figsize=(24, 4*len(tickers)))
            fig.subplots_adjust(wspace=0., hspace=0.)
                
    
            for ti, ticker in enumerate(tickers):

                for pi in range(len(ax)):
                    ax[pi].tick_params(which='both',direction="in", width=1.5)
                    ax[pi].tick_params(which='major',length=14, top=True, 
                                       left=True, right=True)
                    ax[pi].tick_params(which='minor',length=10, top=True, 
                                       left=True, right=True)
                    ax[pi].minorticks_on()
                
            
                if len(tickers)==1:
                    plot_data(ax, i, output_dir, ticker, validation_date, 
                                  output_time_steps, df_list[ti], mean[ti], 
                                  upper_1sigma[ti], lower_1sigma[ti])
                else:
                    plot_data(ax[ti], i, output_dir, ticker, validation_date, 
                                  output_time_steps, df_list[ti], mean[ti], 
                                  upper_1sigma[ti], lower_1sigma[ti])
                    
                
                    
                if len(tickers)==1:
                    ax.legend(frameon=False, loc='upper left', handlelength=1, 
                              prop={'size': 20})
                else:
                    ax[0].legend(frameon=False, loc='upper left', handlelength=1, 
                                 prop={'size': 20})
                            
            
            fig.savefig(output_dir+'pred'+str(i)+'.png', 
                        format='png', dpi=90, bbox_inches = 'tight')
            plt.close(fig)
        
            

    
        
    
