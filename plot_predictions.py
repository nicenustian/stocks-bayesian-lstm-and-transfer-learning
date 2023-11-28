from make_dataset import make_dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_data import plot_data

font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)


def plot_predictions(output_dir, input_time_steps, output_time_steps, 
              tickers, validation_date, columns):
    
 
    fig, ax = plt.subplots(len(tickers), 1, figsize=(24, 4*len(tickers)))
    fig.subplots_adjust(wspace=0., hspace=0.)
    
    for pi in range(len(ax)):
        ax[pi].tick_params(which='both',direction="in", width=1.5)
        ax[pi].tick_params(which='major',length=14, top=True, left=True, right=True)
        ax[pi].tick_params(which='minor',length=10, top=True, left=True, right=True)
        ax[pi].minorticks_on()
    
    for ti, ticker in enumerate(tickers):
    
        _, df, _,_,_,_,_ = make_dataset(output_dir, validation_date, ticker, 
                                             columns, input_time_steps, output_time_steps)

        pred_file_name = output_dir+ticker+'_pred.npy'
        
        length_of_sequence = len(df['Date'])
    
        with open(pred_file_name, 'rb') as f:
                mean = np.load(f)
                upper_1sigma = np.load(f)
                lower_1sigma = np.load(f)

        if len(tickers)==1:
            plot_data(ax, length_of_sequence, output_dir, ticker, validation_date, 
                      output_time_steps, df, 
                  mean, upper_1sigma, lower_1sigma)
        else:
            plot_data(ax[ti], length_of_sequence, output_dir, ticker, 
                      validation_date, output_time_steps, df, 
              mean, upper_1sigma, lower_1sigma)

    if len(tickers)==1:
        ax.legend(frameon=False, loc='upper left', 
              handlelength=1, prop={'size': 20})
    else:
        ax[0].legend(frameon=False, loc='upper left', 
              handlelength=1, prop={'size': 20})

    fig.savefig(output_dir+'pred'+'.jpg', format='jpg', dpi=300, bbox_inches = 'tight')

    
    
