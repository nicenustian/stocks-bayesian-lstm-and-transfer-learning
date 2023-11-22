from make_dataset import make_dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plot_data import plot_data

font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)


def plot_predictions(output_dir, input_time_steps, output_time_steps, 
              tickers, validation_date, columns):
 
    fig, ax = plt.subplots(len(tickers), 1, figsize=(20, 4*len(tickers)))
    fig.subplots_adjust(wspace=0., hspace=0.2)
    
    for ti, ticker in enumerate(tickers):
    
        xx, df, _,_,_,_,_ = make_dataset(output_dir, validation_date, ticker, 
                                             columns, input_time_steps, output_time_steps)

        pred_file_name = output_dir+ticker+'_pred.npy'
    
        with open(pred_file_name, 'rb') as f:
                mean = np.load(f)
                upper_1sigma = np.load(f)
                lower_1sigma = np.load(f)

        plot_data(ax[ti], output_dir, ticker, validation_date, output_time_steps, df, 
              mean, upper_1sigma, lower_1sigma)

    fig.savefig(output_dir+'pred'+'.jpg', format='jpg', dpi=300, bbox_inches = 'tight')
