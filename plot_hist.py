import matplotlib
import matplotlib.pyplot as plt
import json


font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)


def plot_hist(output_dir, input_time_steps, output_time_steps, 
              tickers, columns):
     
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(wspace=0., hspace=0)
    
    for ti, ticker in enumerate(tickers):

        # Load the history from the JSON file
        history_file_name = output_dir+ticker+'_hist.json'
        
        with open(history_file_name, 'r') as file:
                history = json.load(file)

                
        # Plot training loss
        ax[0].plot(history['loss'], label=ticker)
        # Plot validation loss
        ax[1].plot(history['val_loss'])   
        
          
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Vali. Loss')
    ax[0].set_xticks([])
    ax[0].set_ylabel('Loss')
    ax[0].legend(loc='upper center', fontsize=12, handlelength=.5, 
                 bbox_to_anchor=(0.5, 1.3), ncol=8, fancybox=True)
    
    fig.savefig(output_dir+'loss'+'.jpg', format='jpg', dpi=300, bbox_inches = 'tight')
