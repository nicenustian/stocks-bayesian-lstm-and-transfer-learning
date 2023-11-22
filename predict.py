from LSTMNet import LSTMNet
import numpy as np
import tensorflow as tf
from plot_data import plot_data
import matplotlib
import matplotlib.pyplot as plt
            
font = {'family' : 'serif', 'weight' : 'normal','size' : 28}
matplotlib.rc('font', **font)

def predict(output_dir, ticker, validation_date, num_of_layers, lstm_units, input_time_steps, 
            output_time_steps, df, xx, scaler, features, save_series=False):
        
    @tf.function
    def nll_predict(x):       
        dist = model(x, training=False)
        return dist.mean(), dist.stddev()
    
    
    pred_file_name = output_dir+ticker+'_pred.npy'
    model_file_name = output_dir+'models/'+ticker+'_model/'
    model = LSTMNet(num_of_layers, lstm_units, output_time_steps)

    ##############################################################################
    
    print(model_file_name)
    model.build(input_shape=(None, input_time_steps, features))

    model.load_weights(model_file_name)
    xx_scaled = scaler.transform(xx.reshape((-1, features))).reshape((1, -1, features))
    print('input shape ', xx_scaled.shape)
 
    ###########################################################################
    
    print()
    print('predicting using model file', model_file_name)
    
    mean = np.full(xx.shape[1]+output_time_steps, np.nan)
    upper_1sigma = np.full(xx.shape[1]+output_time_steps, np.nan)
    lower_1sigma = np.full(xx.shape[1]+output_time_steps, np.nan)
    
    for i in range(input_time_steps, xx.shape[1]):
            
        mean_out, std_out = nll_predict(tf.convert_to_tensor(xx_scaled[:,i-input_time_steps:i,:]))
        upper_1sigma_out = mean_out + std_out
        lower_1sigma_out = mean_out - std_out
        
        mean_out = mean_out * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        upper_1sigma_out = upper_1sigma_out * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        lower_1sigma_out = lower_1sigma_out * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        
        mean[i:i+output_time_steps] = tf.reshape(mean_out, [-1])
        upper_1sigma[i:i+output_time_steps] = tf.reshape(upper_1sigma_out, [-1])
        lower_1sigma[i:i+output_time_steps] = tf.reshape(lower_1sigma_out, [-1])

  
        if save_series:
            fig, ax = plt.subplots(1, 1, figsize=(20, 6))
            fig.subplots_adjust(wspace=0., hspace=0.2)
            
            plot_data(ax, output_dir, ticker, validation_date, output_time_steps, df, 
                      mean, upper_1sigma, lower_1sigma)
            
            fig.savefig(output_dir+'pred_'+ticker+'_timestep'+str(i)+'.png', 
                        format='png', dpi=300, bbox_inches = 'tight')
            plt.close(fig)
    
    print('prediction file name ',pred_file_name)
    
    with open(pred_file_name, 'wb') as f:
            np.save(f, mean)
            np.save(f, upper_1sigma)
            np.save(f, lower_1sigma)
