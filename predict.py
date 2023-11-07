
from LSTMNet import LSTMNet
import numpy as np
import tensorflow as tf


def predict(output_dir, ticker, num_of_layers, lstm_units, input_time_steps, 
            output_time_steps, xx, scaler, features):
    
        
    @tf.function
    def nll_predict(x):       
        dist = model(x, training=False)
        return dist.mean(), dist.stddev()
    
    
    pred_file_name = output_dir+ticker+'_pred.npy'
    model_file_name = output_dir+'models/'+ticker+'_model'
    model = LSTMNet(num_of_layers, lstm_units, output_time_steps)

    ##############################################################################
    
    print(model_file_name)
    model.build(input_shape=(None, input_time_steps, features))

    model.load_weights(model_file_name)
    xx_scaled = scaler.transform(xx.reshape((-1, features))).reshape((1, -1, features))
    
    xx_scaled_segments = np.int32(xx_scaled.shape[1]/input_time_steps)
    
    xx_scaled = xx_scaled[:,-xx_scaled_segments*input_time_steps:,:]
    xx_scaled = xx_scaled.reshape((-1, input_time_steps, features))
    
    print(xx_scaled.shape)
 
    ###########################################################################
    print()
    print('predicting..')
    
    mean = np.full(xx.shape[1]+output_time_steps, np.nan)
    std = np.full(xx.shape[1]+output_time_steps, np.nan)
            
    mean_out, std_out = nll_predict(tf.convert_to_tensor(xx_scaled))
    
    mean[-xx_scaled_segments*output_time_steps:] = tf.reshape(mean_out, [-1])
    std[-xx_scaled_segments*output_time_steps:] = tf.reshape(std_out, [-1])
    
    ###########################################################################

    upper_1sigma= mean + std
    lower_1sigma = mean - std
    
    mean = mean * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    upper_1sigma = upper_1sigma * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    lower_1sigma = lower_1sigma * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    
    
    print(pred_file_name)
    
    with open(pred_file_name, 'wb') as f:
            np.save(f, mean)
            np.save(f, upper_1sigma)
            np.save(f, lower_1sigma)
