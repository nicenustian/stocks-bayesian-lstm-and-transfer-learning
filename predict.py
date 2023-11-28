from LSTMNet import LSTMNet
import numpy as np
import tensorflow as tf

def predict(output_dir, ticker, validation_date, num_of_layers, lstm_units, input_time_steps, 
            output_time_steps, df, xx, scaler, features):
        
    @tf.function
    def nll_predict(x):       
        dist = model(x, training=False)
        return dist.mean(), dist.stddev()
    
    
    pred_file_name = output_dir+ticker+'_pred.npy'
    pred_sequence_file_name = output_dir+ticker+'_pred_sequence.npy'

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
    
    mean = np.full(xx_scaled.shape[1] + output_time_steps, np.nan)
    upper_1sigma = np.full(xx_scaled.shape[1] + output_time_steps, np.nan)
    lower_1sigma = np.full(xx_scaled.shape[1] + output_time_steps, np.nan)

    input_sequence = []
    
    for i in range(input_time_steps, xx_scaled.shape[1]):
        input_sequence.append(xx_scaled[0,i-input_time_steps:i,:])
    
    input_sequence = np.array(input_sequence)

    mean_sequence, std_sequence = nll_predict(tf.convert_to_tensor(input_sequence))
    upper_1sigma_sequence = mean_sequence + std_sequence
    lower_1sigma_sequence = mean_sequence - std_sequence
        
    mean_sequence = mean_sequence * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    upper_1sigma_sequence = upper_1sigma_sequence * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    lower_1sigma_sequence = lower_1sigma_sequence * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]   
    
    for i in range(mean_sequence.shape[0]): 
        mean[i+input_time_steps] = mean_sequence[i,0]
        upper_1sigma[i+input_time_steps] = upper_1sigma_sequence[i,0]
        lower_1sigma[i+input_time_steps] = lower_1sigma_sequence[i,0]
        
    # Save the last sequence left
    mean[-output_time_steps:] = mean_sequence[-1]
    upper_1sigma[-output_time_steps:] = upper_1sigma_sequence[-1]
    lower_1sigma[-output_time_steps:] = lower_1sigma_sequence[-1]
    
    
    print()
    print('prediction file name ', pred_file_name)
    
    with open(pred_file_name, 'wb') as f:
        np.save(f, mean)
        np.save(f, upper_1sigma)
        np.save(f, lower_1sigma)
    
    with open(pred_sequence_file_name, 'wb') as f:
        np.save(f, mean_sequence)
        np.save(f, upper_1sigma_sequence)
        np.save(f, lower_1sigma_sequence)
            
            

    
