import os
from LSTMNet import LSTMNet
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow import keras
import json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_probability as tfp
import keras.backend as K
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers


def train(output_dir, ti, combined_model, ticker, xx, X_train, y_train, X_val, y_val, scaler, epochs,
                  batch_size=32, num_of_layers=4, lstm_units=120, patience_epochs=10, lr=1e-4):

    features = X_train.shape[2]
    output_units = y_train.shape[1]
    
    combined_model_file_name = output_dir+'models/'+'combined_model/'
    
    if combined_model:
        model_file_name = output_dir+'models/'+'combined_model/'
    else:
        model_file_name = output_dir+'models/'+ticker+'_model/'
    
    history_file_name = output_dir+ticker+'_hist.json'
    
    
    # Define a callback to save the model when validation loss improves
    checkpoint_callback = ModelCheckpoint(
        model_file_name,  # Filepath to save the model
        monitor='val_loss',  # Metric to monitor (validation loss)
        save_best_only=True,  # Save only if the monitored metric improves
        mode='min',  # Mode for improvement (minimize validation loss)
        verbose=1,  # Verbosity (1: display messages, 0: no messages)
        save_weights_only=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=patience_epochs,         # Stop after 10 consecutive epochs without improvement
        restore_best_weights=True  # Restore the best model weights when stopping
    )

    
    @tf.function
    def nll(y_true, y_pred):
            nll = -y_pred.log_prob(y_true)
            nll = K.mean(nll)
            return nll
    
        
    model = LSTMNet(num_of_layers, lstm_units, output_units)
    model.summary(X_train.shape[1], features)
    
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=nll)    
    model.build(input_shape=(None, X_train.shape[1], features))
    
    # check if combined model exists load
    if os.path.exists(combined_model_file_name):
        print()
        print('Loading weights from previous model..', combined_model_file_name)
        print()
        model.load_weights(combined_model_file_name)
        

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
              callbacks=[checkpoint_callback, early_stopping_callback], 
              epochs=epochs, batch_size=batch_size, shuffle=True)
    
    
    with open(history_file_name, 'w') as file:
        json.dump(history.history, file)
    
