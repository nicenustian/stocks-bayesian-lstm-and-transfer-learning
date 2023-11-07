import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
from keras.layers import Dense, Bidirectional, Dropout, LSTM
from tensorflow.keras import Input, Model
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
tfkl = tf.keras.layers


class LSTMNet(keras.Model):

    def __init__(self, num_of_layers=4, lstm_units=120, output_time_steps=120, 
                 seed=12345, name="LSTM_Net", **kwargs):
        super(LSTMNet, self).__init__(name=name, **kwargs)

        self.lstmnet_layers = []
        self.lstm_units = lstm_units

        self.num_of_layers = num_of_layers
        self.output_time_steps = output_time_steps
        
        for li in range(self.num_of_layers):
            
            if li==num_of_layers-1:
                return_seq=False
            else:
                return_seq=True
                
            self.lstmnet_layers.append(
                Bidirectional(LSTM(units=self.lstm_units, 
                                   return_sequences=return_seq, name = 'lstm'+str(li+1))))
                
            self.lstmnet_layers.append(Dropout(0.2, name = 'drop'+str(li+1)))
        
        self.dense = Dense(units=output_time_steps*2)
        self.prob = tfpl.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :self.output_time_steps],
                scale=1e-5 + tf.math.softplus(t[...,self.output_time_steps:])), name = 'dist')
    
    def summary(self, pixels, features):
        x = Input(shape=(pixels, features))
        model = Model(inputs=x, outputs=self.call(x))
        return model.summary()

        

    def call(self, inputs):
        x = inputs
        for li, layer in enumerate(self.lstmnet_layers.layers):
            layer._name = 'LSTMLayer'+str(li+1)
            x = layer(x)
        
        x = self.prob(self.dense(x))
        
        return x
    
