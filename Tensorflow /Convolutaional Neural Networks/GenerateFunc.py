import numpy as np 
import tensorflow as tf 

## Strides use  for the control of the window steps
def generate_model() :
    
    model = tf.keras.models.Sequential([
        ## this is the first convo layer
        tf.keras.layers.Convo2D(32,filter_size = 3 , activation= "relu")
        , tf.keras.layers.MaxPool2D(pool_size=2 , strides = 2) , 
        ## this is the second convo layer
        tf.keras.layers.Convo2D(64 , filter_size =3 , activation ="relu") , 
        tf.keras.layers.MaxPool2D(pool_size = 2 , strides = 2 )  ,
        ##  fully connected layer 
        tf.keras.layers.Flatten() , 
        tf.keras.layers.Dense(1024 , activation = "relu") , 
        ## softmax makes this CNN  architrcture
        tf.keras.layers.Dense(10 , activation = "softmax")
    ])
    
    return model 
model =  generate_model()
model.summary() 
