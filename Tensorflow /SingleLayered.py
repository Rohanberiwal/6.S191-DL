import tensorflow as tf 
## This is a single layer  Neural network 
## layer init by using the unit funtion that creates more than one precptorns for the NN 

layer =  tf.keras.layers.Dense(unit =2)
class Layer(tf.keras.layers.Layer) :
    def __init__(self , input_dim ,  output_dim)  :
        super(Layer, self).__init__()
        self.W = self.add_weights([input_dim  , output_dim])
        self.b = self.add_weights([1,output_dim])
        
    def call(self, inputs) :
        z = tf.matmul(inputs , self.W) + self.b
        ##Non linear function conversion 
        output =  tf.math.sigmoid(z) 
        return output 
    
    
