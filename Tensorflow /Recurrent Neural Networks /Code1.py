import tensorflow as tf 

class RNN(tf.keras.layers.Layer):
    def __init__(self):
        super(RNN, self).__init__()
        
    def call(self, input_data, hidden_state):
        return prediction, hidden_state


first = RNN()
hidden_state = [0, 0, 0, 0]
sentence = ["I", "love", "Recurrent", "neural"]

for s in sentence:
    prediction, hidden_state = first(s, hidden_state)

output = prediction
print(output)

