import tensorflow as tf 
weights = tf.variable([tf.random.normal])
def compute_loss(wrights) :
    ## MSE computaiton algo 
    loss = tf.reduce_mean(tf.square(tf.subtract(y,weights)))
    return loss 

def main() :
    while True  :
        with tf.GradientTape() as tape :
            loss = compute_loss(weights)
            gradient  = tape.gradient(loss, weights)
        weights  = weights - lr*gradient
        return weights
main()
