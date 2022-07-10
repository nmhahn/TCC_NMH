import tensorflow as tf
import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay


def LSTM(rnn_units): 
    return tf.keras.layers.LSTM(
        rnn_units, 
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        recurrent_activation='relu',
        stateful=True,
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
    # Layer 1: Embedding layer to transform indices into dense vectors 
    #          of a fixed embedding size
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),

    # Layer 2: LSTM with `rnn_units` number of units. 
    LSTM(rnn_units),

    # Layer 3: Dense (fully-connected) layer that transforms the LSTM output
    #          into the vocabulary size. 
    tf.keras.layers.Dense(units=vocab_size)
    ])

    return model

# define the loss function to compute and return the loss between the true labels and predictions (logits). Set the argument from_logits=True.
def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss 

# print-out the model's progress through training
class PeriodicPlotter:
    def __init__(self, sec, xlabel='', ylabel='', scale=None):
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.sec = sec
        self.scale = scale
        self.tic = time.time()

    def plot(self, data):
        if time.time() - self.tic > self.sec:
            plt.cla()

        if self.scale is None:
            plt.plot(data)
        elif self.scale == 'semilogx':
            plt.semilogx(data)
        elif self.scale == 'semilogy':
            plt.semilogy(data)
        elif self.scale == 'loglog':
            plt.loglog(data)
        else:
            raise ValueError("unrecognized parameter scale {}".format(self.scale))

        plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
        ipythondisplay.clear_output(wait=True)
        ipythondisplay.display(plt.gcf())

        self.tic = time.time()

def train_step(x, y, model, optimizer, device):
    if device == 'cpu':
        device = '/cpu:0'
    elif device == 'gpu':
        device = '/device:GPU:0'
    
    with tf.device(device):
        # Use tf.GradientTape()
        with tf.GradientTape() as tape:
            # feed the current input into the model and generate predictions
            y_hat = model(x)
            # compute the loss!
            loss = compute_loss(y, y_hat)

        # function call for gradient computation. 
        grads = tape.gradient(loss, model.trainable_variables)
  
        # Apply the gradients to the optimizer so it can update the model accordingly
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss