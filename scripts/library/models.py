# libraries --------------------------------------------------------------------------------------------------
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from tqdm import tqdm

from library.reader import *

# functions --------------------------------------------------------------------------------------------------
def build_model_LSTM(vocab_size, embedding_dim, rnn_units, batch_size, seed=None,
                     rnn_init='glorot_uniform', rnn_activation='sigmoid'):
    tf.random.set_seed(seed)
    model = tf.keras.Sequential([
        # layer 1: inputs
        tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            batch_input_shape=[batch_size, None]
        ),
        # layer 2: LSTM
        tf.keras.layers.LSTM(
            rnn_units, 
            recurrent_initializer=rnn_init,
            recurrent_activation=rnn_activation,
            return_sequences=True,
            stateful=True,
        ),
        # layer 3: dense fully-connected layer that transforms the LSTM output into the vocabulary size
        tf.keras.layers.Dense(
            units=vocab_size
        )
    ])
    return model


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss


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


#@tf.function # I need to study more to use it
def train_step(model, x, y, optimizer):
    m = model
    with tf.GradientTape() as tape:
        y_hat = m(x)
        loss = compute_loss(y, y_hat)

    grads = tape.gradient(loss, m.trainable_variables)
    optimizer.apply_gradients(zip(grads, m.trainable_variables))
    return loss

# 
# def fit_model(model, string, epochs, seq_length, batch_size, learning_rate, trace=True, optimizer='Adam', seed=None):
#     if hasattr(tqdm, '_instances'): tqdm._instances.clear() # clear if it exists
    
#     if trace==True: 
#         history = []
#         plotter = PeriodicPlotter(sec=2, xlabel='Iterations', ylabel='Loss')

#     if optimizer=='Adam': optimizer = tf.keras.optimizers.Adam(learning_rate)

#     m = model
#     for iter in tqdm(range(epochs)):
#         x_batch, y_batch = get_batch(string, seq_length, batch_size, seed=None if seed==None else seed+iter)
#         loss = train_step(m, x_batch, y_batch, optimizer)
#         if trace==True:
#             history.append(loss.numpy().mean()) 
#             plotter.plot(history)
#     return m


def generate_text(model, start_string, vocab, choose, generation_length=None, n_songs=None):
    input_eval = char2idx(start_string, vocab)
    input_eval = tf.expand_dims(input_eval, 0)
    text = start_string
    text_generated = []

    model.reset_states()
    tqdm._instances.clear()

    if choose=='length': 
        n = generation_length
        
        for i in tqdm(range(n)):
            pred = model(input_eval)[0]
            pred_id = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()
            input_eval = tf.expand_dims([pred_id],0)
            text_generated.append(idx2char(pred_id, vocab))
        
        text = (start_string + ''.join(text_generated))
    

    if choose=='songs': 
        n = 0

        with tqdm(total=n_songs):
            while n < n_songs:
                pred = model(input_eval)[0]
                pred_id = tf.random.categorical(pred, num_samples=1)[-1,0].numpy()
                input_eval = tf.expand_dims([pred_id],0)
                text_generated.append(idx2char(pred_id, vocab))
                text = start_string + ''.join(text_generated)
                aux = extract_song_snippet(text)
                n = len(aux)

    return text
