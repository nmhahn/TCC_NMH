# libraries --------------------------------------------------------------------------------------------------
import regex as re
import numpy as np
import collections


# functions --------------------------------------------------------------------------------------------------
def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    return songs


def extract_vocab(text):
    vocab = sorted(set(text))
    return {u:i for i, u in enumerate(vocab)}


def char2idx(string, vocab=None):
    if vocab==None:
        vocab = extract_vocab(string)
    vectorized_list = np.array([vocab[s] for s in string])
    return vectorized_list


def idx2char(idx, vocab):
    keys = list(vocab.keys())
    string = ''
    if isinstance(idx, collections.Iterable):
        for i in idx:
            string += keys[i]
    else:
        string += keys[idx]
    return string


def get_batch(vectorized_songs, seq_length, batch_size, seed=None):
    n = vectorized_songs.shape[0] - 1
    np.random.seed(seed)
    idx = np.random.choice(n-seq_length, batch_size)
    input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
    output_batch = [vectorized_songs[i+1: i+1+seq_length] for i in idx]
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch