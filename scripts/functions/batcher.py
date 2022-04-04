import numpy as np


def get_batch(vectorized_songs, seq_length, batch_size):
  # the length of the vectorized songs string
  n = vectorized_songs.shape[0] - 1
  # randomly choose the starting indices for the examples in the training batch
  idx = np.random.choice(n-seq_length, batch_size)

  # list of input sequences for the training batch
  input_batch = [vectorized_songs[i:i+seq_length] for i in idx]
  # list of output sequences for the training batch
  output_batch = [vectorized_songs[i+1: i+1+seq_length] for i in idx]

  # x_batch, y_batch provide the true inputs and targets for network training
  x_batch = np.reshape(input_batch, [batch_size, seq_length])
  y_batch = np.reshape(output_batch, [batch_size, seq_length])
  return x_batch, y_batch

def test_batch_func_types(func, args):
    ret = func(*args)
    assert len(ret) == 2, "[FAIL] get_batch must return two arguments (input and label)"
    assert type(ret[0]) == np.ndarray, "[FAIL] test_batch_func_types: x is not np.array"
    assert type(ret[1]) == np.ndarray, "[FAIL] test_batch_func_types: y is not np.array"
    print("[PASS] test_batch_func_types")
    return True

def test_batch_func_shapes(func, args):
    dataset, seq_length, batch_size = args
    x, y = func(*args)
    correct = (batch_size, seq_length)
    assert x.shape == correct, "[FAIL] test_batch_func_shapes: x {} is not correct shape {}".format(x.shape, correct)
    assert y.shape == correct, "[FAIL] test_batch_func_shapes: y {} is not correct shape {}".format(y.shape, correct)
    print("[PASS] test_batch_func_shapes")
    return True

def test_batch_func_next_step(func, args):
    x, y = func(*args)
    assert (x[:,1:] == y[:,:-1]).all(), "[FAIL] test_batch_func_next_step: x_{t} must equal y_{t-1} for all t"
    print("[PASS] test_batch_func_next_step")
    return True

def letsTest(vectorized_songs, seq_length, batch_size, idx2char):
    # Perform some simple tests to make sure your batch function is working properly!
    print('\nPerforming some simple tests to make sure the batch function is working properly') 
    test_args = (vectorized_songs, seq_length, batch_size)
    if not test_batch_func_types(get_batch, test_args) or \
       not test_batch_func_shapes(get_batch, test_args) or \
       not test_batch_func_next_step(get_batch, test_args): 
        print("======\n[FAIL] could not pass tests")
    else: 
        print("======\n[PASS] passed all tests!")

    # Example of how it works
    print('\nExample of how it works:\n')
    x_batch, y_batch = get_batch(vectorized_songs, seq_length, batch_size)

    for i, (input_idx, target_idx) in enumerate(zip(np.squeeze(x_batch), np.squeeze(y_batch))):
        print("Step {:3d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))