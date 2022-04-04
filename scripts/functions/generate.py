import tensorflow as tf
import tqdm

def generate_text(model, start_string, generation_length, char2idx, idx2char):
    # Evaluation step (generating ABC text using the learned RNN model)

    # convert the start string to numbers (vectorize)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Here batch size == 1
    model.reset_states()
    #tqdm._instances.clear()

    for i in range(int(generation_length)):
        predictions = model(input_eval)
        # Remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        # multinomial distribution to sample
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        # Pass the prediction along with the previous hidden state
        #   as the next inputs to the model
        input_eval = tf.expand_dims([predicted_id], 0)
        # add the predicted character to the generated text!
        # Hint: consider what format the prediction is in vs. the output
        text_generated.append(idx2char[predicted_id])
    
    return (start_string + ''.join(text_generated))