import tensorflow as tf
from tensorflow import convert_to_tensor
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
import numpy as np
import matplotlib.pyplot as plt
import csv

class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                P[k, 2*i] = np.sin(k/denominator)
                P[k, 2*i+1] = np.cos(k/denominator)
        return P

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


output_sequence_length = 5
vocab_size = 10
sentences = [["Jiwasa ni robot"], ["janiña robot"]]
sentence_data = tf.data.Dataset.from_tensor_slices(sentences)

# Create the TextVectorization layer
vectorize_layer = TextVectorization(
    output_sequence_length=output_sequence_length,
    max_tokens=vocab_size
)

# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)

# Convert all sentences to tensors
word_tensors = convert_to_tensor(sentences, dtype=tf.string)

# Use the word tensors to get vectorized phrases
vectorized_words = vectorize_layer(word_tensors)

print("Vocabulary: ", vectorize_layer.get_vocabulary())
print("Vectorized words: ", vectorized_words)

output_length = 6
embedding_layer = PositionEmbeddingFixedWeights(
    sequence_length=output_sequence_length,
    vocab_size=vocab_size,
    output_dim=output_length
)

embedded_output = embedding_layer(vectorized_words)
print("Final output: ", embedded_output)

# # Visualization code
# fig = plt.figure(figsize=(15, 5))
# title = ["Sentence 1", "Sentence 2"]
# for i in range(2):
#     ax = plt.subplot(1, 2, 1 + i)
#     matrix = tf.reshape(embedded_output[i, :, :], (output_sequence_length, output_length))
#     cax = ax.matshow(matrix)
#     plt.gcf().colorbar(cax)
#     plt.title(title[i], y=1.2)
# fig.suptitle("Embedded Output")
# plt.show()

# Create the TextVectorization layer
vectorize_layer = TextVectorization(
    output_sequence_length=output_sequence_length,
    max_tokens=vocab_size
)

# Train the layer to create a dictionary
vectorize_layer.adapt(sentence_data)

# Save the vocabulary as a CSV file
vocabulary = vectorize_layer.get_vocabulary()
with open('vocabulary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(vocabulary)

print("Vocabulary: ", vocabulary)
