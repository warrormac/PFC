import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.metrics import Mean
from tensorflow import data, train, math, reduce_sum, cast, equal, argmax, float32, GradientTape, TensorSpec, function, int64
from keras.losses import sparse_categorical_crossentropy
from model import TransformerModel
from prepare_dataset import PrepareDataset
from time import time
from keras.preprocessing.text import tokenizer_from_json

# Define the model parameters
h = 8  # Number of self-attention heads
d_k = 64  # Dimensionality of the linearly projected queries and keys
d_v = 64  # Dimensionality of the linearly projected values
d_model = 512  # Dimensionality of model layers' outputs
d_ff = 2048  # Dimensionality of the inner fully connected layer
n = 6  # Number of layers in the encoder stack

# Define the training parameters
epochs = 50
batch_size = 64
beta_1 = 0.9
beta_2 = 0.98
epsilon = 1e-9
dropout_rate = 0.1

# Implementing a learning rate scheduler
class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        step_num = cast(step_num, float32)
        warmup_steps = cast(self.warmup_steps, float32)
        arg1 = step_num ** -0.5
        arg2 = step_num * (warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)

# Instantiate an Adam optimizer
optimizer = Adam(LRScheduler(d_model, warmup_steps=4000), beta_1, beta_2, epsilon)

# Prepare the training data
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('Dataset.csv', 'english', 'aymara' )

# Prepare the dataset batches
train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
train_dataset = train_dataset.batch(batch_size)


# Convert sequence to text
def convert_sequence_to_text(sequence, tokenizer):
    texts = tokenizer.sequences_to_texts([sequence.numpy().tolist()])
    text = texts[0] if len(texts) > 0 else ""
    return text


# Create model
training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v, d_model, d_ff, n, dropout_rate)

# Defining the loss function
def loss_fcn(target, prediction):
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask
    return reduce_sum(loss) / reduce_sum(padding_mask)

# Defining the accuracy function
def accuracy_fcn(target, prediction):
    padding_mask = math.logical_not(equal(target, 0))
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Include metrics monitoring
train_loss = Mean(name='train_loss')
train_accuracy = Mean(name='train_accuracy')

# Create a checkpoint object and manager to manage multiple checkpoints
ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        prediction = training_model(encoder_input, decoder_input, training=True)
        loss = loss_fcn(decoder_output, prediction)
        accuracy = accuracy_fcn(decoder_output, prediction)
    gradients = tape.gradient(loss, training_model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))
    train_loss(loss)
    train_accuracy(accuracy)

# Training loop
start_time = time()
training_outputs = []  # List to store training predictions

for epoch in range(epochs):
    train_loss.reset_states()
    train_accuracy.reset_states()

    print("\nStart of epoch %d" % (epoch + 1))
    print_time = time()

    for step, (train_batchX, train_batchY) in enumerate(train_dataset):
        if step >= 100:
            break

        encoder_input = train_batchX[:, 1:]
        decoder_input = train_batchY[:, :-1]
        decoder_output = train_batchY[:, 1:]

        train_step(encoder_input, decoder_input, decoder_output)

        print(f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        prediction = training_model(encoder_input, decoder_input, training=False)
        training_outputs.append(prediction.numpy())  # Save the generated predictions during training

    print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (epoch + 1, train_loss.result(), train_accuracy.result()))
    print("Time taken for epoch %d: %.2fs" % (epoch + 1, time() - print_time))

    if (epoch + 1) % 5 == 0:
        save_path = ckpt_manager.save()
        print("Saved checkpoint at epoch %d" % (epoch + 1))

print("Total time taken: %.2fs" % (time() - start_time))

# Reshape predictions for saving as CSV
training_outputs_reshaped = np.concatenate(training_outputs, axis=0)

# Save predictions as a CSV file
with open('training_outputs.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(training_outputs_reshaped)

# Set the tokenizer for the decoder
enc_tokenizer = dataset.enc_tokenizer
dec_tokenizer = dataset.dec_tokenizer

for i in range(len(trainX)):
    # Convert input sequence to text
    input_text = convert_sequence_to_text(trainX[i], enc_tokenizer)
    
    # Convert target sequence to text
    target_text = convert_sequence_to_text(trainY[i], dec_tokenizer)
    
    print("Input:", input_text)
    print("Translation:", target_text)
    print()


# Open a file for writing
with open('output_translations.txt', 'w') as f:
    # Iterate over the sequences and their translations
    for i in range(len(trainX)):
        # Convert input sequence to text
        input_text = convert_sequence_to_text(trainX[i], dataset.enc_tokenizer)
        
        # Convert target sequence to text
        target_text = convert_sequence_to_text(trainY[i], dataset.dec_tokenizer)
        
        # Write the input and translation to the file
        f.write(f"Input: {input_text}\n")
        f.write(f"Translation: {target_text}\n\n")
