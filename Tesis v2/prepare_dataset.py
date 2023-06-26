import pandas as pd
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import convert_to_tensor, int64


class PrepareDataset:
    def __init__(self, **kwargs):
        self.n_sentences = 10000  # Number of sentences to include in the dataset
        self.train_split = 0.9  # Ratio of the training data split

    # Fit a tokenizer
    def create_tokenizer(self, dataset):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(dataset)

        return tokenizer

    def find_seq_length(self, dataset):
        return max(len(seq.split()) for seq in dataset)

    def find_vocab_size(self, tokenizer, dataset):
        tokenizer.fit_on_texts(dataset)

        return len(tokenizer.word_index) + 1

    def __call__(self, csv_filename, aymara_column, english_column):
        # Load the CSV file
        df = pd.read_csv(csv_filename)

        # Extract the Aymara and English sentences from the columns
        aymara_sentences = df[aymara_column].tolist()
        english_sentences = df[english_column].tolist()

        # Reduce dataset size
        dataset = list(zip(aymara_sentences[:self.n_sentences], english_sentences[:self.n_sentences]))

        # Include start and end of string tokens
        dataset = [("<START> " + aym + " <EOS>", "<START> " + eng + " <EOS>") for aym, eng in dataset]

        # Random shuffle the dataset
        shuffle(dataset)

        # Split the dataset
        train = dataset[:int(self.n_sentences * self.train_split)]

        # Prepare tokenizer for the encoder input
        enc_tokenizer = self.create_tokenizer([pair[0] for pair in train])
        enc_seq_length = self.find_seq_length([pair[0] for pair in train])
        enc_vocab_size = self.find_vocab_size(enc_tokenizer, [pair[0] for pair in train])

        # Encode and pad the input sequences
        trainX = enc_tokenizer.texts_to_sequences([pair[0] for pair in train])
        trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
        trainX = convert_to_tensor(trainX, dtype=int64)

        # Prepare tokenizer for the decoder input
        dec_tokenizer = self.create_tokenizer([pair[1] for pair in train])
        dec_seq_length = self.find_seq_length([pair[1] for pair in train])
        dec_vocab_size = self.find_vocab_size(dec_tokenizer, [pair[1] for pair in train])

        # Encode and pad the input sequences
        trainY = dec_tokenizer.texts_to_sequences([pair[1] for pair in train])
        trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
        trainY = convert_to_tensor(trainY, dtype=int64)

        return trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size



################################# TEST ##################################
# Prepare the training data
dataset = PrepareDataset()
trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset('Dataset.csv', 'aymara', 'english')

print(train_orig[0][0], '\n', trainX[0])

print('Encoder sequence length:', enc_seq_length)

print(train_orig[0][1], '\n', trainY[0])

print('Decoder sequence length:', dec_seq_length)
