import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np

# Set the device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tokenizers for Aymara and English languages
def aymara_tokenizer(sentence):
    return sentence.lower().split()

def english_tokenizer(sentence):
    return sentence.lower().split()

# Define the Fields for Aymara and English sentences
class Field:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.vocab = None

    def build_vocab(self, dataset):
        counter = Counter()
        for sentence in dataset:
            tokens = self.tokenizer(sentence)
            counter.update(tokens)
        self.vocab = {token: i for i, (token, _) in enumerate(counter.most_common())}

    def process(self, sentence):
        tokens = self.tokenizer(sentence)
        return [self.vocab[token] for token in tokens]

# Load the dataset
dataset = [
    {'aymara': 'Phaxsi', 'english': 'Sun'},
    {'aymara': 'Jach\'a', 'english': 'Big'},
    {'aymara': 'Mallku', 'english': 'Condor'},
    {'aymara': 'Jiska', 'english': 'Moon'},
    {'aymara': 'Pacha', 'english': 'World'},
    {'aymara': 'Illimani', 'english': 'Mountain'},
    {'aymara': 'Llaki', 'english': 'Love'},
    {'aymara': 'Q\'illu', 'english': 'White'},
    {'aymara': 'Pirwa', 'english': 'Cloud'},
    {'aymara': 'Inti', 'english': 'Fire'},
    {'aymara': 'Nina', 'english': 'Child'},
    {'aymara': 'Suma', 'english': 'Water'},
    {'aymara': 'Ch\'allay', 'english': 'Goodbye'},
    {'aymara': 'Puma', 'english': 'Cougar'},
    {'aymara': 'Sara', 'english': 'Tooth'},
    {'aymara': 'Munay', 'english': 'To love'},
    {'aymara': 'Q\'ara', 'english': 'Black'},
    {'aymara': 'Apu', 'english': 'Lord'}
]

# Create the Fields
aymara_field = Field(tokenizer=aymara_tokenizer)
english_field = Field(tokenizer=english_tokenizer)

# Build the vocabularies
aymara_field.build_vocab([d['aymara'] for d in dataset])
english_field.build_vocab([d['english'] for d in dataset])

# Tokenize Aymara sentences
aymara_sentences = [aymara_field.process(d['aymara']) for d in dataset]
