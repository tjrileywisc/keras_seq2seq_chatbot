
# pylint: disable=C0103

import spacy
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Embedding
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, TimeDistributedDense, Dropout, Activation


# sequence length; shorter sequences are padded with zeros
MAX_SEQ_LEN = 20
# default from glove embedding vectors
EMBEDDING_DIM = 300
# limits to number of words
MAX_NB_WORDS = 10000
# train/dev split
DEV_SPLIT = 0.1


def get_embeddings(vocab):
    """
    get embeddings from spacy's glove vectors
    """

    max_rank = max(lex.rank for lex in vocab if lex.has_vector)
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors

def get_features(docs, max_length):
    """
    turn input docs into onehot encoded vectors
    """
    Xs = np.zeros((len(list(docs)), max_length), dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc):
            Xs[i, j] = token.rank if token.has_vector else 0
    return Xs
        

# load chat logs data
train_texts = []
train_targets = []
with open("./data/train.txt") as texts:

    for line in texts:
        line = line.replace('\n', "")
        if line != "----":
            train_instance = line.split("----$----")
            train_texts.append(train_instance[0])
            train_targets.append(train_instance[1])

assert len(train_texts) == len(train_targets)

# load word embeddings
# populate training X and y
nlp = spacy.load('en')

embeddings = get_embeddings(nlp.vocab)

train_X = np.array((get_features(nlp.pipe(train_texts), MAX_SEQ_LEN)))

sequence_input = Input(shape = (MAX_SEQ_LEN,), dtype='int32')
embedded_sequences = Embedding(embeddings.shape[0],
                            EMBEDDING_DIM, weights = [embeddings],
                            input_length = MAX_SEQ_LEN, trainable = False)(sequence_input)

encoder = LSTM(128, return_sequences = True)(embedded_sequences)
encoder = LSTM(128)(encoder)

repeat = RepeatVector(MAX_SEQ_LEN)(encoder)

model = Model(input = sequence_input, output = repeat)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.predict(train_X)



