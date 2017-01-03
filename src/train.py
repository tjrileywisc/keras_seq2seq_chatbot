
import spacy
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Flatten, Embedding
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers.core import RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed


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

    max_rank = MAX_NB_WORDS
    # add 1 to array so we can handle <UNK> words
    vectors = np.ndarray((max_rank + 1, vocab.vectors_length), dtype='float32')
    for lex in vocab:
        if lex.has_vector and lex.rank < MAX_NB_WORDS:
            vectors[lex.rank] = lex.vector
    return vectors

def get_features(docs):
    """
    turn input docs into onehot encoded vectors
    """
    Xs = np.zeros((1, MAX_SEQ_LEN), dtype='int32')
    for j, token in enumerate(docs[:MAX_SEQ_LEN]):
            Xs[0, j] = token.rank if token.has_vector and token.rank < MAX_NB_WORDS else 0
    return Xs

    # if we received several docs
    # need to force evaluation of nlp generator
    docs = list(docs)
    Xs = np.zeros((len(docs), MAX_SEQ_LEN), dtype='int32')
    for i, doc in enumerate(docs):
        for j, token in enumerate(doc[:MAX_SEQ_LEN]):
            Xs[i, j] = token.rank if token.has_vector and token.rank < MAX_NB_WORDS else 0
    return Xs

def targets_to_categorical(targets_arr):
    """
    each target instance is a 1 x timesteps array,
    where each entry in the array is an integer < MAX_NB_WORDS+1.

    the output from the network will be 1 x timesteps x MAX_NB_WORDS+1,
    so we need to one-hot encode /each/ timestep, so we have

    batch_size x timesteps x MAX_NB_WORDS+1
    """
    Xs_categorical = np.zeros((targets_arr.shape[0], targets_arr.shape[1], MAX_NB_WORDS+1), dtype = 'int16')
    for i in range(targets_arr.shape[0]):
        for j in range(targets_arr.shape[1]):
            word_index = targets_arr[i, j]
            Xs_categorical[i, j, word_index] = 1

    return Xs_categorical

def data_gen():
    """ 
    returns nb_samples x nb_timesteps (for X data, each entry is a word integer)
    returns nb_samples x nb_timesteps x nb_words (for y data, each entry one-hot encode of word integer)
    """

    while True:
        with open("./data/train.txt") as texts:
            for line in texts:
                line = line.replace('\n', "")
                if line != "----":
                    train_instance = line.split("----$----")
                    train_X = get_features(nlp(train_instance[0].lower()))
                    train_y = targets_to_categorical(get_features(nlp(train_instance[1].lower())))
                yield (train_X, train_y)
            


# # load chat logs data
# train_texts = []
# train_targets = []
# with open("./data/train.txt") as texts:

#     for line in texts:
#         line = line.replace('\n', "")
#         if line != "----":
#             train_instance = line.split("----$----")
#             train_texts.append(train_instance[0])
#             train_targets.append(train_instance[1])

# assert len(train_texts) == len(train_targets)

# load word embeddings
# populate training X and y
nlp = spacy.load('en')

embeddings = get_embeddings(nlp.vocab)

#train_X = np.array((get_features(nlp.pipe(train_texts), MAX_SEQ_LEN)))
#train_y = np.array((get_features(nlp.pipe(train_targets), MAX_SEQ_LEN)))

data_gen()

sequence_input = Input(shape = (MAX_SEQ_LEN,), dtype='int32')
embedded_sequences = Embedding(embeddings.shape[0],
                            EMBEDDING_DIM, weights = [embeddings],
                            input_length = MAX_SEQ_LEN, trainable = False)(sequence_input)

encoder = LSTM(128, return_sequences = True, activation = 'relu')(embedded_sequences)
encoder = LSTM(128, activation = 'relu')(encoder)

# the current input so far encoded as a single vector (repeated once for every output word)
thought_vector = RepeatVector(MAX_SEQ_LEN)(encoder)

decoder = LSTM(128, return_sequences = True, activation = 'relu')(thought_vector)
decoder = LSTM(128, return_sequences = True, activation = 'relu')(decoder)

# at every decoder step, output a prediction (a word)
# TODO: this might not go well due to the softmax over many (100k!) words
preds = TimeDistributed(Dense(embeddings.shape[0], activation = 'softmax'))(decoder)


model = Model(input = sequence_input, output = preds)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit_generator(data_gen(), samples_per_epoch = 10000, nb_epoch = 10)



