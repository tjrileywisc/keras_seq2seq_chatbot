# A seq2seq based chatbot

## Requirements
- keras
- tensorflow
- spacy (for glove word vector embeddings)

## Data
Either bring your own, or use the provided dataset in the ./data folder to start with.
This dataset is from the (https://people.mpi-sws.org/~cristian/Cornell_Movie-Dialogs_Corpus.html)[Cornesll Movie Dialogs Corpus]. Some of
these seem to be from movie screenplays actually.

## Models

Several models are provided:
- seq2seq
- seq2seq with attention
- seq2seq with additional pos tags as another set of input features
(embeddings of input words are fed into one rnn with one-hot of pos tags fed into another rnn,
and then these two are merged before the thought vector)