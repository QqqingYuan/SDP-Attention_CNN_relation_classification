import numpy as np
import re
import itertools
from collections import Counter
import util

word_embedding_size = util.word_embedding_size

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    max_document_length = max([len(x.split(" ")) for x in x_text])
    x = np.ndarray(shape=(len(x_text),max_document_length,word_embedding_size),dtype=np.float32)
    for i in range(len(x_text)):
        x[i] = util.getSentence_matrix(x_text[i],max_document_length)

    # Generate labels
    positive_labels = [[0,1] for _ in positive_examples]
    negative_labels = [[1,0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x, y]


def load2_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open("./rt-polaritydata/rt-polarity.pos", "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open("./rt-polaritydata/rt-polarity.neg", "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    max_document_length = max([len(x.split(" ")) for x in x_text])
    x = np.ndarray(shape=(len(x_text),max_document_length,word_embedding_size),dtype=np.float32)
    for i in range(len(x_text)):
        x[i] = util.getSentence_matrix(x_text[i],max_document_length)

    # Generate labels
    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    len(data) is not divisible by batch_size .
    """
    data = np.array(data)
    data_size = len(data)
    if len(data) % batch_size == 0:
        num_batches_per_epoch = int(len(data)/batch_size)
    else:
        num_batches_per_epoch = int(len(data)/batch_size) + 1

    # train shuffle
    train_shuffle = np.ndarray(shape=(num_epochs, data_size), dtype=np.int32)
    # train_shuffle = np.load('train_shuffle_2.npy')
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # shuffle_indices = train_shuffle[epoch]
            shuffle_indices = np.random.permutation(np.arange(data_size))
            train_shuffle[epoch] = shuffle_indices
            if epoch == (num_epochs - 1):
                np.save('train_shuffle.npy', train_shuffle)

            shuffled_data = data[shuffle_indices]

        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

