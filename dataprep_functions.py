# This file contains functions to do with cleaning, filtering and setting up the dataset.

# Possible extensions: Nothing much. We could think about other ways to handle punctuation. 
#


######################################

seed = 42
PAD_token = 0
SOS_token = 1
EOS_token = 2

######################################


import random
import re
import unicodedata
import numpy as np
import torch
from torch.utils.data import Dataset


#rng = np.random.default_rng(seed)
random.seed(seed)
torch.manual_seed(seed)


class Lang:
    # A class for initiating a language and a language index mapping tokens to integers.
    #

    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        # The below counts unique words:
        self.n_words = 3
        self.trimmed = False

    def index_word(self, word):
        # This function adds a new word to the class, updating the word2index, word2count, index2word, n_words.
        if word not in self.word2index:
            # assign the next token index available
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def index_words(self, sentence):
        # This indexes all words in a sentence.
        for word in sentence.split(' '):
            self.index_word(word)

    def trim(self, min_count):
        # Remove rare words that lie below a certain count threshold.
        # This can only be called once on the class.
        if self.trimmed: return

        # Set the class variable to True
        self.trimmed = True
        # Now trim.

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('Words kept: %s out of %s, a fraction of %.2f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Re-initialize dictionaries
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count default tokens

        # So the count statistics get lost after the trim operation!
        # And tokens get reassigned.
        for word in keep_words:
            self.index_word(word)


def unicode_to_ascii(s):
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    # Lowercase, trim, and remove non-letter characters.
    s = unicode_to_ascii(s.lower().strip())
    # A decision is made here to keep 4 punctuation marks only.
    s = re.sub(r"([,.!?])", r" \1 ", s)
    
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def read_langs(lang1, lang2, path, reverse=False):
    # This function reads in the data and initiates two Lang class instances.

    print("Reading file...")

    # Read the file and split into lines
    filename = path + '/%s-%s.txt' % (lang1, lang2)

    lines = open(filename).read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    pairs = [i[:2] for i in pairs]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs



def filter_pairs(pairs, MIN_LENGTH, MAX_LENGTH):
    # Filter, based on character length.
    filtered_pairs = []
    for pair in pairs:
        if len(pair[0].split()) >= MIN_LENGTH and len(pair[0].split()) <= MAX_LENGTH \
            and len(pair[1].split()) >= MIN_LENGTH and len(pair[1].split()) <= MAX_LENGTH:
                filtered_pairs.append(pair)
    return filtered_pairs


def indexes_from_sentence(lang, sentence):
    # Return a list of indexes corresponding to the sentence, plus EOS
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def indexes_from_sentences(lang, sentences):
    # run indexes_from_sentence on multiple sentences
    return [indexes_from_sentence(lang, k) for k in sentences]


def pad_seq(seq, max_length):
    # Pad a seq with the PAD symbol up to some length
    assert len(seq) <= max_length
    seq += [PAD_token for _ in range(max_length - len(seq))]
    return seq


class LanguageDataset(Dataset):
    # This class takes a list of two-item lists as input, each item in there being a string.
    # We shuffle and do tokenization and padding.
    # And we return tuples.

    def __init__(self, pairs, input_lang, output_lang, seed):
        super().__init__()
        self.pairs = pairs
        # https://stackoverflow.com/questions/17873384/how-to-deep-copy-a-list
        random.Random(seed).shuffle(self.pairs)
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # index must accept an integer or a range
        lines = self.pairs[index]
        # therefore, this line makes sure you always deal with a 2D case
        lines = np.array(lines).reshape(-1, 2)
        input_strs = lines[:, 0]
        output_strs = lines[:, 1]
        # the below give lists of lists
        input_tokens = indexes_from_sentences(self.input_lang, input_strs)
        output_tokens = indexes_from_sentences(self.output_lang, output_strs)

        # In doing the padding, we consider the longest sequence in the bunch of interest and pad to that.
        input_lengths = [len(s) for s in input_tokens]
        input_tokens = [pad_seq(s, max(input_lengths)) for s in input_tokens]

        output_lengths = [len(s) for s in output_tokens]
        output_tokens = [pad_seq(s, max(output_lengths)) for s in output_tokens]

        # we do not really need to return output_lengths
        return ((torch.LongTensor(input_tokens).transpose(0, 1), input_lengths),
                (torch.LongTensor(output_tokens).transpose(0, 1), output_lengths))