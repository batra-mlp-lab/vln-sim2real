''' Utils for io, language, etc '''

import os
import sys
import re
import string
import json
import time
import math
from collections import Counter, defaultdict
import numpy as np


# padding, unknown word, end of sentence
base_vocab = ['<PAD>', '<UNK>', '<EOS>']
padding_idx = base_vocab.index('<PAD>')



class Tokenizer(object):
    ''' Class to tokenize and encode a sentence. '''
    SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)') # Split on any non-alphanumeric character
  
    def __init__(self, vocab=None, encoding_length=20):
        self.encoding_length = encoding_length
        self.vocab = vocab
        self.word_to_index = {}
        self.index_to_word = {}
        if vocab:
            for i,word in enumerate(vocab):
                self.word_to_index[word] = i
            new_w2i = defaultdict(lambda: self.word_to_index['<UNK>'])
            new_w2i.update(self.word_to_index)
            self.word_to_index = new_w2i
            for key, value in self.word_to_index.items():
                self.index_to_word[value] = key
        old = self.vocab_size()
        self.add_word('<BOS>')
        assert self.vocab_size() == old+1
        print("OLD_VOCAB_SIZE", old)
        print("VOCAB_SIZE", self.vocab_size())
        print("VOACB", len(vocab))

    def finalize(self):
        """
        This is used for debug
        """
        self.word_to_index = dict(self.word_to_index)   # To avoid using mis-typing tokens

    def add_word(self, word):
        assert word not in self.word_to_index
        self.word_to_index[word] = self.vocab_size()    # vocab_size() is the
        self.index_to_word[self.vocab_size()] = word

    @staticmethod
    def split_sentence(sentence):
        ''' Break sentence into a list of words and punctuation '''
        toks = []
        for word in [s.strip().lower() for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
            # Break up any words containing punctuation only, e.g. '!?', unless it is multiple full stops e.g. '..'
            if all(c in string.punctuation for c in word) and not all(c in '.' for c in word):
                toks += list(word)
            else:
                toks.append(word)
        return toks

    def vocab_size(self):
        return len(self.index_to_word)

    def encode_sentence(self, sentence, max_length=None):
        if max_length is None:
            max_length = self.encoding_length
        if len(self.word_to_index) == 0:
            sys.exit('Tokenizer has no vocab')

        encoding = [self.word_to_index['<BOS>']]
        for word in self.split_sentence(sentence):
            encoding.append(self.word_to_index[word])   # Default Dict
        encoding.append(self.word_to_index['<EOS>'])

        if len(encoding) <= 2:
            return None
        #assert len(encoding) > 2

        if len(encoding) < max_length:
            encoding += [self.word_to_index['<PAD>']] * (max_length-len(encoding))  # Padding
        elif len(encoding) > max_length:
            encoding[max_length - 1] = self.word_to_index['<EOS>']                  # Cut the length with EOS

        return np.array(encoding[:max_length])

    def decode_sentence(self, encoding, length=None):
        sentence = []
        if length is not None:
            encoding = encoding[:length]
        for ix in encoding:
            if ix == self.word_to_index['<PAD>']:
                break
            else:
                sentence.append(self.index_to_word[ix])
        return " ".join(sentence)

    def shrink(self, inst):
        """
        :param inst:    The id inst
        :return:  Remove the potential <BOS> and <EOS>
                  If no <EOS> return empty list
        """
        if len(inst) == 0:
            return inst
        end = np.argmax(np.array(inst) == self.word_to_index['<EOS>'])     # If no <EOS>, return empty string
        if len(inst) > 1 and inst[0] == self.word_to_index['<BOS>']:
            start = 1
        else:
            start = 0
        # print(inst, start, end)
        return inst[start: end]


def read_vocab(path):
    with open(path) as f:
        vocab = [word.strip() for word in f.readlines()]
    return vocab


def quaternion_to_euler(q):

    x=q.x; y=q.y; z=q.z; w=q.w

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def angle_feature(heading, elevation, angle_feat_size):
    # twopi = math.pi * 2
    # heading = (heading + twopi) % twopi     # From 0 ~ 2pi
    # It will be the same
    return np.array([math.sin(heading), math.cos(heading),
                     math.sin(elevation), math.cos(elevation)] * (angle_feat_size // 4),
                    dtype=np.float32)



