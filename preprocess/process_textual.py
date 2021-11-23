import json
import os
import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torchtext as text
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import vocab as Vocab
from utils import load_text_data


def encode(seq_tokens, token_to_idx):
    seq_idx = []
    for token in seq_tokens:
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(seq_idx, idx_to_token, delim=None, stop_at_end=True):
    tokens = []
    for idx in seq_idx:
        tokens.append(idx_to_token[idx])
        if stop_at_end and tokens[-1] == "<END>":
            break
    if delim is None:
        return tokens
    else:
        return delim.join(tokens)


def encode_and_save(
    args, vocab, tokenizer: Callable, texts: List[str], output: str = "encoded_text.pb"
):
    ### TODO: this whole function require testing

    texts_encoded: List[List[int]] = []
    texts_lens: List[int] = []
    max_len: int = 0

    # encode
    for i, text in enumerate(texts):
        # tokenize the text
        tokens: List[str] = tokenizer(text)
        # encode text
        text_encoded: List[int] = encode(tokens, vocab)
        texts_encoded.append(text_encoded)

        encoded_len: int = len(text_encoded)
        texts_lens.append(encoded_len)
        max_len = max(max_len, encoded_len)

    # pad all texts to the same length (max_len)
    for encoded in texts_encoded:
        encoded.extend([0] * (max_len - len(encoded)))

    # save encoded texts
    texts_encoded: np.ndarray = np.asarray(texts_encoded, dtype=np.int32)
    texts_lens: np.ndarray = np.asarray(texts_lens, dtype=np.int32)

    obj = {
        "texts_encoded": texts_encoded,
        "texts_lens": texts_lens,
    }

    with open(output, "wb") as f:
        pickle.dump(obj, f)


def process_text_and_save(args, train=True):
    ### TODO: this whole function require testing

    EMBEDDING_DIM = 300
    VOCAB_SIZE = 20000

    # Load texts data from NFT json labels
    ids, texts = load_text_data()

    # Load English tokenizer, tagger, parser and NER
    tokenizer = get_tokenizer("spacy", language="en")

    if not train:
        ### Load vocab from disk
        # TODO
        ...
    else:
        ### Build vocab for training

        # build the vocab
        counter = Counter()
        for i, line in enumerate(texts):
            counter.update(tokenizer(line))

        if len(counter) > VOCAB_SIZE:
            ordered_dict = OrderedDict(counter.most_common()[:VOCAB_SIZE])
        else:
            ordered_dict = OrderedDict(counter.most_common())

        # Build the vocab
        vocab = Vocab(ordered_dict)

        # insert special tokens and set default index to 'unknown'
        vocab.insert_token("<PAD>", 0)
        vocab.insert_token("<UNK>", 1)

        # default index to return when out of vocab, idx 1 is leading to <UNK>
        vocab.set_default_index(1)

    ### TODO: the logic here is not done yet
    ### TODO: call encode_and_save or merge encode_and_save

    # # Load 300-dim GloVe embeddings
    # vec = text.vocab.GloVe(name="840B", dim=300)
    # # create the embedding matrix, a torch tensor in the shape (num_words+1, embedding_dim)
    # word_emb = vec.get_vecs_by_tokens(vocab.get_itos())
    # print(word_emb.shape)  # words -> vector
