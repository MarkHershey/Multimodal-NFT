import json
import os
import pickle
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torchtext as text
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab as Vocab
from utils import load_text_data


def encode(tokens: List[str], token_to_idx) -> List[int]:
    seq_idx = []
    for token in tokens:
        seq_idx.append(token_to_idx[token])
    return seq_idx


def decode(
    encoded_sequence: List[int],
    idx_to_token: dict,
    delim: str = None,
) -> List[str]:
    decoded_tokens = []
    for idx in encoded_sequence:
        decoded_tokens.append(idx_to_token[idx])

    if delim is None:
        return decoded_tokens
    elif isinstance(delim, str):
        return delim.join(decoded_tokens)


def process_text_and_save(args, train=True):

    EMBEDDING_DIM = 300
    VOCAB_SIZE = 20000

    # Load texts data from NFT json labels
    ids, texts = load_text_data()

    # Load English tokenizer
    tokenizer: Callable = get_tokenizer("spacy", language="en")

    ### Load vocab from disk for testing or build vocab for training
    if not train:
        # Load vocab from disk
        # TODO
        ...
    else:
        # Build the vocab
        counter = Counter()
        for i, line in enumerate(texts):
            counter.update(tokenizer(line))

        if len(counter) > VOCAB_SIZE:
            ordered_dict = OrderedDict(counter.most_common()[:VOCAB_SIZE])
        else:
            ordered_dict = OrderedDict(counter.most_common())

        vocab = Vocab(ordered_dict)

        # insert special tokens and set default index to 'unknown'
        vocab.insert_token("<PAD>", 0)
        vocab.insert_token("<UNK>", 1)

        # default index to return when out of vocab, idx 1 is leading to <UNK>
        vocab.set_default_index(1)

    # encode texts using vocab
    texts_encoded: List[List[int]] = []
    texts_lens: List[int] = []
    max_len: int = 0

    # encode
    for i, t in enumerate(texts):
        # tokenize the text
        tokens: List[str] = tokenizer(t)
        # encode text
        text_encoded: List[int] = encode(tokens=tokens, token_to_idx=vocab)
        texts_encoded.append(text_encoded)

        encoded_len: int = len(text_encoded)
        texts_lens.append(encoded_len)
        max_len = max(max_len, encoded_len)

    # pad all texts to the same length (max_len)
    for encoded in texts_encoded:
        encoded.extend([0] * (max_len - len(encoded)))

    # convert lists to numpy array
    texts_encoded: np.ndarray = np.asarray(texts_encoded, dtype=np.int32)
    texts_lens: np.ndarray = np.asarray(texts_lens, dtype=np.int32)

    if train:
        # Load the pretrained GloVe embeddings
        vec = text.vocab.GloVe(name="840B", dim=EMBEDDING_DIM)
        # get all tokens from vocab
        all_tokens: List[str] = vocab.get_itos()  # len = num_words
        # get embeddings for all tokens
        embedding_matrix = vec.get_vecs_by_tokens(
            all_tokens
        )  # (num_words, embedding_dim)
    else:
        # load from disk later
        embedding_matrix = None

    obj = {
        "texts_encoded": texts_encoded,
        "texts_lengths": texts_lens,
        "embedding_matrix": embedding_matrix.numpy(),
    }

    with open("encoded_text.pb", "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    process_text_and_save(args=None)
