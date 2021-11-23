from collections import Counter, OrderedDict

import torchtext as text
from read_json import load_data
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import vocab

EMBEDDING_DIM = 300
VOCAB_SIZE = 20000

# Load data from NFT json data
nft_names, pic_paths, texts, labels = load_data()


# Load English tokenizer, tagger, parser and NER
tokenizer = get_tokenizer("spacy", language="en")

# build the vocab
counter = Counter()
for i, line in enumerate(texts):
    counter.update(tokenizer(line))

if len(counter) > VOCAB_SIZE:
    ordered_dict = OrderedDict(counter.most_common()[:VOCAB_SIZE])
else:
    ordered_dict = OrderedDict(counter.most_common())

# Build the vocab
vocab = vocab(ordered_dict)

# insert special tokens and set default index to 'unknown'
vocab.insert_token("<PAD>", 0)
vocab.insert_token("<UNK>", 1)

# default index to return when out of vocab, idx 1 is leading to <UNK>
vocab.set_default_index(1)

# Load 300-dim GloVe embeddings
vec = text.vocab.GloVe(name="840B", dim=300)
# create the embedding matrix, a torch tensor in the shape (num_words+1, embedding_dim)
word_emb = vec.get_vecs_by_tokens(vocab.get_itos())
print(word_emb.shape)  # words -> vector
