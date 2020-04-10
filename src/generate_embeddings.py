import torch, torch.nn as nn
import pickle
import io

import vocabulary

with open('../word_vocab.pkl','rb') as picklefile:
        word_vocab = pickle.load(picklefile)

# ----------- Loaded vocab

with io.open("../../wiki-news-300d-1M.vec", 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
	n, d = map(int, fin.readline().split())
	ft_data = {}
	print(n,d)

	count_s =0
	count_b = 0
	for line in fin:
		tokens = line.rstrip().split(' ')
		ft_data[tokens[0]] = list(map(float, tokens[1:]))
		count_s += 1
		if count_s % 10000 == 0:
			count_b += 1
			print("Loading fasttext: ",count_b)
			count_s = 0

print("Loaded fasttext")
# ------------ Loaded Fasttext

vocab_size = word_vocab.size
word_dim = 300

word_embedding_weights = torch.rand(vocab_size,word_dim)

for wd in word_vocab.indices:
	if wd in ft_data:
		word_embedding_weights[word_vocab.indices[wd]] =torch.tensor(ft_data[wd])

word_embeddings = nn.Embedding.from_pretrained(word_embedding_weights)
torch.save(word_embeddings,'fasttext_init_embeddings.pt') 