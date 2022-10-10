from collections import Counter, defaultdict
from glob import glob
import os
import gc
from tqdm import tqdm
import numpy as np

# Pytorch
import torch
from torch.utils.data import Dataset


class Word2VecDataset(Dataset):
    def __init__(self, DATA_DIR, CONTEXT_SIZE, TYPE="cbow"):
        vocab, word_idx, idx_word, training_data = self._load_data(
            DATA_DIR, CONTEXT_SIZE, TYPE
        )
        self.vocab = vocab
        self.word_idx = word_idx  # word -> index
        self.idx_word = idx_word  # index -> word
        # (x, y)
        if TYPE == "skipgram":
            self.data = torch.tensor(training_data, dtype=torch.long)
        elif TYPE == "cbow":
            self.data = [
                [torch.tensor(x[0]), torch.tensor(x[1])] for x in training_data
            ]
        else:
            self.data = [
                [[torch.tensor(x[0]), torch.tensor(x[1])], torch.tensor(x[2])]
                for x in training_data
            ]

    def __getitem__(self, i):
        x = self.data[i][0]
        y = self.data[i][1]
        return x, y

    def __len__(self):
        return len(self.data)

    def _load_data(self, data_dir, context_size, type):
        # Load files
        txt_files = glob(os.path.join(f"{data_dir}/", "*.txt"))
        corpus = []
        print("Joining files to make a corpus")
        for txt in txt_files:
            with open(txt, "r") as f:
                lines = f.readlines()
                corpus += [line.split() for line in tqdm(lines, total=len(lines))]
                print(f"Added contents of {txt} to corpus")

        # get the frequency table
        (
            sent_list_tokenized_filtered,
            vocab,
            word_idx,
            idx_word,
        ) = self._gather_word_freqs(corpus)
        del corpus
        gc.collect()

        # get the training data
        training_data = self._gather_training_data(
            sent_list_tokenized_filtered, word_idx, vocab, context_size, type
        )
        del sent_list_tokenized_filtered
        gc.collect()
        return vocab, word_idx, idx_word, training_data

    def _gather_word_freqs(self, sent_list_tokenized):
        vocab = Counter()
        word_idx = {}
        idx_word = {}
        for sent in tqdm(
            sent_list_tokenized, total=len(sent_list_tokenized), desc="Creating vocab"
        ):
            for word in sent:
                if vocab.get(word, 0) == 0:
                    word_idx[word] = len(word_idx)
                    idx_word[len(idx_word)] = word
            vocab.update(sent)
        total = len(vocab)

        sampling_rate = 0.001
        for sent in tqdm(
            sent_list_tokenized, total=len(sent_list_tokenized), desc="Subsampling"
        ):
            for i, word in enumerate(sent):
                frac = vocab[word] / total
                prob = 1 - np.sqrt(sampling_rate / frac)
                sampling = np.random.sample()
                if sampling < prob:
                    del sent[i]
                    i -= 1

        return sent_list_tokenized, vocab, word_idx, idx_word

    def _gather_training_data(
        self, sent_list_tokenized, word_idx, vocab, context_size, type
    ):
        training_data = []
        coo_counts = Counter()

        for sent in tqdm(
            sent_list_tokenized,
            total=len(sent_list_tokenized),
            desc="Creating Training Data",
        ):
            indices = [word_idx[w] for w in sent]
            if type == "skipgram" or type == "glove":
                for i in range(len(indices)):
                    for j in range(
                        max(-context_size, 0), min(context_size + 1, len(indices))
                    ):
                        if i == j:
                            continue
                        training_data.append((indices[i], indices[j]))
            else:
                if len(indices) < (2 * context_size + 1):
                    continue
                for i in range(context_size, len(indices) - context_size):
                    context = []
                    for j in range(i - context_size, i + context_size + 1):
                        if i == j:
                            continue
                        context.append(indices[j])
                    training_data.append([context, indices[i]])

        if type == "glove":
            coo_counts.update(training_data)
            tokens = defaultdict(lambda: -1)
            for word, count in vocab.most_common(len(vocab)):
                if count >= 5:
                    tokens[word_idx[word]] = count
            training_data = [
                (w[0], w[1], count)
                for w, count in coo_counts.items()
                if tokens[w[0]] > 0 and tokens[w[1]] > 0
            ]
        return training_data
