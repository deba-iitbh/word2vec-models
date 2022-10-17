#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import os
from glob import glob
from tqdm import tqdm
import pickle

def data_prep(DATA_DIR):
    txt_files = glob(os.path.join(f"{DATA_DIR}/", "*.txt"))
    lem = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    stop_words.update(list(string.punctuation)) # punctutation
    stop_words.update(["''", "``"]) # extra based on corpus

    print(txt_files)

    for txt in txt_files:
        proc_file = None

        with open(txt, "r") as f:
            proc_lines = []
            lines = f.readlines()
            f.seek(0)
            for line in tqdm(lines, total = len(lines)):
                s_line = sent_tokenize(line)
                for l in s_line:
                    proc_lines.append(" ".join([lem.lemmatize(w.lower(), 'v') for w in word_tokenize(l) if w not in stop_words]))
            proc_file = "\n".join(proc_lines)

        with open(txt, "w") as f:
            f.write(proc_file)