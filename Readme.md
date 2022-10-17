# Wor2Vec Models
This repo contains the 3 popular word2Vec models implemented in pytorch.
Implemeted models:
- Skipgram
- Continious Bag of Words(CBOW)
- Global Vectors (GloVe)

## Dataset
We tried to train it o0n 10% of the latest Wiki Dump,  but were unable to process it, due to computational resources. Thus trained it on a small dataset, included in the repo.
We downloaded the Wiki Xml file and preprocessed it to .txt file using the extractor [script](data/Wiki/wikiex).

## Run the Code
```sh
cd src
python3 main.py
```

## Evaluation
The word vectors are evaluated on SimLex-999 dataset, as you can see in the [notebook](notebooks/evaluation.ipynb).  
The word vectors are also generated using this [notebook](notebooks/word_embedding.ipynb).  
The word vector visualization can be seen [here](notebooks/visualization.ipynb)


References:
- Wikipedia Cleaning [Script](https://towardsdatascience.com/pre-processing-a-wikipedia-dump-for-nlp-model-training-a-write-up-3b9176fdf67)
- https://jalammar.github.io/illustrated-word2vec/
- http://www.foldl.me/2014/glove-python/
- https://github.com/hans/glove.py
- https://github.com/noaRricky/pytorch-glove
- https://github.com/n0obcoder/Skip-Gram-Model-PyTorch

