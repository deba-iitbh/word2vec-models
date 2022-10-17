import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Word2VecDataset
from model import SkipGramModel, CBOWModel, GloVeModel
from tqdm import tqdm, trange
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pickle


def neg_sampling(freq_distr, num_neg_samples):
    freq_distr_norm = F.normalize(torch.Tensor(freq_distr).pow(0.75), dim=0)
    weights = torch.ones(len(freq_distr))
    for _ in trange(len(freq_distr)):
        for _ in range(num_neg_samples):
            neg_ix = torch.multinomial(freq_distr_norm, 1)[0]
            weights[neg_ix] += 1
    return weights


def train(
    DATA_DIR,
    CONTEXT_SIZE,
    TYPE,
    BATCH_SIZE,
    EMBEDDING_DIM,
    DEVICE,
    NEGATIVE_SAMPLES,
    NUM_EPOCHS,
    LR,
    MODEL_DIR,
):
    print("==DATA Loading==")
    # Loading normally, but needs a huge amount of RAM.
    train_dataset = Word2VecDataset(DATA_DIR, CONTEXT_SIZE, TYPE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=not True)
    vocab = train_dataset.vocab
    word_to_ix = train_dataset.word_idx
    ix_to_word = train_dataset.idx_word

    # Negative Subsampling
    if NEGATIVE_SAMPLES > 0 and TYPE != "glove":
        print("== Negative Sampling ==")
        word_freqs = np.array(list(vocab.values()))
        unigram_dist = word_freqs / sum(word_freqs)
        weights = neg_sampling(unigram_dist, NEGATIVE_SAMPLES)
    else:
        weights = torch.ones(len(vocab))
    weights = weights.to(DEVICE)

    # Model Parameters
    if TYPE == "skipgram":
        model = SkipGramModel(EMBEDDING_DIM, len(vocab)).to(DEVICE)
    elif TYPE == "cbow":
        model = CBOWModel(EMBEDDING_DIM, len(vocab)).to(DEVICE)
    else:
        model = GloVeModel(EMBEDDING_DIM, len(vocab), NEGATIVE_SAMPLES).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_function = nn.NLLLoss(weights)

    losses = []
    print("==Model Training==")
    tepoch = tqdm(range(NUM_EPOCHS), total=NUM_EPOCHS, unit="batch")
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        if TYPE == "glove":
            for (focal, context), counts in tqdm(train_loader, total=len(train_loader)):
                model.train()
                focal = focal.to(DEVICE)
                context = context.to(DEVICE)
                counts = counts.to(DEVICE)
                optimizer.zero_grad()
                loss = model(focal, context, counts)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        else:
            for context, target in tqdm(train_loader, total=len(train_loader)):
                model.train()
                context_var = context.to(DEVICE)
                target_var = target.to(DEVICE)
                optimizer.zero_grad()
                log_probs = model(context_var)
                loss = loss_function(log_probs, target_var)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "losses": losses,
            "word_to_ix": word_to_ix,
            "ix_to_word": ix_to_word,
        },
        f"{MODEL_DIR}/model_{TYPE}_neg{NEGATIVE_SAMPLES}.pth",
    )
    print(min(losses))
