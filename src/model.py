import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(SkipGramModel, self).__init__()
        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)  # linear layer

    def forward(self, input_word):
        embeds = self.embeddings_input(input_word)
        out = self.linear(embeds)  # nonlinear + projection
        log_probs = F.log_softmax(out, dim=1)  # softmax compute log probability
        return log_probs


class CBOWModel(nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super(CBOWModel, self).__init__()
        self.embeddings_input = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)  # linear layer

    def forward(self, input_word):
        embeds = self.embeddings_input(input_word)
        embeds = torch.sum(embeds, dim=1)
        out = self.linear(embeds)  # nonlinear + projection
        log_probs = F.log_softmax(out, dim=1)  # softmax compute log probability
        return log_probs


class GloVeModel(nn.Module):
    def __init__(self, embedding_size, vocab_size, x_max):
        super(GloVeModel, self).__init__()
        self.x_max = x_max
        self._focal_embeddings = nn.Embedding(vocab_size, embedding_size)
        self._context_embeddings = nn.Embedding(vocab_size, embedding_size)
        self._focal_biases = nn.Embedding(vocab_size, 1).type(torch.float64)
        self._context_biases = nn.Embedding(vocab_size, 1).type(torch.float64)

    def forward(self, focal_input, context_input, coocurrence_count):
        x_max = max(self.x_max, 1)
        focal_embed = self._focal_embeddings(focal_input) # embed layer
        context_embed = self._context_embeddings(context_input) # embed layer
        focal_bias = self._focal_biases(focal_input) # bias for each embedding
        context_bias = self._context_biases(context_input) # bias for each embedding

        # count weight factor
        weight_factor = torch.pow(coocurrence_count / x_max, 0.75)
        weight_factor[weight_factor > 1] = 1

        embedding_products = torch.sum(focal_embed * context_embed, dim=1)
        log_cooccurrences = torch.log(coocurrence_count)

        distance_expr = (
            embedding_products + focal_bias + context_bias + log_cooccurrences
        ) ** 2

        single_losses = weight_factor * distance_expr
        mean_loss = torch.mean(single_losses)
        return mean_loss
